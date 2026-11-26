//! Top-k BM25 over a segment.
//!
//! OR queries (agents' long questions) use MaxScore (Turtle & Flood 1995) refined per block, as
//! Lucene's `MaxScoreBulkScorer` does: documents are taken in windows that end at a block boundary,
//! each list's bound in the window is the best of its blocks there, and the lists whose bounds add
//! up to no more than the current k-th score are "non-essential". A document only they contain
//! can't make the top k, so they're only probed for documents the others found, and only while the
//! score could still get in. A window where every list is non-essential is skipped without decoding.
//!
//! AND queries (people's 1–3 words) intersect by leapfrogging: each word's lists jump to the
//! highest document any other word is at.
//!
//! Scores are Tantivy's BM25 to the bit (k1 1.2, b 0.75, its idf and length table) for a query that
//! doesn't repeat a word (a repeated word counts as often as it's repeated, as in Lucene), so results can
//! be compared document for document. `exhaustive` scores every matching document, for checking
//! that pruning never changes the results.


use crate::bitpack::BLOCK;
use crate::segment::{ListView, Segment, TermInfo};
use crate::tokenize::tokens;

pub const DONE: u32 = u32::MAX;

/// Blocks decoded, for the benchmark's pruning figures.
#[cfg(feature = "bench")]
pub static DECODED_BLOCKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Candidates finished (scored by the essential lists, then the others probed), and probes of a
/// non-essential list, counted for benchmarks (`bench` feature).
pub static CANDIDATES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROBES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// For benchmarks (`bench` feature): 0 lets `term_at_a_time` decide, 1 never scores term at a
/// time, 2 always does; and each decision's inputs, (segment documents, [(list length, bound)]).
#[cfg(feature = "bench")]
pub static TERM_AT_A_TIME: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
#[cfg(feature = "bench")]
pub static PLANS: std::sync::Mutex<Vec<(u32, Vec<(u32, f32)>)>> = std::sync::Mutex::new(Vec::new());

/// A result: `doc` is the passage's number in its segment, with the segment's ordinal in the high
/// 32 bits when several are searched together.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub doc: u64,
    pub score: f32,
}

/// One (field, term) of a query in one segment, with its BM25 weight: idf × (k1 + 1) × boost.
/// The weight comes from whatever statistics the caller uses (the segment's own, or global ones
/// across segments); the segment only provides the postings.
#[derive(Clone, Copy, Debug)]
pub struct Clause {
    pub info: TermInfo,
    pub weight: f32,
}

/// Words, each matched in any of its fields (a group of clauses). `all`: every word must match.
#[derive(Clone, Debug, Default)]
pub struct Query {
    pub groups: Vec<Vec<Clause>>,
    pub all: bool,
    /// A word of an AND query isn't in the index: nothing can match.
    pub impossible: bool,
}

/// BM25's idf, as Tantivy and Lucene compute it.
pub fn idf(docs: u64, df: u64) -> f32 {
    let x = ((docs.saturating_sub(df)) as f32 + 0.5) / (df as f32 + 0.5);
    (1.0 + x).ln()
}

pub fn weight(docs: u64, df: u64, boost: f32) -> f32 {
    idf(docs, df) * (1.0 + crate::K1) * boost
}

impl Query {
    /// The words of `text`, each in every field of `fields` (field, boost), with the segment's own
    /// statistics.
    pub fn parse(segment: &Segment, text: &str, fields: &[(u8, f32)], all: bool) -> Query {
        let mut q = Query { all, ..Default::default() };
        let mut seen: Vec<String> = Vec::new();
        for word in tokens(text) {
            if seen.contains(&word) {
                continue;
            }
            let group: Vec<Clause> = fields
                .iter()
                .filter_map(|&(f, boost)| segment.lookup(f, &word).map(|info| Clause { info, weight: weight(segment.docs as u64, info.df as u64, boost) }))
                .collect();
            if group.is_empty() {
                q.impossible |= all;
            } else {
                q.groups.push(group);
            }
            seen.push(word);
        }
        q
    }

    /// Search as you type: the `max` most frequent words starting with `prefix`, as one OR group.
    /// Also returns those words (so another engine can be given exactly the same query).
    pub fn prefix(segment: &Segment, prefix: &str, fields: &[(u8, f32)], max: usize) -> (Query, Vec<String>) {
        let mut by_word: Vec<(String, u64, Vec<Clause>)> = Vec::new();
        for &(f, boost) in fields {
            for (word, info) in segment.with_prefix(f, prefix) {
                let clause = Clause { info, weight: weight(segment.docs as u64, info.df as u64, boost) };
                match by_word.iter_mut().find(|(w, ..)| *w == word) {
                    Some(entry) => {
                        entry.1 += info.df as u64;
                        entry.2.push(clause);
                    }
                    None => by_word.push((word, info.df as u64, vec![clause])),
                }
            }
        }
        by_word.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        by_word.truncate(max);
        let words = by_word.iter().map(|(w, ..)| w.clone()).collect();
        (Query { groups: vec![by_word.into_iter().flat_map(|(.., c)| c).collect()], all: false, impossible: false }, words)
    }
}

/// Where a search puts its results: the top k shared by every segment searched, the passages to
/// skip (deleted, or not passing the query's filter), and the segment's ordinal for the results'
/// numbers. Skipped passages never enter the top k, so they never raise the threshold pruning
/// works against: filtered results are as exact as unfiltered ones.
pub struct Sink<'b> {
    pub top: &'b mut TopK,
    /// Deletions and the view's sequence number: a passage deleted at or before it is skipped.
    pub deleted: Option<(&'b crate::bitmap::Deletes, u64)>,
    pub filter: Option<&'b dyn Fn(u32) -> bool>,
    pub ordinal: u32,
    /// Give up early when this says so (the query reports it).
    pub stop: &'b crate::stop::Stop,
    /// For a top k of items ([`TopK::items`]): each passage's item.
    pub key: Option<&'b dyn Fn(u32) -> u64>,
}

impl Sink<'_> {
    #[inline]
    pub fn threshold(&self) -> Option<f32> {
        self.top.threshold()
    }

    #[inline]
    fn offer(&mut self, score: f32, doc: u32) {
        if self.deleted.is_some_and(|(d, seq)| !d.visible(doc, seq)) || self.filter.is_some_and(|f| !f(doc)) {
            return;
        }
        let at = (self.ordinal as u64) << 32 | doc as u64;
        match self.key {
            Some(key) => self.top.offer_item(score, at, key(doc)),
            None => self.top.offer(score, at),
        }
    }
}

pub struct Searcher<'a> {
    segment: &'a Segment,
    caches: Vec<[f32; 256]>,
    /// Per field: scored by dot product (a sparse field: weight × the posting's weight) rather
    /// than BM25.
    linear: Vec<bool>,
}

impl<'a> Searcher<'a> {
    /// With the segment's own average lengths.
    pub fn new(segment: &'a Segment) -> Searcher<'a> {
        let caches = (0..segment.fields).map(|f| crate::norms::tf_cache(segment.average_length(f))).collect();
        Searcher { segment, caches, linear: vec![false; segment.fields] }
    }

    /// With length normalization from other statistics (global ones, across segments).
    pub fn with_caches(segment: &'a Segment, caches: Vec<[f32; 256]>) -> Searcher<'a> {
        let linear = vec![false; caches.len()];
        Searcher { segment, caches, linear }
    }

    /// With which fields are scored by dot product (sparse fields).
    pub fn linear(mut self, linear: Vec<bool>) -> Searcher<'a> {
        self.linear = linear;
        self
    }

    fn cursor(&self, c: &Clause) -> Cursor<'_> {
        let f = c.info.field as usize;
        Cursor::new(self.segment, c.info, c.weight, &self.caches[f], self.linear.get(f).copied().unwrap_or(false))
    }

    pub fn search(&self, q: &Query, k: usize) -> Vec<Hit> {
        let mut top = TopK::new(k);
        self.search_into(q, &mut Sink { top: &mut top, deleted: None, filter: None, ordinal: 0, stop: &crate::stop::Stop::NEVER, key: None });
        top.into_sorted()
    }

    /// How many of `groups` each document has a term of: for requiring some groups of a query
    /// (a text's words) but not others (sparse terms scored with them).
    pub fn count_groups(&self, groups: &[Vec<Clause>]) -> Groups {
        let mut out = Groups::new(self.segment.docs as usize);
        let (mut docs, mut tfs) = ([0u32; BLOCK], [0u32; BLOCK]);
        for (g, group) in groups.iter().enumerate() {
            for clause in group {
                let c = self.cursor(clause);
                for b in 0..c.blocks {
                    let n = c.list.decode(b, &mut docs, &mut tfs, false);
                    for &d in &docs[..n] {
                        out.mark(d as usize, g);
                    }
                }
            }
        }
        out
    }

    pub fn search_into(&self, q: &Query, sink: &mut Sink) {
        if q.impossible || q.groups.is_empty() || sink.top.k == 0 {
            return;
        }
        if self.term_at_a_time(q, sink.top.k) {
            return self.accumulate_into(q, sink);
        }
        let mut cursors: Vec<Cursor> = q.groups.iter().flatten().map(|c| self.cursor(c)).collect();
        if q.all && q.groups.len() > 1 {
            // Rarest word first: it decides where the others jump.
            let mut groups: Vec<Vec<usize>> = Vec::new();
            let mut at = 0;
            for g in &q.groups {
                groups.push((at..at + g.len()).collect());
                at += g.len();
            }
            groups.sort_by_key(|g| g.iter().map(|&i| cursors[i].list.info.df as u64).sum::<u64>());
            conjunction(&mut cursors, &groups, sink);
        } else {
            max_score(&mut cursors, sink);
        }
    }

    /// The postings of `q`'s lists here: what scoring without a filter could read at most.
    pub fn postings(&self, q: &Query) -> u64 {
        q.groups.iter().flatten().map(|c| c.info.df as u64).sum()
    }

    /// `search_into`, but only the passages of `among`, in their order: each list jumps to them,
    /// so the work is theirs (a block decoded per passage and list at most), not the lists'. For
    /// a filter few pass: a common word and a filter of a thousandth, a million passages, 2.1 →
    /// 0.1 ms. No pruning: with few passages there is little to prune.
    pub fn search_among(&self, q: &Query, among: &crate::bitmap::Bitmap, sink: &mut Sink) {
        if q.impossible || q.groups.is_empty() || sink.top.k == 0 {
            return;
        }
        let mut groups: Vec<Vec<Cursor>> = q.groups.iter().map(|g| g.iter().map(|c| self.cursor(c)).collect()).collect();
        for (n, d) in among.ones().enumerate() {
            if n % 1024 == 0 && sink.stop.stopped() {
                return;
            }
            let (mut score, mut matched) = (0f32, 0);
            for group in &mut groups {
                let mut any = false;
                for c in group.iter_mut() {
                    c.seek(d);
                    // Its block may hold `d`: only decoding says.
                    if c.lower_bound() > d {
                        continue;
                    }
                    c.ensure();
                    if c.doc() == d {
                        score += c.score();
                        any = true;
                    }
                }
                if q.all && !any {
                    matched = 0;
                    break;
                }
                matched += any as usize;
            }
            if matched > 0 {
                sink.offer(score, d);
            }
        }
    }

    /// Whether `search` scores `q` term at a time. That pays for every posting and every document
    /// of the segment (an array cleared, then scanned for the top k), so it beats MaxScore when the
    /// lists MaxScore can't skip cover much of the segment: learned sparse queries (FiQA p50
    /// 0.45 → 0.24 ms) and long ones (ArguAna). Counting every list instead, short queries with
    /// stopwords chose it and lost: FiQA, plain analyzer, p50 0.20 → 0.26 ms. The threshold was
    /// chosen on BEIR dev, train and test queries, Quora's 523,000 documents included
    /// (`examples/text_bench.rs`).
    pub fn term_at_a_time(&self, q: &Query, k: usize) -> bool {
        #[cfg(feature = "bench")]
        {
            let lists = q.groups.iter().flatten().map(|c| (c.info.df, self.cursor(c).bound())).collect();
            PLANS.lock().unwrap().push((self.segment.docs, lists));
            match TERM_AT_A_TIME.load(std::sync::atomic::Ordering::Relaxed) {
                1 => return false,
                2 => return !q.all,
                _ => {}
            }
        }
        let postings: u64 = q.groups.iter().flatten().map(|c| c.info.df as u64).sum();
        let docs = self.segment.docs as u64;
        if q.all || postings < docs {
            return false;
        }
        #[cfg(feature = "bench")]
        match TERM_AT_A_TIME.load(std::sync::atomic::Ordering::Relaxed) {
            3 => return true, // the rule before strong lists were counted, for comparison
            4 => return self.strong_postings(q) as f64 >= 0.3 * docs as f64 * (docs as f64 / 57_638.0).sqrt(),
            _ => {}
        }
        // A deep top k keeps MaxScore's threshold low, so it skips little: from FEW_RESULTS on, the
        // lists covering the segment is enough (a hybrid query's text side asks for 200).
        if k > FEW_RESULTS {
            return true;
        }
        // The share of the segment the strong lists must cover grows with its size: a third of
        // the documents at FiQA's 57,638, more in a larger segment, less in a smaller one.
        self.strong_postings(q) as f64 >= 0.3 * docs as f64 * (docs as f64 / 57_638.0).sqrt()
    }

    /// Postings of the lists pruning couldn't skip: lists in order of their score bound, leaving
    /// out the weakest while their bounds add up to less than half the strongest's (a document
    /// only they match can't come near the top, so MaxScore stops reading them early).
    fn strong_postings(&self, q: &Query) -> u64 {
        let mut lists: Vec<(f32, u64)> = q.groups.iter().flatten().map(|c| (self.cursor(c).bound(), c.info.df as u64)).collect();
        lists.sort_by(|a, b| a.0.total_cmp(&b.0));
        let strongest = lists.last().map_or(0.0, |l| l.0);
        let mut weak = 0.0;
        lists.iter().filter(|(bound, _)| {
            weak += bound;
            weak >= 0.5 * strongest
        }).map(|(_, df)| df).sum()
    }

    /// `accumulate_into`'s top k.
    pub fn accumulate(&self, q: &Query, k: usize) -> Vec<Hit> {
        let mut top = TopK::new(k);
        self.accumulate_into(q, &mut Sink { top: &mut top, deleted: None, filter: None, ordinal: 0, stop: &crate::stop::Stop::NEVER, key: None });
        top.into_sorted()
    }

    /// Every posting scored into a score per document, list after list (term at a time), then the
    /// top k taken from those. For queries whose lists together are as long as the segment, where
    /// MaxScore skips little and pays for each candidate: a learned sparse query (~13 terms over
    /// lists of thousands) left it 32,000 of FiQA's 57,638 documents to finish one by one.
    pub fn accumulate_into(&self, q: &Query, sink: &mut Sink) {
        if q.impossible || q.groups.is_empty() || sink.top.k == 0 {
            return;
        }
        thread_local! {
            static SCORES: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        SCORES.with(|scores| {
            let mut scores = scores.borrow_mut();
            scores.clear();
            scores.resize(self.segment.docs as usize, 0.0);
            let (mut docs, mut tfs) = ([0u32; BLOCK], [0u32; BLOCK]);
            for clause in q.groups.iter().flatten() {
                if sink.stop.stopped() {
                    return;
                }
                let c = self.cursor(clause);
                for b in 0..c.blocks {
                    let n = c.list.decode(b, &mut docs, &mut tfs, true);
                    if c.linear {
                        for (&d, &tf) in docs[..n].iter().zip(&tfs[..n]) {
                            scores[d as usize] += c.weight * (tf + 1) as f32;
                        }
                    } else {
                        for (&d, &tf) in docs[..n].iter().zip(&tfs[..n]) {
                            let tf = (tf + 1) as f32;
                            scores[d as usize] += c.weight * (tf / (tf + c.cache[c.norms[d as usize] as usize]));
                        }
                    }
                }
            }
            for (d, &score) in scores.iter().enumerate() {
                if score > 0.0 && sink.threshold().is_none_or(|t| score >= t) {
                    sink.offer(score, d as u32);
                }
            }
        })
    }

    /// Every matching document scored, nothing skipped (for verifying `search`).
    pub fn exhaustive(&self, q: &Query, k: usize) -> Vec<Hit> {
        let mut top = TopK::new(k);
        self.exhaustive_into(q, &mut Sink { top: &mut top, deleted: None, filter: None, ordinal: 0, stop: &crate::stop::Stop::NEVER, key: None });
        top.into_sorted()
    }

    pub fn exhaustive_into(&self, q: &Query, sink: &mut Sink) {
        if q.impossible || q.groups.is_empty() || sink.top.k == 0 {
            return;
        }
        let mut scores = vec![0f32; self.segment.docs as usize];
        // Per document, how many groups it has a term of; counted once per group (a group's terms
        // can share documents), so any number of groups can be required.
        let mut matched = Groups::new(self.segment.docs as usize);
        for (g, group) in q.groups.iter().enumerate() {
            for clause in group {
                let mut c = self.cursor(clause);
                c.ensure();
                while c.doc() != DONE {
                    scores[c.doc() as usize] += c.score();
                    matched.mark(c.doc() as usize, g);
                    c.next();
                }
            }
        }
        let need = if q.all { q.groups.len() as u32 } else { 1 };
        for d in 0..scores.len() {
            if matched.count(d) >= need {
                sink.offer(scores[d], d as u32);
            }
        }
    }
}

// MARK: Posting cursors

/// A position in one posting list. Blocks are decoded only when a document in them is needed;
/// term frequencies only when one is scored.
struct Cursor<'a> {
    list: ListView<'a>,
    blocks: usize,
    block: usize,
    decoded: bool,
    tfs_decoded: bool,
    /// When not decoded: the cursor is at the first document ≥ this in `block`.
    target: u32,
    docs: [u32; BLOCK],
    tfs: [u32; BLOCK],
    len: usize,
    pos: usize,
    weight: f32,
    cache: &'a [f32; 256],
    norms: &'a [u8],
    /// Scored by dot product: weight × frequency (a sparse field's weights).
    linear: bool,
}

impl<'a> Cursor<'a> {
    fn new(segment: &'a Segment, info: TermInfo, weight: f32, cache: &'a [f32; 256], linear: bool) -> Cursor<'a> {
        let list = ListView::new(segment.bytes(), info);
        Cursor {
            list,
            blocks: list.blocks(),
            block: 0,
            decoded: false,
            tfs_decoded: false,
            target: 0,
            docs: [0; BLOCK],
            tfs: [0; BLOCK],
            len: 0,
            pos: 0,
            weight,
            cache,
            norms: segment.norms(info.field as usize),
            linear,
        }
    }

    #[inline]
    fn last(&self, b: usize) -> u32 {
        self.list.last(b)
    }

    /// The best term-frequency factor in block `b`, from its impacts and these statistics.
    #[inline]
    fn factor(&self, b: usize) -> f32 {
        let i = self.list.impacts(b);
        if self.linear { i.max_tf() } else { i.bound(self.cache) }
    }

    #[inline]
    fn exhausted(&self) -> bool {
        self.block >= self.blocks
    }

    /// The current document (after `ensure`), or DONE.
    #[inline]
    fn doc(&self) -> u32 {
        if self.exhausted() { DONE } else { self.docs[self.pos] }
    }

    /// No document before this is left (without decoding).
    #[inline]
    fn lower_bound(&self) -> u32 {
        if self.exhausted() {
            DONE
        } else if self.decoded {
            self.docs[self.pos]
        } else {
            let start = if self.block == 0 { 0 } else { self.last(self.block - 1) + 1 };
            self.target.max(start)
        }
    }

    /// Moves to the first document ≥ `target`, decoding nothing new (headers only).
    #[inline]
    fn seek(&mut self, target: u32) {
        if self.lower_bound() >= target {
            return;
        }
        let mut b = self.block;
        while b < self.blocks && self.last(b) < target {
            b += 1;
        }
        if b != self.block {
            self.block = b;
            self.decoded = false;
            self.target = target;
        } else if self.decoded {
            while self.docs[self.pos] < target {
                self.pos += 1;
            }
        } else {
            self.target = target;
        }
    }

    /// Decodes the current block if a document in it is needed.
    #[inline]
    fn ensure(&mut self) {
        if self.decoded || self.exhausted() {
            return;
        }
        #[cfg(feature = "bench")]
        DECODED_BLOCKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let n = self.list.decode(self.block, &mut self.docs, &mut self.tfs, false);
        // Variable-byte blocks come with their frequencies.
        self.tfs_decoded = n < BLOCK;
        self.len = n;
        self.decoded = true;
        self.pos = self.docs[..n].partition_point(|&d| d < self.target);
    }

    #[inline]
    fn next(&mut self) {
        self.pos += 1;
        if self.pos == self.len {
            self.block += 1;
            self.decoded = false;
            self.target = 0;
            self.ensure();
        }
    }

    #[inline]
    fn score(&mut self) -> f32 {
        if !self.tfs_decoded {
            self.list.decode_tfs(self.block, &mut self.tfs);
            self.tfs_decoded = true;
        }
        let tf = (self.tfs[self.pos] + 1) as f32;
        if self.linear {
            return self.weight * tf;
        }
        let norm = self.cache[self.norms[self.docs[self.pos] as usize] as usize];
        self.weight * (tf / (tf + norm))
    }

    /// Best score this list can give a document in [current, `hi`]: the best of its blocks there.
    #[inline]
    fn bound_until(&self, hi: u32) -> f32 {
        if self.exhausted() {
            return 0.0;
        }
        let mut b = self.block;
        let mut best = self.factor(b);
        while b + 1 < self.blocks && self.last(b) < hi {
            b += 1;
            best = best.max(self.factor(b));
        }
        self.weight * best * SLACK
    }

    fn bound(&self) -> f32 {
        let i = &self.list.info.impacts;
        self.weight * (if self.linear { i.max_tf() } else { i.bound(self.cache) }) * SLACK
    }
}

/// Up to this many results, `term_at_a_time` counts only the lists MaxScore can't skip.
const FEW_RESULTS: usize = 32;

/// Bounds are summed in another order than scores: a hair above them keeps rounding from pruning
/// a document that belongs in the results.
const SLACK: f32 = 1.000_01;

/// How many of a query's groups each document has a term of, for "all words" over term-at-a-time
/// scoring: groups are added one after another, and a document counts once per group.
pub struct Groups {
    last: Vec<u32>,
    count: Vec<u32>,
}

impl Groups {
    pub fn new(docs: usize) -> Groups {
        Groups { last: vec![0; docs], count: vec![0; docs] }
    }

    /// Document `d` has a term of group `g` (groups in order).
    #[inline]
    pub fn mark(&mut self, d: usize, g: usize) {
        let g = g as u32 + 1;
        if self.last[d] != g {
            self.last[d] = g;
            self.count[d] += 1;
        }
    }

    #[inline]
    pub fn count(&self, d: usize) -> u32 {
        self.count[d]
    }
}

/// Documents scored together: the essential lists' scores for an inner window are added into a
/// buffer, list by list, then its documents are finished in order (Lucene's inner window).
const INNER: u32 = 4096;

fn max_score(cursors: &mut [Cursor], top: &mut Sink) {
    let n = cursors.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut ub = vec![0f32; n];
    // Lists before this in `order` are non-essential (from the last partition).
    let mut essential = 0;
    let mut buffer = vec![0f32; INNER as usize];
    let mut present = vec![0u64; INNER as usize / 64];
    loop {
        if top.stop.stopped() {
            return;
        }
        let lo = cursors.iter().map(|c| c.lower_bound()).min().unwrap_or(DONE);
        if lo == DONE {
            return;
        }
        // The window ends where an essential list's block does, but spans at least an inner window:
        // ending it at every list's next block made windows of a few hundred documents for common
        // words, each re-sorting every bound.
        let mut hi = DONE - 1;
        for &i in &order[essential.min(n - 1)..] {
            let c = &mut cursors[i];
            c.seek(lo);
            if !c.exhausted() {
                hi = hi.min(c.last(c.block));
            }
        }
        hi = hi.max(lo.saturating_add(INNER - 1)).min(DONE - 1);
        for (i, c) in cursors.iter_mut().enumerate() {
            c.seek(lo);
            ub[i] = c.bound_until(hi);
        }
        order.sort_unstable_by(|&a, &b| ub[a].total_cmp(&ub[b]));
        // Non-essential: the weakest lists, while their bounds add up to no more than the k-th score.
        let mut weak = 0f32;
        essential = 0;
        partition(&order, &ub, &mut essential, &mut weak, top.threshold());
        let mut inner = lo;
        while essential < n && inner <= hi {
            let inner_hi = inner.saturating_add(INNER - 1).min(hi);
            let before = top.threshold();
            if n - essential == 1 {
                // One essential list: walk it directly.
                let e = order[essential];
                loop {
                    cursors[e].ensure();
                    let d = cursors[e].doc();
                    if d > inner_hi {
                        break;
                    }
                    let score = cursors[e].score();
                    finish(cursors, &order[..essential], &ub, weak, d, score, top);
                    cursors[e].next();
                }
            } else {
                for &i in &order[essential..] {
                    let c = &mut cursors[i];
                    c.ensure();
                    while c.doc() <= inner_hi {
                        let at = (c.doc() - inner) as usize;
                        buffer[at] += c.score();
                        present[at / 64] |= 1 << (at % 64);
                        c.next();
                    }
                }
                for w in 0..present.len() {
                    let mut bits = present[w];
                    present[w] = 0;
                    while bits != 0 {
                        let at = w * 64 + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let score = std::mem::take(&mut buffer[at]);
                        finish(cursors, &order[..essential], &ub, weak, inner + at as u32, score, top);
                    }
                }
            }
            // A higher k-th score can make more lists non-essential for the rest of the window.
            if top.threshold() != before {
                partition(&order, &ub, &mut essential, &mut weak, top.threshold());
            }
            if inner_hi == DONE - 1 {
                return;
            }
            inner = inner_hi + 1;
        }
        if hi == DONE - 1 {
            return;
        }
        for c in cursors.iter_mut() {
            c.seek(hi + 1);
        }
    }
}

/// Moves lists from essential to non-essential while the non-essential bounds stay ≤ the k-th score.
fn partition(order: &[usize], ub: &[f32], essential: &mut usize, weak: &mut f32, theta: Option<f32>) {
    if let Some(t) = theta {
        while *essential < order.len() && *weak + ub[order[*essential]] <= t {
            *weak += ub[order[*essential]];
            *essential += 1;
        }
    }
}

/// Adds the non-essential lists' scores to a candidate, strongest first, while it could still get
/// into the top k; then offers it.
#[inline]
fn finish(cursors: &mut [Cursor], non_essential: &[usize], ub: &[f32], weak: f32, d: u32, mut score: f32, top: &mut Sink) {
    #[cfg(feature = "bench")]
    CANDIDATES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut rest = weak;
    for &i in non_essential.iter().rev() {
        if top.threshold().is_some_and(|t| score + rest <= t) {
            return;
        }
        let c = &mut cursors[i];
        #[cfg(feature = "bench")]
        PROBES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        c.seek(d);
        c.ensure();
        if c.doc() == d {
            score += c.score();
        }
        rest -= ub[i];
    }
    top.offer(score, d);
}

fn conjunction(cursors: &mut [Cursor], groups: &[Vec<usize>], top: &mut Sink) {
    let total: f32 = cursors.iter().map(|c| c.bound()).sum();
    let mut candidate = 0u32;
    'next: loop {
        if top.threshold().is_some_and(|t| total <= t) {
            return;
        }
        let mut g = 0;
        while g < groups.len() {
            let mut at = DONE;
            for &i in &groups[g] {
                cursors[i].seek(candidate);
                cursors[i].ensure();
                at = at.min(cursors[i].doc());
            }
            if at == DONE {
                return;
            }
            if at > candidate {
                candidate = at;
                // Every word again at the new candidate (the first group is already there if g == 0).
                g = if g == 0 { 1 } else { 0 };
                continue;
            }
            g += 1;
        }
        let mut score = 0f32;
        for group in groups {
            for &i in group {
                if cursors[i].doc() == candidate {
                    score += cursors[i].score();
                }
            }
        }
        top.offer(score, candidate);
        if candidate == DONE - 1 {
            return;
        }
        candidate += 1;
        continue 'next;
    }
}

// MARK: Top k

/// The best k so far: a min-heap on (score, then later document), so the root is the one to replace.
/// Or, made with [`TopK::items`], the best k items, each at its best passage.
pub struct TopK {
    pub k: usize,
    heap: Vec<Hit>,
    items: Option<Items>,
}

/// For a top k of items: each heap entry's item, and, while searching a source where an item can
/// have several passages, where that source's items are in the heap. (An item's live passages are
/// all in one segment or chunk, so sources with one passage an item need no lookups at all.)
struct Items {
    keys: Vec<u64>,
    at: std::collections::HashMap<u64, usize>,
    repeats: bool,
}

impl TopK {
    pub fn new(k: usize) -> TopK {
        TopK { k, heap: Vec::with_capacity(k.min(1024) + 1), items: None }
    }

    /// The best k items rather than passages: an item's passages compete for its one place, which
    /// it holds at its best (the lowest-numbered of equal ones), and items of equal scores are
    /// ordered by key. The threshold is the k-th item's, so an item's other passages can't crowd
    /// out the next item, however many score above it.
    pub fn items(k: usize) -> TopK {
        // Room grows with the results, not the limit asked for (which can be "all of them").
        let room = k.min(1024);
        TopK { k, heap: Vec::with_capacity(room + 1), items: Some(Items { keys: Vec::with_capacity(room), at: std::collections::HashMap::new(), repeats: true }) }
    }

    /// The next passages offered come from one segment or chunk, where an item has at most one
    /// passage unless `repeats`.
    pub fn source(&mut self, repeats: bool) {
        if let Some(items) = &mut self.items {
            items.at.clear();
            items.repeats = repeats;
        }
    }

    /// The k-th best score once there are k results; a document must beat it to get in. For items,
    /// just below it: ties go by key, not by the order documents are searched in, so a document
    /// scoring the same can still get in, and pruning may only skip what scores less.
    #[inline]
    pub fn threshold(&self) -> Option<f32> {
        (self.heap.len() == self.k).then(|| if self.items.is_some() { self.heap[0].score.next_down() } else { self.heap[0].score })
    }

    /// Whether entry `i` ranks below entry `j`.
    #[inline]
    fn worse(&self, i: usize, j: usize) -> bool {
        let (a, b) = (&self.heap[i], &self.heap[j]);
        a.score < b.score
            || (a.score == b.score
                && match &self.items {
                    Some(items) => items.keys[i] > items.keys[j],
                    None => a.doc > b.doc,
                })
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.heap.swap(i, j);
        if let Some(items) = &mut self.items {
            items.keys.swap(i, j);
            if !items.at.is_empty() {
                for i in [i, j] {
                    if let Some(at) = items.at.get_mut(&items.keys[i]) {
                        *at = i;
                    }
                }
            }
        }
    }

    fn up(&mut self, mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if !self.worse(i, p) {
                break;
            }
            self.swap(i, p);
            i = p;
        }
    }

    fn down(&mut self, mut i: usize) {
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut m = i;
            if l < self.heap.len() && self.worse(l, m) {
                m = l;
            }
            if r < self.heap.len() && self.worse(r, m) {
                m = r;
            }
            if m == i {
                return;
            }
            self.swap(i, m);
            i = m;
        }
    }

    pub fn offer(&mut self, score: f32, doc: u64) {
        debug_assert!(self.items.is_none(), "a top k of items is offered passages with their item");
        let hit = Hit { doc, score };
        if self.heap.len() < self.k {
            self.heap.push(hit);
            self.up(self.heap.len() - 1);
        } else if self.k > 0 && (self.heap[0].score < score || (self.heap[0].score == score && self.heap[0].doc > doc)) {
            self.heap[0] = hit;
            self.down(0);
        }
    }

    /// A passage of item `key` (for a top k made with [`TopK::items`]).
    pub fn offer_item(&mut self, score: f32, doc: u64, key: u64) {
        let Some(items) = &mut self.items else { return self.offer(score, doc) };
        if let Some(&i) = items.at.get(&key).filter(|_| items.repeats) {
            let h = self.heap[i];
            if score > h.score || (score == h.score && doc < h.doc) {
                self.heap[i] = Hit { doc, score };
                self.down(i);
            }
            return;
        }
        if self.heap.len() < self.k {
            items.keys.push(key);
            if items.repeats {
                items.at.insert(key, self.heap.len());
            }
            self.heap.push(Hit { doc, score });
            self.up(self.heap.len() - 1);
        } else if self.k > 0 && (self.heap[0].score < score || (self.heap[0].score == score && items.keys[0] > key)) {
            if items.repeats {
                items.at.remove(&items.keys[0]);
                items.at.insert(key, 0);
            }
            items.keys[0] = key;
            self.heap[0] = Hit { doc, score };
            self.down(0);
        }
    }

    /// Best first: by score, then document (or, for items, key).
    pub fn into_sorted(mut self) -> Vec<Hit> {
        match self.items {
            Some(items) => {
                let mut v: Vec<(Hit, u64)> = self.heap.into_iter().zip(items.keys).collect();
                v.sort_by(|a, b| b.0.score.total_cmp(&a.0.score).then(a.1.cmp(&b.1)));
                v.into_iter().map(|(h, _)| h).collect()
            }
            None => {
                self.heap.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc)));
                self.heap
            }
        }
    }
}
