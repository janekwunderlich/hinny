//! Name search: one SIMD scan (memchr's memmem, NEON on Apple silicon) over the folded names, split
//! across all cores, then a ranking per match. There's no index to keep in sync: at ~150 MB of names
//! the scan itself is a few milliseconds.

use memchr::memmem;
use rayon::prelude::*;

use crate::fold::fold;
use crate::store::{Index, Traits};

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub index: usize,
    pub score: f32,
}

impl Hit {
    /// Higher score first; equal scores by entry number, so results are the same whichever way
    /// they were found and don't flicker between keystrokes.
    #[inline]
    pub fn better(&self, other: &Hit) -> bool {
        self.score > other.score || (self.score == other.score && self.index < other.index)
    }

    pub fn order(a: &Hit, b: &Hit) -> std::cmp::Ordering {
        b.score.total_cmp(&a.score).then(a.index.cmp(&b.index))
    }
}

/// Best `limit` entries whose name contains every word of `query` (folded: case and accents ignored).
/// `now` is seconds since 1970, for recency.
pub fn search(index: &Index, query: &str, limit: usize, now: u32) -> Vec<Hit> {
    let folded = fold(query.trim().as_bytes());
    let terms: Vec<&[u8]> = folded.split(|&b| b == b' ').filter(|t| !t.is_empty()).collect();
    if terms.is_empty() || limit == 0 {
        return Vec::new();
    }
    // Scan for the longest word (fewest matches); check the others on each match.
    let main = *terms.iter().max_by_key(|t| t.len()).unwrap();
    let others: Vec<&[u8]> = {
        let mut skipped = false;
        terms.iter().copied().filter(|t| if !skipped && *t == main { skipped = true; false } else { true }).collect()
    };
    let query = Query { main, others: &others, whole: terms.len() == 1, now, limit };

    // (The original engine, kept for comparison: rayon, a full scan per keystroke. `fast::Searcher` is
    // what the app uses.) Many more chunks than cores: matches cluster (whole folders of "node_modules"), so even
    // chunks of entries aren't even chunks of work; small ones balance by stealing.
    let chunks = rayon::current_num_threads() * 16;
    let per = index.count.div_ceil(chunks).max(1);
    let mut hits: Vec<Hit> = (0..chunks)
        .into_par_iter()
        .map(|c| scan(index, c * per, ((c + 1) * per).min(index.count), &query, &mut None))
        .flatten()
        .collect();
    // Entries added by live updates since the file was written: few, checked one by one.
    for (i, added) in index.added.iter().enumerate() {
        if added.deleted {
            continue;
        }
        if let Some(at) = memmem::find(&added.folded, main) {
            let e = index.count + i;
            if let Some(score) = rank(index, e, &added.folded, at, &query) {
                hits.push(Hit { index: e, score });
            }
        }
    }
    let mut top = Top::new(limit);
    alias_hits(index, &query, &mut top);
    hits.extend(top.into_vec());
    finish(hits, limit)
}

/// Best first, one hit per entry (an entry can match by name and by alias), `limit` of them.
pub(crate) fn finish(mut hits: Vec<Hit>, limit: usize) -> Vec<Hit> {
    hits.sort_unstable_by(Hit::order);
    let mut seen = std::collections::HashSet::with_capacity(hits.len());
    hits.retain(|h| seen.insert(h.index));
    hits.truncate(limit);
    hits
}

/// Where `word` starts a word in `name` (the start, or after a non-alphanumeric byte).
pub(crate) fn word_start(name: &[u8], word: &[u8]) -> Option<usize> {
    memmem::find_iter(name, word).find(|&p| p == 0 || !name[p - 1].is_ascii_alphanumeric())
}

/// Localized display names ("Dokumente" for Documents): few, checked one by one, matched like names.
pub(crate) fn alias_hits(index: &Index, q: &Query, top: &mut Top) {
    for (e, alias) in &index.aliases {
        let e = *e as usize;
        if e >= index.total() || index.is_deleted(e) {
            continue;
        }
        let at = match q.main.len() {
            1 => (alias.first() == Some(&q.main[0])).then_some(0),
            2 => word_start(alias, q.main),
            _ => memmem::find(alias, q.main),
        };
        if let Some(at) = at {
            if let Some(score) = rank(index, e, alias, at, q) {
                top.push(Hit { index: e, score });
            }
        }
    }
}

pub(crate) struct Query<'a> {
    pub main: &'a [u8],
    pub others: &'a [&'a [u8]],
    pub whole: bool,
    pub now: u32,
    pub limit: usize,
}

/// Every match's id, for refining the next keystroke (`fast::Searcher`); None once it would exceed
/// the cap (a common query's matches aren't worth keeping: the next one scans again).
pub(crate) type Collect = Option<(Vec<u32>, usize)>;

/// Scans entries `start..end` and keeps their best `limit` matches (and, with `collect`, every match's id).
pub(crate) fn scan(index: &Index, start: usize, end: usize, q: &Query, collect: &mut Collect) -> Vec<Hit> {
    let mut top = Top::new(q.limit);
    if start >= end {
        return top.into_vec();
    }
    let fold_off = unsafe { std::slice::from_raw_parts(index.fold_off, index.count) };
    let lo = fold_off[start] as usize;
    let hi = if end < index.count { fold_off[end] as usize } else { index.folded_bytes };
    let hay = unsafe { std::slice::from_raw_parts(index.folded.add(lo), hi - lo) };

    if q.main.len() == 1 {
        // One letter: only names that start with it (every name containing an "e" isn't useful).
        let c = q.main[0];
        for e in start..end {
            let off = fold_off[e] as usize - lo;
            if hay[off] == c {
                consider(index, e, 0, q, &mut top, collect);
            }
        }
        return top.into_vec();
    }

    let finder = memmem::Finder::new(q.main);
    // Two letters match millions of names: only word starts count then ("in" → "Invoice", "my_inbox",
    // not "Training"), which is also what someone typing the start of a word means.
    let word_starts_only = q.main.len() <= 2;
    // One pass over every occurrence, moving through the entries alongside it (restarting a search
    // after each hit cost ~4 ms for "in", which occurs mid-word in millions of names).
    let mut e = start;
    let mut taken = usize::MAX;
    for p in finder.find_iter(hay) {
        // Move to the entry holding p: step for nearby ones, binary search across a long gap.
        if e + 1 < end && fold_off[e + 1] as usize - lo <= p {
            if e + 32 < end && fold_off[e + 32] as usize - lo <= p {
                e += fold_off[e..end].partition_point(|&o| o as usize - lo <= p) - 1;
            } else {
                while e + 1 < end && fold_off[e + 1] as usize - lo <= p {
                    e += 1;
                }
            }
        }
        if e == taken {
            continue; // one hit per entry
        }
        let at = p - (fold_off[e] as usize - lo);
        if word_starts_only && at > 0 && hay[p - 1].is_ascii_alphanumeric() {
            continue;
        }
        consider(index, e, at, q, &mut top, collect);
        taken = e;
    }
    top.into_vec()
}

#[inline]
pub(crate) fn consider(index: &Index, e: usize, at: usize, q: &Query, top: &mut Top, collect: &mut Collect) {
    if index.is_deleted(e) {
        return;
    }
    let name = index.folded_name(e);
    if let Some(score) = rank(index, e, name, at, q) {
        top.push(Hit { index: e, score });
        if let Some((ids, cap)) = collect {
            if ids.len() < *cap {
                ids.push(e as u32);
            } else {
                *collect = None;
            }
        }
    }
}

/// The score for entry `e` whose folded `name` contains the main word at `at`; None if another word is missing.
pub(crate) fn rank(index: &Index, e: usize, name: &[u8], first: usize, q: &Query) -> Option<f32> {
    if !q.others.iter().all(|t| memmem::find(name, t).is_some()) {
        return None;
    }
    let qlen = q.main.len();
    // The best place the word occurs: the start of the name, then the start of a word in it.
    let mut at = first;
    if at > 0 && name[at - 1].is_ascii_alphanumeric() {
        if let Some(word_start) = memmem::find_iter(name, q.main).find(|&p| p == 0 || !name[p - 1].is_ascii_alphanumeric()) {
            at = word_start;
        }
    }
    let t = index.traits(e);
    let mut s = 100.0f32;
    if at == 0 {
        s += 40.0;
    } else if !name[at - 1].is_ascii_alphanumeric() {
        s += 25.0;
    }
    // The whole name, or all of it but the extension ("lease" → "Lease.pdf").
    let stem = name.iter().rposition(|&b| b == b'.').filter(|&d| d > 0).unwrap_or(name.len());
    // (Not for code: "screenshot.ts" in a project shouldn't beat the screenshots.)
    if q.whole && (name.len() == qlen || (at == 0 && stem == qlen)) && !t.has(Traits::CODE) && !t.has(Traits::CODE_FILE) {
        s += 50.0;
    }
    s -= 0.4 * name.len().saturating_sub(qlen) as f32;
    s -= 2.0 * index.depth(e) as f32;
    if t.has(Traits::APPLICATION) {
        s += 45.0;
    }
    if t.has(Traits::DIRECTORY) {
        s += 5.0;
    }
    if t.has(Traits::DOCUMENT) {
        s += 10.0;
    }
    if t.has(Traits::HIDDEN) {
        s -= 40.0;
    }
    if t.has(Traits::LIBRARY) {
        s -= 25.0;
    }
    if t.has(Traits::CODE) {
        s -= 35.0;
    }
    if t.has(Traits::IN_PACKAGE) {
        s -= 35.0;
    }
    if t.has(Traits::CODE_FILE) {
        s -= 20.0;
    }
    let age = q.now.saturating_sub(index.modified(e));
    if age < 30 * 86_400 {
        s += 8.0;
    } else if age < 365 * 86_400 {
        s += 3.0;
    }
    Some(s)
}

/// The best `limit` hits seen so far (small: a linear minimum beats a heap here).
pub(crate) struct Top {
    hits: Vec<Hit>,
    limit: usize,
    min: usize,
}

impl Top {
    pub(crate) fn new(limit: usize) -> Top {
        Top { hits: Vec::with_capacity(limit), limit, min: 0 }
    }

    #[inline]
    pub(crate) fn push(&mut self, hit: Hit) {
        if self.hits.len() < self.limit {
            self.hits.push(hit);
            if self.hits.len() == self.limit {
                self.refresh();
            }
        } else if hit.better(&self.hits[self.min]) {
            self.hits[self.min] = hit;
            self.refresh();
        }
    }

    fn refresh(&mut self) {
        // The worst hit: lowest score, and among equal scores the highest entry number.
        self.min = (0..self.hits.len()).max_by(|&a, &b| Hit::order(&self.hits[a], &self.hits[b])).unwrap_or(0);
    }

    pub(crate) fn into_vec(self) -> Vec<Hit> {
        self.hits
    }
}
