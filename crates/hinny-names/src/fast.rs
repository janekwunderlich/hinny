//! The search as you type: each keystroke takes the cheapest way to its answer.
//!
//! 1. The same query as a moment ago (a backspace, a repeat): the remembered result.
//! 2. One or two letters: the word-start lists (`starts`), a few microseconds.
//! 3. A query that extends an earlier one ("inv" → "invo"): only that query's matches are checked
//!    again, plus entries added since, not all names. Its matches are a superset of the new query's
//!    whenever every earlier word is contained in a new one and the earlier main word had 3+ letters
//!    (shorter ones only matched word starts). fzf and nucleo do the same (nucleo's `append`).
//!    Changes don't invalidate this: deletions and dates are checked when re-scoring, and additions
//!    are appended to `index.added` in order, so the ones since are checked too. (Invalidating on
//!    every change threw the cache away constantly: ~/Library changes many times a second.)
//! 4. Otherwise a full scan, on a pool that parks between keystrokes (`pool`), remembering every
//!    match (up to a cap) for the next keystroke.

use std::sync::{Arc, Mutex};

use memchr::memmem;

use crate::fold::fold;
use crate::pool;
use crate::search::{Collect, Hit, Query, Top, alias_hits, finish, rank, scan, word_start};
use crate::starts::WordStarts;
use crate::store::Index;

/// Matches remembered per query; a query matching more is common enough that refining from it
/// wouldn't beat a fresh scan.
const REMEMBER: usize = 200_000;
/// Remembered queries (enough for a few backspaces).
const CACHED: usize = 8;
/// Below this many candidates a refinement runs on the calling thread (waking others costs more).
const REFINE_SINGLE: usize = 20_000;

/// How the last search was answered, for benchmarks and logs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Path {
    Repeat,
    WordStarts,
    Refined { candidates: usize },
    Scan,
}

struct Cached {
    folded: Vec<u8>,
    terms: Vec<Vec<u8>>,
    main_len: usize,
    /// Every match, if there weren't more than `REMEMBER`.
    ids: Option<Arc<Vec<u32>>>,
    hits: Vec<Hit>,
    limit: usize,
    /// Any change: a repeat is only reused within one.
    generation: u64,
    /// `index.added.len()` when cached: entries added after it are checked when refining.
    added: usize,
    build_id: u64,
}

pub struct Searcher {
    starts: Option<WordStarts>,
    cache: Vec<Cached>,
    pub last: Path,
}

impl Searcher {
    pub fn new(starts: Option<WordStarts>) -> Searcher {
        Searcher { starts, cache: Vec::new(), last: Path::Scan }
    }

    /// Opens the word-start lists that belong to `index` (none if missing or from another build).
    pub fn for_index(index: &Index, index_path: &std::path::Path) -> Searcher {
        Searcher::new(WordStarts::open(&WordStarts::path_for(index_path), index))
    }

    pub fn has_word_starts(&self) -> bool {
        self.starts.is_some()
    }

    pub fn search(&mut self, index: &Index, query: &str, limit: usize, now: u32) -> Vec<Hit> {
        let folded = fold(query.trim().as_bytes());
        let terms: Vec<Vec<u8>> = folded.split(|&b| b == b' ').filter(|t| !t.is_empty()).map(<[u8]>::to_vec).collect();
        if terms.is_empty() || limit == 0 {
            return Vec::new();
        }
        let main_i = (0..terms.len()).max_by_key(|&i| terms[i].len()).unwrap();
        let main = terms[main_i].as_slice();
        let others: Vec<&[u8]> = terms.iter().enumerate().filter(|&(i, _)| i != main_i).map(|(_, t)| t.as_slice()).collect();
        let q = Query { main, others: &others, whole: terms.len() == 1, now, limit };

        // 1. A repeat.
        if let Some(c) = self.cache.iter().find(|c| c.build_id == index.build_id && c.generation == index.generation && c.folded == folded && c.limit >= limit) {
            self.last = Path::Repeat;
            return c.hits.iter().take(limit).copied().collect();
        }

        let (hits, ids) = if terms.len() == 1 && main.len() <= 2 && self.starts.as_ref().is_some_and(|s| s.build_id == index.build_id) {
            // 2. One or two letters.
            self.last = Path::WordStarts;
            (self.from_word_starts(index, &q), None)
        } else if let Some((previous, added)) = self.superset_of(&terms, index.build_id) {
            // 3. Refine an earlier query's matches.
            self.last = Path::Refined { candidates: previous.len() };
            refine(index, &previous, added, &q)
        } else {
            // 4. Scan.
            self.last = Path::Scan;
            full_scan(index, &q)
        };

        self.cache.insert(0, Cached { folded, main_len: main.len(), terms, ids: ids.map(Arc::new), hits: hits.clone(), limit, generation: index.generation, added: index.added.len(), build_id: index.build_id });
        self.cache.truncate(CACHED);
        hits
    }

    /// The smallest remembered match set that contains every match of `terms` (with the number of
    /// added entries it already covers).
    fn superset_of(&self, terms: &[Vec<u8>], build_id: u64) -> Option<(Arc<Vec<u32>>, usize)> {
        self.cache
            .iter()
            .filter(|c| c.build_id == build_id && c.main_len >= 3)
            .filter(|c| c.terms.iter().all(|old| terms.iter().any(|new| memmem::find(new, old).is_some())))
            .filter_map(|c| c.ids.clone().map(|ids| (ids, c.added)))
            .min_by_key(|(ids, _)| ids.len())
    }

    fn from_word_starts(&self, index: &Index, q: &Query) -> Vec<Hit> {
        let starts = self.starts.as_ref().unwrap();
        let mut top = Top::new(q.limit);
        // The lists are in score order as of the build; re-score a few times the limit with today's
        // clock (recency moves slowly) and the current deletions.
        let mut taken = 0;
        for &e in starts.get(q.main) {
            let e = e as usize;
            if index.is_deleted(e) {
                continue;
            }
            let name = index.folded_name(e);
            let at = if q.main.len() == 1 { 0 } else { word_start(name, q.main).unwrap_or(0) };
            if let Some(score) = rank(index, e, name, at, q) {
                top.push(Hit { index: e, score });
                taken += 1;
                if taken >= q.limit * 4 {
                    break;
                }
            }
        }
        // Entries added since the build aren't in the lists.
        for (i, added) in index.added.iter().enumerate() {
            if added.deleted {
                continue;
            }
            let name = &added.folded;
            let at = if q.main.len() == 1 { (name.first() == Some(&q.main[0])).then_some(0) } else { word_start(name, q.main) };
            if let Some(at) = at {
                let e = index.count + i;
                if let Some(score) = rank(index, e, name, at, q) {
                    top.push(Hit { index: e, score });
                }
            }
        }
        sorted(index, q, top.into_vec(), q.limit)
    }
}

/// Adds alias matches, then best first, one per entry.
fn sorted(index: &Index, q: &Query, mut hits: Vec<Hit>, limit: usize) -> Vec<Hit> {
    let mut top = Top::new(limit);
    alias_hits(index, q, &mut top);
    hits.extend(top.into_vec());
    finish(hits, limit)
}

/// Checks `q` against `candidates` (every possible match as of when they were collected) and the
/// entries added since (`index.added[covered..]`).
fn refine(index: &Index, candidates: &[u32], covered: usize, q: &Query) -> (Vec<Hit>, Option<Vec<u32>>) {
    let added: Vec<u32> = (index.count + covered.min(index.added.len())..index.total()).map(|e| e as u32).collect();
    if !added.is_empty() {
        let mut all = Vec::with_capacity(candidates.len() + added.len());
        all.extend_from_slice(candidates);
        all.extend(added);
        return refine_ids(index, &all, q);
    }
    refine_ids(index, candidates, q)
}

fn refine_ids(index: &Index, candidates: &[u32], q: &Query) -> (Vec<Hit>, Option<Vec<u32>>) {
    // One searcher for all candidates: building one per name (memmem::find) was most of the cost.
    let finder = memmem::Finder::new(q.main);
    let others: Vec<memmem::Finder> = q.others.iter().map(|t| memmem::Finder::new(t)).collect();
    let check = |slice: &[u32], top: &mut Top, ids: &mut Vec<u32>| {
        for &e in slice {
            let e = e as usize;
            if index.is_deleted(e) {
                continue;
            }
            let name = index.folded_name(e);
            let Some(at) = finder.find(name) else { continue };
            if !others.iter().all(|f| f.find(name).is_some()) {
                continue;
            }
            if let Some(score) = rank(index, e, name, at, q) {
                top.push(Hit { index: e, score });
                ids.push(e as u32);
            }
        }
    };
    if candidates.len() <= REFINE_SINGLE {
        let mut top = Top::new(q.limit);
        let mut ids = Vec::new();
        check(candidates, &mut top, &mut ids);
        return (sorted(index, q, top.into_vec(), q.limit), Some(ids));
    }
    let workers = pool::workers();
    let chunks = workers * 4;
    let per = candidates.len().div_ceil(chunks);
    let results: Vec<Mutex<(Vec<Hit>, Vec<u32>)>> = (0..chunks).map(|_| Mutex::new((Vec::new(), Vec::new()))).collect();
    pool::for_each_chunk(chunks, workers, &|c| {
        let slice = &candidates[(c * per).min(candidates.len())..((c + 1) * per).min(candidates.len())];
        let mut top = Top::new(q.limit);
        let mut ids = Vec::new();
        check(slice, &mut top, &mut ids);
        *results[c].lock().unwrap() = (top.into_vec(), ids);
    });
    let mut hits = Vec::new();
    let mut ids = Vec::new();
    for r in results {
        let (h, i) = r.into_inner().unwrap();
        hits.extend(h);
        ids.extend(i);
    }
    (sorted(index, q, hits, q.limit), Some(ids))
}

/// Every name, on the parked pool; remembers the matches unless there are more than `REMEMBER`.
fn full_scan(index: &Index, q: &Query) -> (Vec<Hit>, Option<Vec<u32>>) {
    let workers = pool::scan_workers();
    // A few chunks per worker: matches cluster (whole node_modules folders), so equal entry counts
    // aren't equal work; workers take the next chunk when done.
    let chunks = workers * 8;
    let per = index.count.div_ceil(chunks).max(1);
    let results: Vec<Mutex<(Vec<Hit>, Collect)>> = (0..chunks).map(|_| Mutex::new((Vec::new(), None))).collect();
    pool::for_each_chunk(chunks, workers, &|c| {
        let mut collect: Collect = Some((Vec::new(), REMEMBER));
        let hits = scan(index, c * per, ((c + 1) * per).min(index.count), q, &mut collect);
        *results[c].lock().unwrap() = (hits, collect);
    });
    let mut hits = Vec::new();
    let mut ids: Option<Vec<u32>> = Some(Vec::new());
    for r in results {
        let (h, collected) = r.into_inner().unwrap();
        hits.extend(h);
        match (ids.as_mut(), collected) {
            (Some(all), Some((part, _))) if all.len() + part.len() <= REMEMBER => all.extend(part),
            _ => ids = None,
        }
    }
    // Entries added since the file was written.
    for (i, added) in index.added.iter().enumerate() {
        if added.deleted {
            continue;
        }
        let name = &added.folded;
        let at = if q.main.len() == 1 {
            (name.first() == Some(&q.main[0])).then_some(0)
        } else if q.main.len() <= 2 {
            word_start(name, q.main)
        } else {
            memmem::find(name, q.main)
        };
        if let Some(at) = at {
            let e = index.count + i;
            if let Some(score) = rank(index, e, name, at, q) {
                hits.push(Hit { index: e, score });
                if let Some(all) = ids.as_mut() {
                    all.push(e as u32);
                }
            }
        }
    }
    (sorted(index, q, hits, q.limit), ids)
}
