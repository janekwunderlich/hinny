//! A text search returns the best k items, each at its best passage, wherever the passages are
//! (memory, segments, merged), whatever pruning skipped: an item's many strong passages can't crowd
//! out the next item, equal scores go to the lower key, and "all words" holds for any number of
//! words. Checked against an independent oracle (which passages hold which words) and against the
//! same search asked for every item, where nothing can be pruned.

use std::collections::{BTreeSet, HashMap};

use hinny::index::{Found, Index, Request};

fn dir(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("hinny-items-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn search(ix: &Index, text: &str, k: usize, all: bool) -> Vec<Found> {
    ix.search(&Request { text, fields: &[(0, 1.0)], k, all, expand: 0, prefix_last: 0, filter: None, sparse: &[], stop: None })
}

fn keys(found: &[Found]) -> Vec<u64> {
    found.iter().map(|f| f.key).collect()
}

/// Memory, then on disk, then merged into one segment.
fn each_layout(ix: &mut Index, check: &mut dyn FnMut(&Index, &str)) {
    check(ix, "memory");
    ix.checkpoint().unwrap();
    check(ix, "a segment");
    ix.start_merge(true);
    ix.wait_merge().unwrap();
    check(ix, "merged");
}

#[test]
fn an_items_passages_dont_crowd_out_the_next_item() {
    let path = dir("crowd");
    let mut ix = Index::open(&path, 1).unwrap();
    ix.auto_merge = false;
    ix.put(1, 1, &vec![vec!["needle needle needle needle".to_string()]; 8]).unwrap();
    ix.put(2, 1, &[vec!["needle filler filler filler filler filler filler filler filler filler".to_string()]]).unwrap();
    each_layout(&mut ix, &mut |ix, at| {
        for k in [1, 2, 3, 100] {
            let found = search(ix, "needle", k, false);
            assert_eq!(keys(&found), [1, 2][..k.min(2)], "k {k}, {at}");
        }
        assert_eq!(search(ix, "needle", 0, false), [], "k 0, {at}");
    });
    drop(ix);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn equal_scores_go_to_the_lower_key_wherever_the_items_are() {
    let path = dir("ties");
    let mut ix = Index::open(&path, 1).unwrap();
    ix.auto_merge = false;
    // Inserted in reverse key order, across a segment and memory.
    for key in (1..=20u64).rev() {
        ix.put(key, 1, &[vec!["same words here".to_string()], vec!["other".to_string()]]).unwrap();
        if key == 11 {
            ix.checkpoint().unwrap();
        }
    }
    each_layout(&mut ix, &mut |ix, at| {
        for k in [1, 3, 10, 20, 30] {
            assert_eq!(keys(&search(ix, "same words", k, false)), (1..=k.min(20) as u64).collect::<Vec<_>>(), "k {k}, {at}");
        }
    });
    drop(ix);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn all_words_holds_for_any_number_of_words() {
    for n in [2usize, 63, 64, 65, 128] {
        let path = dir(&format!("and{n}"));
        let mut ix = Index::open(&path, 1).unwrap();
        ix.auto_merge = false;
        let words: Vec<String> = (0..n).map(|i| format!("w{i}")).collect();
        // Each item lacks one word, except item 1000; item 2000 has them all, but split over two
        // passages.
        for missing in 0..n as u64 {
            let have: Vec<&str> = words.iter().enumerate().filter(|(i, _)| *i as u64 != missing).map(|(_, w)| w.as_str()).collect();
            ix.put(missing + 1, 1, &[vec![have.join(" ")]]).unwrap();
        }
        ix.put(1000, 1, &[vec![words.join(" ")]]).unwrap();
        ix.put(2000, 1, &[vec![words[..n / 2].join(" ")], vec![words[n / 2..].join(" ")]]).unwrap();
        let query = words.join(" ");
        each_layout(&mut ix, &mut |ix, at| {
            assert_eq!(keys(&search(ix, &query, 10, true)), [1000], "{n} words, {at}");
            assert_eq!(search(ix, &query, 1000, false).len(), n + 2, "{n} words, any of them, {at}");
        });
        drop(ix);
        std::fs::remove_dir_all(path).unwrap();
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Roughly Zipf over `vocabulary` words.
    fn word(&mut self, vocabulary: usize) -> String {
        let u = (self.next() % 1_000_000) as f64 / 1_000_000.0;
        let i = ((vocabulary as f64 + 1.0).powf(u) - 1.0) as usize;
        format!("w{}", i.min(vocabulary - 1))
    }
}

#[test]
fn pruned_items_equal_every_item_ranked() {
    let path = dir("random");
    let mut ix = Index::open(&path, 1).unwrap();
    ix.auto_merge = false;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    // Which words each live item's passages hold: the oracle for "all words".
    let mut held: HashMap<u64, Vec<BTreeSet<String>>> = HashMap::new();
    for round in 0..4 {
        for _ in 0..1500 {
            let key = rng.next() % 4000;
            if rng.next() % 10 == 0 {
                ix.delete(key).unwrap();
                held.remove(&key);
                continue;
            }
            let passages: Vec<Vec<String>> = (0..1 + rng.next() % 6).map(|_| vec![(0..1 + rng.next() % 30).map(|_| rng.word(400)).collect::<Vec<_>>().join(" ")]).collect();
            held.insert(key, passages.iter().map(|p| p[0].split(' ').map(str::to_string).collect()).collect());
            ix.put(key, round + 1, &passages).unwrap();
        }
        if round < 3 {
            ix.checkpoint().unwrap(); // three segments, then memory
        }
    }
    let check = |ix: &Index, at: &str| {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for q in 0..300 {
            let words: Vec<String> = (0..1 + rng.next() % 6).map(|_| rng.word(400)).collect();
            let all = q % 3 == 0;
            let query = words.join(" ");
            let every = search(ix, &query, usize::MAX >> 1, all);
            if all {
                let expect: BTreeSet<u64> = held.iter().filter(|(_, ps)| ps.iter().any(|p| words.iter().all(|w| p.contains(w)))).map(|(k, _)| *k).collect();
                assert_eq!(every.iter().map(|f| f.key).collect::<BTreeSet<_>>(), expect, "query {q} ({query}), {at}");
            }
            assert_eq!(every.len(), every.iter().map(|f| f.key).collect::<BTreeSet<_>>().len(), "each item once");
            for k in [1, 2, 5, 10, 50] {
                let top = search(ix, &query, k, all);
                assert_eq!(top.len(), every.len().min(k), "query {q} k {k}, {at}");
                for (i, (a, b)) in top.iter().zip(&every).enumerate() {
                    // Summed in another order (pruned or term at a time): within rounding.
                    assert!((a.score - b.score).abs() <= 1e-4 * b.score.max(1.0), "query {q} k {k} #{i}, {at}: {a:?} vs {b:?}");
                    // Where scores clearly differ, the same items in the same order.
                    let tied = every.iter().any(|c| c.key != b.key && (c.score - b.score).abs() <= 1e-4 * b.score.max(1.0));
                    if !tied {
                        assert_eq!(a.key, b.key, "query {q} k {k} #{i}, {at}");
                    }
                }
            }
        }
    };
    check(&ix, "segments and memory");
    ix.checkpoint().unwrap();
    ix.start_merge(true);
    ix.wait_merge().unwrap();
    check(&ix, "merged");
    drop(ix);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn all_words_with_sparse_terms_requires_the_words_not_the_terms() {
    use hinny::{Database, Document, Schema};
    let path = dir("sparse-all");
    let db = Database::open(&path).unwrap();
    let ns = db.namespace("docs", Schema::new().text("body", 1.0).sparse("terms")).unwrap();
    ns.upsert(Document::new(1).text("body", "tax return for march").sparse("terms", &[("filing", 0.5)])).unwrap();
    ns.upsert(Document::new(2).text("body", "tax office").sparse("terms", &[("filing", 3.0)])).unwrap();
    ns.upsert(Document::new(3).text("body", "march tax deadline")).unwrap();
    ns.upsert(Document::new(4).text("body", "unrelated").sparse("terms", &[("filing", 9.0)])).unwrap();
    for at in ["memory", "disk"] {
        let keys = |all: bool| {
            let q = ns.query().text("tax march").sparse("terms", [("filing", 1.0)]).limit(10);
            let mut k: Vec<u64> = if all { q.all_words() } else { q }.run().unwrap().iter().map(|h| h.key).collect();
            k.sort();
            k
        };
        assert_eq!(keys(true), [1, 3], "{at}: both words, with or without the sparse term");
        assert_eq!(keys(false), [1, 2, 3, 4], "{at}");
        // A required word no document has: nothing, though the sparse term matches.
        let none = ns.query().text("tax nowhere").sparse("terms", [("filing", 1.0)]).all_words().run().unwrap();
        assert!(none.is_empty(), "{at}");
        ns.flush().unwrap();
    }
    drop(ns);
    std::fs::remove_dir_all(path).unwrap();
}
