//! The live index: several segments plus memory, deletes and replacements, merges, and recovery
//! from the log after a crash.

use std::collections::HashMap;
use std::path::PathBuf;

use hinny::index::{Found, Index, Request};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn word(&mut self, vocabulary: usize) -> String {
        let u = (self.next() % 1_000_000) as f64 / 1_000_000.0;
        let i = ((vocabulary as f64 + 1.0).powf(u) - 1.0) as usize;
        format!("w{}", i.min(vocabulary - 1))
    }
    fn item(&mut self) -> Vec<Vec<String>> {
        (0..1 + self.next() % 3)
            .map(|_| {
                let title: Vec<String> = (0..1 + self.next() % 5).map(|_| self.word(800)).collect();
                let body: Vec<String> = (0..self.next() % 120).map(|_| self.word(800)).collect();
                vec![title.join(" "), body.join(" ")]
            })
            .collect()
    }
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("hinny-text-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

const FIELDS: [(u8, f32); 2] = [(0, 2.0), (1, 1.0)];

fn search(ix: &Index, text: &str, k: usize, all: bool) -> Vec<Found> {
    ix.search(&Request { text, fields: &FIELDS, k, all, expand: 0, prefix_last: 0, filter: None, sparse: &[], stop: None })
}

fn queries(rng: &mut Rng) -> Vec<(String, bool)> {
    (0..120).map(|i| ((0..1 + rng.next() % 6).map(|_| rng.word(800)).collect::<Vec<_>>().join(" "), i % 4 == 0)).collect()
}

/// Builds a history: items put, some replaced, some deleted, over several checkpoints. Returns the
/// live items in the order an index holds them after merging everything (oldest segment first).
fn history(ix: &mut Index, rng: &mut Rng) -> HashMap<u64, Vec<Vec<String>>> {
    let mut live = HashMap::new();
    for round in 0..6 {
        for _ in 0..300 {
            let key = rng.next() % 1500;
            if rng.next() % 7 == 0 {
                ix.delete(key).unwrap();
                live.remove(&key);
            } else {
                let item = rng.item();
                ix.put(key, round, &item).unwrap();
                live.insert(key, item);
            }
        }
        if round < 5 {
            ix.checkpoint().unwrap();
        }
    }
    live
}

#[test]
fn pruning_across_segments_equals_scoring_everything() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let path = dir("prune");
    let mut ix = Index::open(&path, 2).unwrap();
    ix.auto_merge = false;
    history(&mut ix, &mut rng);
    let stats = ix.stats();
    assert!(stats.segments >= 4 && stats.memory > 0 && stats.deleted > 0, "{stats:?}");
    for (text, all) in queries(&mut rng) {
        let everything = search(&ix, &text, 100_000, all);
        for k in [1, 5, 20] {
            let top = search(&ix, &text, k, all);
            assert!(top.len() <= k);
            // The pruned results are the best of everything, in order.
            for (a, b) in top.iter().zip(&everything) {
                assert!((a.score - b.score).abs() <= 1e-4 * b.score.max(1.0), "{text:?} k {k}: {a:?} vs {b:?}");
            }
        }
    }
    drop(ix);
    std::fs::remove_dir_all(&path).unwrap();
}

#[test]
fn merged_index_equals_a_fresh_one() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let path = dir("merge");
    let mut ix = Index::open(&path, 2).unwrap();
    ix.auto_merge = false;
    let live = history(&mut ix, &mut rng);
    ix.checkpoint().unwrap();
    assert!(ix.start_merge(true));
    assert!(ix.wait_merge().unwrap());
    let stats = ix.stats();
    assert_eq!((stats.segments, stats.deleted), (1, 0), "{stats:?}");
    assert_eq!(ix.versions().len(), live.len());

    // The same live items, indexed from scratch: scores depend on the passages and statistics, not
    // on their order, so they must match (ties may order documents differently).
    let fresh_path = dir("fresh");
    let mut fresh = Index::open(&fresh_path, 2).unwrap();
    fresh.auto_merge = false;
    let mut keys: Vec<&u64> = live.keys().collect();
    keys.sort();
    for key in keys {
        fresh.put(*key, 0, &live[key]).unwrap();
    }
    fresh.checkpoint().unwrap();
    for (text, all) in queries(&mut rng) {
        let (a, b) = (search(&ix, &text, 10, all), search(&fresh, &text, 10, all));
        assert_eq!(a.len(), b.len(), "{text:?}");
        for (x, y) in a.iter().zip(&b) {
            assert!((x.score - y.score).abs() <= 1e-5 * y.score.max(1.0), "{text:?}: {x:?} vs {y:?}");
        }
    }
    drop((ix, fresh));
    std::fs::remove_dir_all(&path).unwrap();
    std::fs::remove_dir_all(&fresh_path).unwrap();
}

#[test]
fn a_crash_loses_nothing_acknowledged() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let path = dir("crash");
    let mut ix = Index::open(&path, 2).unwrap();
    ix.auto_merge = false;
    history(&mut ix, &mut rng);
    let qs = queries(&mut rng);
    let before: Vec<Vec<Found>> = qs.iter().map(|(t, all)| search(&ix, t, 10, *all)).collect();
    let versions = ix.versions();
    // No checkpoint, no drop: as if the process died. The log has everything since the last one.
    ix.simulate_crash();
    let ix = Index::open(&path, 2).unwrap();
    assert_eq!(ix.versions(), versions);
    for ((t, all), want) in qs.iter().zip(&before) {
        let got = search(&ix, t, 10, *all);
        assert_eq!(got.len(), want.len(), "{t:?}");
        for (a, b) in got.iter().zip(want) {
            assert!((a.score - b.score).abs() <= 1e-5 * b.score.max(1.0), "{t:?}: {a:?} vs {b:?}");
        }
    }
    drop(ix);
    std::fs::remove_dir_all(&path).unwrap();
}

/// Filling memory writes it out in the background: every write is visible at once, through the
/// freeze, the build and the install; deletes that land during the build stay deleted; a crash
/// before the install loses nothing (the frozen memtable's log stays until its segment is named).
#[test]
fn background_flushes() {
    let path = std::env::temp_dir().join(format!("hinny-live-flush-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    let mut ix = Index::open(&path, 2).unwrap();
    ix.flush_passages = 700;
    ix.auto_merge = false;
    let doc = |k: u64| vec![vec![format!("title {k}"), format!("common words for item{k}")]];
    let mut live = std::collections::BTreeSet::new();
    for k in 0..5000u64 {
        ix.put(k, 1, &doc(k)).unwrap();
        live.insert(k);
        // Deletes, some of them landing while a frozen memtable is being written.
        if k % 13 == 5 {
            ix.delete(k - 3).unwrap();
            live.remove(&(k - 3));
        }
        if k % 250 == 0 {
            let have: std::collections::BTreeSet<u64> = ix.versions().into_keys().collect();
            assert_eq!(have, live, "after {k}");
            let hits = search(&ix, &format!("item{k}"), 3, false);
            assert_eq!(hits.first().map(|f| f.key), Some(k), "{k} searchable at once");
        }
    }
    assert!(ix.stats().segments >= 3, "{:?}", ix.stats());
    // A crash with a flush perhaps in flight: everything comes back from segments and logs.
    let before: std::collections::BTreeSet<u64> = ix.versions().into_keys().collect();
    ix.simulate_crash();
    let ix = Index::open(&path, 2).unwrap();
    assert_eq!(ix.versions().into_keys().collect::<std::collections::BTreeSet<_>>(), before);
    assert_eq!(search(&ix, "item4999", 1, false)[0].key, 4999);
    drop(ix);
    // After a clean close, no log but the current one is left.
    let logs = std::fs::read_dir(&path).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with("wal")).count();
    assert_eq!(logs, 1);
    let ix = Index::open(&path, 2).unwrap();
    assert_eq!(ix.versions().len(), before.len());
    drop(ix);
    std::fs::remove_dir_all(&path).unwrap();
}
