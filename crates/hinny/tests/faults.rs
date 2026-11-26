//! Every I/O failure the engine can meet, one at a time: for each named point (`hinny::faults`)
//! and each time a workload reaches it, fail that call, stop as a crash would, reopen, and check
//! what survived. Run with `cargo test --features failpoints --test faults`.
//!
//! What must hold, whatever failed: the index reopens; it holds exactly the state after some
//! prefix of the batches, each whole or not at all; and that prefix includes every batch
//! acknowledged before the last successful `sync` or `checkpoint`. These are the engine's own
//! errors after a failed call, not lost writes in the OS or the disk (see DESIGN.md).

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use hinny::faults;
use hinny::index::{Doc, Index, Op};

const BATCHES: u64 = 24;

/// How a workload runs: with merges (which rewrite deletions into new segments, hiding a lost
/// deletion file) or without, and deleting in every batch or only in a few early ones (so a
/// segment's deletions stop changing and a failed save of them can't be papered over).
#[derive(Clone, Copy, Debug)]
struct Config {
    merges: bool,
    /// The first and last batch that delete.
    deletes: (u64, u64),
}

const CONFIGS: [Config; 2] = [Config { merges: true, deletes: (3, BATCHES) }, Config { merges: false, deletes: (6, 8) }];

fn dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("hinny-faults-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn doc(key: u64, version: u64) -> Doc {
    Doc { key, version, passages: vec![vec![format!("batch {version} key {key}")]], attrs: Vec::new(), vector: None }
}

/// Batch `b`: puts three new keys, moves key 0 to version `b`, and in the configured batches
/// deletes the first key of the batch two before it.
fn batch(c: Config, b: u64) -> (Vec<Doc>, Vec<u64>) {
    let mut docs = vec![doc(0, b)];
    docs.extend((1..=3).map(|i| doc(b * 10 + i, b)));
    let deletes = if (c.deletes.0..=c.deletes.1).contains(&b) { vec![(b - 2) * 10 + 1] } else { Vec::new() };
    (docs, deletes)
}

/// The keys and versions after batches 1..=m.
fn model(c: Config, m: u64) -> HashMap<u64, u64> {
    let mut state = HashMap::new();
    for b in 1..=m {
        let (docs, deletes) = batch(c, b);
        for d in docs {
            state.insert(d.key, d.version);
        }
        for k in deletes {
            state.remove(&k);
        }
    }
    state
}

struct Outcome {
    /// Batches whose write returned Ok.
    acknowledged: u64,
    /// The last batch before a successful sync or checkpoint.
    synced: u64,
}

/// Writes the batches with a sync every third and a checkpoint every fifth, carrying on after any
/// failure: later syncs and checkpoints retry what a failed one couldn't save, and a writer
/// whose log failed must refuse every later write (one acknowledged behind a torn record would
/// be lost on replay, and the prefix check would see the gap).
fn workload(c: Config, ix: &mut Index) -> Outcome {
    let mut out = Outcome { acknowledged: 0, synced: 0 };
    let mut failed = false;
    for b in 1..=BATCHES {
        let (docs, deletes) = batch(c, b);
        let mut ops: Vec<Op> = docs.iter().map(Op::Put).collect();
        ops.extend(deletes.iter().map(|&k| Op::Delete(k)));
        if ix.write(&ops).is_err() {
            failed = true;
            continue;
        }
        assert!(!failed, "batch {b} was acknowledged after an earlier write failed");
        out.acknowledged = b;
        let boundary = if b % 5 == 0 {
            Some(ix.checkpoint())
        } else if b % 3 == 0 {
            Some(ix.sync())
        } else {
            None
        };
        if let Some(Ok(())) = boundary {
            out.synced = b;
        }
    }
    out
}

fn run(c: Config, name: &str, fail: Option<(&str, usize)>) -> (Outcome, BTreeMap<String, usize>, HashMap<u64, u64>) {
    let path = dir(name);
    faults::reset();
    let mut ix = Index::open(&path, 1).unwrap();
    ix.auto_merge = c.merges;
    ix.merge_factor = 2;
    if let Some((point, nth)) = fail {
        faults::fail_at(point, nth);
    }
    let outcome = workload(c, &mut ix);
    let _ = ix.wait_merge();
    let hits = faults::reset();
    ix.simulate_crash();
    let reopened = Index::open(&path, 1).unwrap_or_else(|e| panic!("{c:?} {fail:?}: reopening failed: {e}"));
    let state = reopened.versions();
    drop(reopened);
    std::fs::remove_dir_all(&path).unwrap();
    (outcome, hits, state)
}

#[test]
fn every_failure_leaves_a_whole_prefix_with_every_synced_batch() {
    let mut cases = 0;
    for (i, c) in CONFIGS.into_iter().enumerate() {
        // A clean run says which points the workload reaches, and how often.
        let (clean, hits, state) = run(c, &format!("clean-{i}"), None);
        assert_eq!(clean.acknowledged, BATCHES);
        assert_eq!(state, model(c, BATCHES));
        assert!(hits.contains_key("wal append") && hits.contains_key("write manifest") && hits.contains_key("write del"), "{hits:?}");
        for (point, reached) in &hits {
            for nth in 1..=*reached {
                let (out, _, state) = run(c, &format!("{i}-{}-{nth}", point.replace(' ', "-")), Some((point, nth)));
                let Some(m) = (0..=BATCHES).find(|&m| model(c, m) == state) else {
                    panic!("{c:?}, {point} #{nth}: the reopened index isn't the state after any prefix of batches");
                };
                assert!(m >= out.synced, "{c:?}, {point} #{nth}: batch {} was synced but only {m} survived", out.synced);
                assert!(m <= out.acknowledged + 1, "{c:?}, {point} #{nth}: {m} batches survived, more than were written");
                cases += 1;
            }
        }
    }
    println!("{cases} injected failures in {} configurations", CONFIGS.len());
}
