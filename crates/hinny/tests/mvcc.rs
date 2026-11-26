//! Versions: snapshots see exactly the batches before them, through later writes, flushes and
//! merges; readers never see half a batch; and a batch torn by a crash comes back whole or not at
//! all.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use hinny::index::{Doc, Index, Op};
use hinny::{Change, Database, Document, Schema};

fn dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("hinny-mvcc-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn keys(hits: &[hinny::Hit]) -> Vec<u64> {
    let mut k: Vec<u64> = hits.iter().map(|h| h.key).collect();
    k.sort_unstable();
    k
}

#[test]
fn snapshots_hold_through_writes_flushes_and_merges() {
    let d = dir("snap");
    let ns = Database::open(&d).unwrap().namespace("docs", Schema::new().text("body", 1.0).int("n").vector(3)).unwrap();
    let doc = |k: u64, word: &str| Document::new(k).text("body", format!("{word} number{k}")).int("n", k as i64).vector(vec![1.0, k as f32, 0.0]);
    ns.upsert_many((0..50).map(|k| doc(k, "apple"))).unwrap();
    ns.flush().unwrap();
    ns.upsert_many((50..100).map(|k| doc(k, "apple"))).unwrap();
    let snap = ns.snapshot();
    let before = keys(&snap.query().text("apple").limit(200).run().unwrap());
    assert_eq!(before, (0..100).collect::<Vec<_>>());
    // Later: deletes (from a segment and from memory), replacements, new documents, a flush and
    // a merge of everything.
    for k in [3u64, 60] {
        ns.delete(k).unwrap();
    }
    ns.upsert(doc(10, "banana")).unwrap();
    ns.upsert_many((100..130).map(|k| doc(k, "apple"))).unwrap();
    ns.flush().unwrap();
    ns.optimize().unwrap();
    assert_eq!(ns.stats().segments, 1);
    // The snapshot: as it was.
    assert_eq!(keys(&snap.query().text("apple").limit(200).run().unwrap()), before);
    assert!(snap.query().text("banana").run().unwrap().is_empty());
    assert_eq!(snap.get(3).map(|a| a.len()), Some(1));
    assert_eq!(snap.versions().len(), 100);
    assert_eq!(snap.query().vector(vec![1.0, 60.0, 0.0]).limit(1).run().unwrap()[0].key, 60);
    // The namespace now.
    let now = keys(&ns.query().text("apple").limit(200).run().unwrap());
    assert_eq!(now.len(), 100 + 30 - 3);
    assert!(!now.contains(&3) && !now.contains(&60) && !now.contains(&10));
    assert_eq!(keys(&ns.query().text("banana").run().unwrap()), [10]);
    assert!(ns.get(3).is_none());
    drop(snap);
    drop(ns);
    std::fs::remove_dir_all(&d).unwrap();
}

/// A writer rewrites pairs of documents in one batch, again and again; readers on other threads
/// must always see both of a pair at the same version, and never a missing document.
#[test]
fn readers_never_see_half_a_batch() {
    let d = dir("atomic");
    let ns = Database::open(&d).unwrap().namespace("docs", Schema::new().text("body", 1.0).int("round")).unwrap();
    let pair = |round: i64| (0..20u64).map(move |k| Change::Put(Document::new(k).version(round as u64).text("body", format!("pair{} round{round}", k / 2)).int("round", round)));
    ns.apply(pair(0)).unwrap();
    let done = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let (ns, done) = (ns.clone(), done.clone());
            std::thread::spawn(move || {
                let mut views = 0;
                while !done.load(Ordering::Relaxed) {
                    let snap = ns.snapshot();
                    let v = snap.versions();
                    assert_eq!(v.len(), 20, "every document, always");
                    let round = v[&0];
                    assert!(v.values().all(|&r| r == round), "one round across the batch: {v:?}");
                    // Queries on the snapshot agree with it.
                    let hits = snap.query().text(format!("round{round}")).limit(50).run().unwrap();
                    assert_eq!(hits.len(), 20);
                    views += 1;
                }
                views
            })
        })
        .collect();
    for round in 1..400 {
        ns.apply(pair(round)).unwrap();
        if round % 100 == 0 {
            ns.flush().unwrap();
        }
    }
    done.store(true, Ordering::Relaxed);
    let views: usize = readers.into_iter().map(|r| r.join().unwrap()).sum();
    assert!(views > 20, "readers ran alongside ({views} views)");
    drop(ns);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_torn_batch_comes_back_whole_or_not_at_all() {
    let d = dir("torn");
    let doc = |k: u64| Doc { key: k, version: 1, passages: vec![vec![format!("item{k}")]], attrs: vec![], vector: None };
    let mut ix = Index::open(&d, 1).unwrap();
    let first: Vec<Doc> = (0..10).map(doc).collect();
    ix.write(&first.iter().map(Op::Put).collect::<Vec<_>>()).unwrap();
    let second: Vec<Doc> = (10..20).map(doc).collect();
    let mut ops: Vec<Op> = second.iter().map(Op::Put).collect();
    ops.push(Op::Delete(0));
    ix.write(&ops).unwrap();
    ix.simulate_crash();
    // The log ends mid-way through the second batch.
    let log = std::fs::read_dir(&d).unwrap().flatten().map(|e| e.path()).find(|p| p.file_name().unwrap().to_string_lossy().starts_with("wal-")).unwrap();
    let bytes = std::fs::read(&log).unwrap();
    let whole = Index::open_read_only(&d, 1).unwrap().versions().len();
    assert_eq!(whole, 19, "both batches, with the delete");
    std::fs::write(&log, &bytes[..bytes.len() - 30]).unwrap();
    let ix = Index::open(&d, 1).unwrap();
    let mut have: Vec<u64> = ix.versions().into_keys().collect();
    have.sort_unstable();
    assert_eq!(have, (0..10).collect::<Vec<_>>(), "the first batch, none of the second (not even its delete)");
    drop(ix);
    std::fs::remove_dir_all(&d).unwrap();
}

/// A view's count of deletions is its own: what was deleted when it was taken, in segments and in
/// memory, whatever is deleted afterwards. (Searches size their work by it, so it's kept with the
/// view instead of being counted, passage by passage, by every query.)
#[test]
fn a_view_counts_the_deletions_it_sees() {
    let d = dir("counts");
    let mut ix = Index::open(&d, 1).unwrap();
    let doc = |k: u64| Doc { key: k, version: 1, passages: vec![vec![format!("word{k}")]], attrs: vec![], vector: Some(vec![1.0, k as f32, 0.5]) };
    for k in 0..300 {
        ix.upsert(&doc(k)).unwrap();
    }
    ix.checkpoint().unwrap();
    for k in 300..400 {
        ix.upsert(&doc(k)).unwrap();
    }
    for k in [1, 2, 350] {
        ix.delete(k).unwrap();
    }
    let old = ix.view();
    assert_eq!(old.stats().deleted, 3);
    for k in (10..60).chain([360]) {
        ix.delete(k).unwrap();
    }
    assert_eq!(old.stats().deleted, 3, "a view from before later deletions");
    assert_eq!(ix.view().stats().deleted, 3 + 50 + 1);
    // And through a merge that drops them: the old view still has its segment and its count.
    ix.checkpoint().unwrap();
    ix.start_merge(true);
    ix.wait_merge().unwrap();
    assert_eq!(ix.view().stats().deleted, 0);
    assert_eq!(old.stats().deleted, 3);
    assert_eq!(old.nearest(&[1.0, 20.0, 0.5], 1, None, 0)[0].key, 20, "deleted since, still in the old view");
    assert!(ix.view().nearest(&[1.0, 20.0, 0.5], 400, None, 0).iter().all(|f| f.key != 20));
    drop(old);
    drop(ix);
    std::fs::remove_dir_all(&d).unwrap();
}
