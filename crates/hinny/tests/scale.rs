//! Work that grows with the collection where it shouldn't. Not run by default (a million vectors:
//! a few minutes and 3 GB):
//!
//!   cargo test --release -p hinny --test scale -- --ignored --nocapture
//!
//! Timings differ by machine, so a search is held against scoring every vector at full precision
//! on the same machine in the same run. At a million 256-dimension vectors a search takes well
//! under 1% of that (an M1 Max under Linux: 0.12 ms of 20 ms). A step in the query path that looks
//! at every passage shows as several percent: 4.8% when the count of a segment's deletions was
//! one. `SCALE_VECTORS` changes how many (10000000 for a nightly run).

use std::collections::HashSet;
use std::time::Instant;

use hinny::index::{Doc, Index, View};

const DIMS: usize = 256;
/// A search may take this share of an exact scan.
const SHARE: f64 = 0.02;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn normal(&mut self) -> f32 {
        (0..4).map(|_| (self.next() % 1_000_000) as f32 / 1_000_000.0).sum::<f32>() - 2.0
    }
}

/// The median of three rounds' medians, in milliseconds, and recall@10 against `truth`.
fn search(view: &View, queries: &[&[f32]], truth: &[HashSet<u64>]) -> (f64, f64) {
    let mut medians = Vec::new();
    let mut recall = 0.0;
    for _ in 0..3 {
        let mut times = Vec::new();
        recall = 0.0;
        for (q, want) in queries.iter().zip(truth) {
            let t = Instant::now();
            let got = view.nearest(q, 10, None, 0);
            times.push(t.elapsed().as_secs_f64() * 1000.0);
            recall += got.iter().filter(|f| want.contains(&f.key)).count() as f64 / want.len().max(1) as f64;
        }
        times.sort_by(f64::total_cmp);
        medians.push(times[times.len() / 2]);
    }
    medians.sort_by(f64::total_cmp);
    (medians[1], recall / queries.len() as f64)
}

#[test]
#[ignore = "a million vectors: run with --ignored"]
fn a_search_does_not_grow_with_the_collection() {
    let n: usize = std::env::var("SCALE_VECTORS").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let queries = 200;
    // Clustered, like embeddings: 1,000 topics, each vector a topic centre plus noise.
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let topics: Vec<Vec<f32>> = (0..1000).map(|_| (0..DIMS).map(|_| rng.normal()).collect()).collect();
    let mut data = Vec::with_capacity((n + queries) * DIMS);
    for _ in 0..n + queries {
        let t = &topics[(rng.next() % 1000) as usize];
        data.extend(t.iter().map(|c| c + 0.8 * rng.normal()));
    }
    let row = |i: usize| &data[i * DIMS..(i + 1) * DIMS];

    let dir = std::env::temp_dir().join(format!("hinny-scale-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.flush_passages = 200_000;
    ix.auto_merge = false;
    let t = Instant::now();
    for i in 0..n {
        ix.upsert(&Doc { key: i as u64, version: 1, passages: vec![], attrs: vec![], vector: Some(row(i).to_vec()) }).unwrap();
    }
    ix.checkpoint().unwrap();
    if ix.stats().segments > 1 {
        ix.start_merge(true);
        ix.wait_merge().unwrap();
    }
    println!("{n} vectors of {DIMS} dimensions in {:.0} s, {} segment(s)", t.elapsed().as_secs_f64(), ix.stats().segments);

    let qs: Vec<&[f32]> = (n..n + queries).map(row).collect();
    let view = ix.view();
    let t = Instant::now();
    let truth: Vec<HashSet<u64>> = qs.iter().map(|q| view.nearest_exact(q, 10, None).iter().map(|f| f.key).collect()).collect();
    let exact = t.elapsed().as_secs_f64() * 1000.0 / qs.len() as f64;
    let (clean, recall) = search(&view, &qs, &truth);
    println!("exact {exact:.2} ms; a search {clean:.3} ms ({:.2}% of it), recall@10 {recall:.3}", clean / exact * 100.0);
    assert!(recall >= 0.99, "recall@10 {recall:.3}");
    assert!(clean <= SHARE * exact, "a search takes {clean:.3} ms, {:.1}% of an exact scan's {exact:.2} ms (at most {:.0}%)", clean / exact * 100.0, SHARE * 100.0);

    // The same with deletions all over the segment, from a view taken before them (which must not
    // count what was deleted since, passage by passage) and from one after.
    let before = ix.view();
    let t = Instant::now();
    for key in (0..n as u64).step_by(97) {
        ix.delete(key).unwrap();
    }
    let per_delete = t.elapsed().as_secs_f64() * 1e6 / (n as f64 / 97.0);
    let (old, old_recall) = search(&before, &qs, &truth);
    let after = ix.view();
    let (new, _) = search(&after, &qs, &truth);
    println!("after {} deletions ({per_delete:.1} µs each): a search from before them {old:.3} ms, from after {new:.3} ms", n / 97 + 1);
    assert!(old_recall >= 0.99, "recall@10 from the view before the deletions {old_recall:.3}");
    assert_eq!(before.stats().deleted, 0);
    assert_eq!(after.stats().deleted, n / 97 + 1);
    for (name, ms) in [("before", old), ("after", new)] {
        assert!(ms <= SHARE * exact, "from the view {name} the deletions a search takes {ms:.3} ms, {:.1}% of an exact scan", ms / exact * 100.0);
    }

    // The writer: a batch after everything else is in place takes what a batch takes, not a walk
    // over the collection (the merge policy and the view it publishes both ask how much is deleted).
    let t = Instant::now();
    for i in 0..1000 {
        ix.upsert(&Doc { key: (n + i) as u64, version: 1, passages: vec![], attrs: vec![], vector: Some(row(n + i % queries).to_vec()) }).unwrap();
    }
    let per_write = t.elapsed().as_secs_f64() * 1e6 / 1000.0;
    println!("a write into the full index: {per_write:.1} µs");
    assert!(per_write < 1000.0, "a write takes {per_write:.0} µs with {n} vectors in place");

    drop((view, before, after));
    drop(ix);
    std::fs::remove_dir_all(dir).unwrap();
}
