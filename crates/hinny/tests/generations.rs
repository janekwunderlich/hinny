//! Segments share their centroids (a generation): a flush assigns its vectors to the ones there
//! are, a merge within a generation copies codes and changes no answer, and vectors that have
//! outgrown a generation train the next.

use hinny::index::{Doc, Found, Index};

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

const DIMS: usize = 64;

/// Clustered, like embeddings: 80 topics, each vector a topic centre plus noise.
fn vectors(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut topics = Rng(7);
    let centres: Vec<Vec<f32>> = (0..80).map(|_| (0..DIMS).map(|_| topics.normal()).collect()).collect();
    let mut rng = Rng(seed);
    (0..n).map(|_| centres[(rng.next() % 80) as usize].iter().map(|c| c + 0.7 * rng.normal()).collect()).collect()
}

fn dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("hinny-generations-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn put(ix: &mut Index, first: u64, vs: &[Vec<f32>]) {
    for (i, v) in vs.iter().enumerate() {
        ix.upsert(&Doc { key: first + i as u64, version: 1, passages: vec![], attrs: vec![], vector: Some(v.clone()) }).unwrap();
    }
    ix.checkpoint().unwrap();
}

fn recall(ix: &Index, queries: &[Vec<f32>]) -> f64 {
    let mut found = 0;
    for q in queries {
        let want: Vec<u64> = ix.nearest_exact(q, 10, None).iter().map(|f| f.key).collect();
        found += ix.nearest(q, 10, None, 0).iter().filter(|f| want.contains(&f.key)).count();
    }
    found as f64 / (queries.len() * 10) as f64
}

fn answers(ix: &Index, queries: &[Vec<f32>]) -> Vec<Vec<Found>> {
    queries.iter().map(|q| ix.nearest(q, 10, None, 0)).collect()
}

#[test]
fn flushes_share_centroids_and_a_merge_within_them_changes_no_answer() {
    let d = dir("shared");
    let mut ix = Index::open(&d, 1).unwrap();
    ix.auto_merge = false;
    ix.flush_passages = 1_000_000;
    let queries = vectors(40, 99);
    // The first segment with enough vectors to cluster trains a generation, with clusters for
    // twice its vectors; the next ones are assigned to its centroids.
    put(&mut ix, 0, &vectors(6000, 1));
    let s = ix.stats();
    assert_eq!((s.segments, s.vector_generations), (1, 1));
    let clusters = s.vector_clusters;
    assert_eq!(clusters, ((2.0 * 6000.0f64).sqrt() as usize), "sized for twice what it was trained on");
    put(&mut ix, 10_000, &vectors(3000, 2));
    put(&mut ix, 20_000, &vectors(500, 3));
    let s = ix.stats();
    assert_eq!((s.segments, s.vector_generations, s.vector_clusters), (3, 1, clusters));
    assert!(recall(&ix, &queries) >= 0.97, "{}", recall(&ix, &queries));
    // Deletions in every segment, seen by the search of all three as one.
    for key in [5, 6, 10_003, 20_001] {
        ix.delete(key).unwrap();
    }
    assert!(answers(&ix, &queries).iter().flatten().all(|f| ![5, 6, 10_003, 20_001].contains(&f.key)));
    let before = answers(&ix, &queries);
    // Merged: the same lists, one after another, with their codes as they were. Nothing is
    // clustered or calibrated again, and every answer is the one from before.
    ix.start_merge(true);
    ix.wait_merge().unwrap();
    let s = ix.stats();
    assert_eq!((s.segments, s.vector_generations, s.vector_clusters), (1, 1, clusters));
    assert_eq!(answers(&ix, &queries), before);
    // And reopened.
    drop(ix);
    let ix = Index::open(&d, 1).unwrap();
    assert_eq!(answers(&ix, &queries), before);
    assert_eq!(ix.stats().vector_clusters, clusters);
    drop(ix);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn vectors_that_outgrow_a_generation_train_the_next() {
    let d = dir("outgrown");
    let mut ix = Index::open(&d, 1).unwrap();
    ix.auto_merge = false;
    ix.flush_passages = 1_000_000;
    let queries = vectors(40, 98);
    // Too few to cluster: compared exactly, in no generation.
    put(&mut ix, 0, &vectors(1000, 11));
    assert_eq!((ix.stats().vector_generations, ix.stats().vector_clusters), (0, 0));
    put(&mut ix, 10_000, &vectors(5000, 12));
    let first = ix.stats().vector_clusters;
    assert_eq!(ix.stats().vector_generations, 1);
    // Four more segments in it, by then more than four times what it was trained on.
    for (i, seed) in (13..17).enumerate() {
        put(&mut ix, 20_000 + 10_000 * i as u64, &vectors(4500, seed));
    }
    let s = ix.stats();
    assert_eq!((s.segments, s.vector_generations, s.vector_clusters), (6, 1, first));
    let before = recall(&ix, &queries);
    assert!(before >= 0.97, "{before}");
    // Their merge has 24,000 vectors: it trains the next generation, with more centroids, and
    // what's flushed afterwards goes into that one.
    ix.start_merge(true);
    ix.wait_merge().unwrap();
    let s = ix.stats();
    assert_eq!((s.segments, s.vector_generations), (1, 1));
    assert!(s.vector_clusters > first, "{} centroids after {first}", s.vector_clusters);
    let grown = s.vector_clusters;
    put(&mut ix, 90_000, &vectors(2000, 18));
    let s = ix.stats();
    assert_eq!((s.segments, s.vector_generations, s.vector_clusters), (2, 1, grown));
    assert!(recall(&ix, &queries) >= 0.97, "{}", recall(&ix, &queries));
    drop(ix);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_segment_of_an_earlier_generation_moves_when_it_is_merged() {
    let d = dir("moves");
    let mut ix = Index::open(&d, 1).unwrap();
    ix.auto_merge = false;
    ix.flush_passages = 1_000_000;
    let queries = vectors(40, 97);
    put(&mut ix, 0, &vectors(4500, 21));
    let small = ix.stats().vector_clusters;
    // A segment four times the first trains its own generation: two are live, and a search
    // ranks each one's centroids.
    put(&mut ix, 10_000, &vectors(20_000, 22));
    let s = ix.stats();
    assert_eq!((s.segments, s.vector_generations), (2, 2));
    assert!(s.vector_clusters > small);
    let large = s.vector_clusters;
    assert!(recall(&ix, &queries) >= 0.97, "{}", recall(&ix, &queries));
    // Merged, the earlier one's vectors are assigned to the larger one's centroids.
    ix.start_merge(true);
    ix.wait_merge().unwrap();
    let s = ix.stats();
    assert_eq!((s.segments, s.vector_generations, s.vector_clusters), (1, 1, large));
    assert!(recall(&ix, &queries) >= 0.97, "{}", recall(&ix, &queries));
    drop(ix);
    std::fs::remove_dir_all(&d).unwrap();
}
