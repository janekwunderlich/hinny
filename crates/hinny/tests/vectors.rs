//! Vectors: approximate search finds what exact search finds, across segments, deletes, selective
//! filters, merges and a crash.

use std::collections::HashSet;

use hinny::attrs::{Filter, Value};
use hinny::index::{Doc, Index};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 1_000_000) as f32 / 1_000_000.0
    }
    /// Roughly normal (sum of uniforms).
    fn normal(&mut self) -> f32 {
        (0..4).map(|_| self.unit()).sum::<f32>() - 2.0
    }
}

const DIMS: usize = 128;

/// Clustered data, like embeddings: 60 topics, each vector a topic centre plus noise.
fn data(n: usize, rng: &mut Rng) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let topics: Vec<Vec<f32>> = (0..60).map(|_| (0..DIMS).map(|_| rng.normal()).collect()).collect();
    let make = |rng: &mut Rng| {
        let t = &topics[(rng.next() % 60) as usize];
        t.iter().map(|c| c + 0.6 * rng.normal()).collect::<Vec<f32>>()
    };
    let docs = (0..n).map(|_| make(rng)).collect();
    let queries = (0..60).map(|_| make(rng)).collect();
    (docs, queries)
}

fn recall(got: &[hinny::index::Found], want: &[hinny::index::Found]) -> f64 {
    let w: HashSet<u64> = want.iter().map(|f| f.key).collect();
    got.iter().filter(|f| w.contains(&f.key)).count() as f64 / want.len().max(1) as f64
}

#[test]
fn approximate_search_finds_what_exact_search_finds() {
    let dir = std::env::temp_dir().join(format!("hinny-vectors-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let (vectors, queries) = data(30_000, &mut rng);
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.auto_merge = false;
    ix.flush_passages = 1_000_000;
    for (i, v) in vectors.iter().enumerate() {
        ix.upsert(&Doc { key: i as u64, version: 1, passages: vec![vec![format!("doc {i}")]], attrs: vec![("group".into(), Value::Int((i % 50) as i64))], vector: Some(v.clone()) }).unwrap();
        // Three segments, and the rest in memory.
        if i % 9000 == 8999 {
            ix.checkpoint().unwrap();
        }
    }
    for i in (0..30_000).step_by(7) {
        ix.delete(i as u64).unwrap();
    }
    assert!(ix.stats().segments >= 3 && ix.stats().memory > 0, "{:?}", ix.stats());

    let check = |ix: &Index, label: &str| {
        // Unfiltered.
        let mut total = 0.0;
        for q in &queries {
            let want = ix.nearest_exact(q, 10, None);
            let got = ix.nearest(q, 10, None, 16);
            assert!(got.iter().all(|f| f.key % 7 != 0), "{label}: a deleted document came back");
            total += recall(&got, &want);
        }
        let r = total / queries.len() as f64;
        assert!(r >= 0.95, "{label}: recall@10 {r:.3}");
        // Selective: one group in fifty (2%).
        let f = Filter::Eq("group".into(), Value::Int(3));
        let mut total = 0.0;
        for q in &queries {
            let want = ix.nearest_exact(q, 10, Some(&f));
            let got = ix.nearest(q, 10, Some(&f), 16);
            assert_eq!(got.len(), want.len(), "{label}: filtered results missing");
            assert!(got.iter().all(|h| h.key % 50 == 3), "{label}: a result outside the filter");
            total += recall(&got, &want);
        }
        let r = total / queries.len() as f64;
        assert!(r >= 0.95, "{label}: filtered recall@10 {r:.3}");
    };
    check(&ix, "segments + memory");
    // A crash: replayed from the log, vectors included.
    ix.simulate_crash();
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.auto_merge = false;
    check(&ix, "after replay");
    ix.checkpoint().unwrap();
    assert!(ix.start_merge(true));
    ix.wait_merge().unwrap();
    assert_eq!(ix.stats().segments, 1);
    check(&ix, "merged");
    drop(ix);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn wrong_dimensions_are_refused() {
    let dir = std::env::temp_dir().join(format!("hinny-vectors-dims-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.upsert(&Doc { key: 1, version: 1, passages: vec![], attrs: vec![], vector: Some(vec![1.0, 0.0, 0.0]) }).unwrap();
    assert!(ix.upsert(&Doc { key: 2, version: 1, passages: vec![], attrs: vec![], vector: Some(vec![1.0, 0.0]) }).is_err());
    assert_eq!(ix.nearest(&[1.0, 0.1, 0.0], 5, None, 4).len(), 1);
    drop(ix);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Documents written before vectors were on are found, in memory and in segments, and drop out
/// once they get one; a mix of both kinds searches correctly.
#[test]
fn without_vectors_and_mixed_segments() {
    let dir = std::env::temp_dir().join(format!("hinny-vec-missing-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut rng = Rng(7);
    let (vecs, _) = data(300, &mut rng);
    let mut index = Index::open(&dir, 1).unwrap();
    let doc = |k: u64, v: Option<Vec<f32>>| Doc { key: k, version: 1, passages: vec![vec![format!("item {k}")]], attrs: vec![], vector: v };
    for k in 0..300u64 {
        // Every third without a vector.
        index.upsert(&doc(k, (k % 3 != 0).then(|| vecs[k as usize].clone()))).unwrap();
    }
    let missing: Vec<u64> = (0..300).filter(|k| k % 3 == 0).collect();
    assert_eq!(index.without_vectors(), missing);
    index.checkpoint().unwrap();
    assert_eq!(index.without_vectors(), missing);
    // Nearest neighbours come only from documents with vectors, and find themselves.
    for k in [1u64, 2, 151, 299] {
        let hits = index.nearest(&vecs[k as usize], 5, None, 64);
        assert_eq!(hits[0].key, k);
        assert!(hits.iter().all(|h| h.key % 3 != 0));
    }
    // Backfill half of them.
    for &k in missing.iter().take(50) {
        index.upsert(&doc(k, Some(vecs[k as usize].clone()))).unwrap();
    }
    assert_eq!(index.without_vectors(), missing[50..].to_vec());
    assert_eq!(index.nearest(&vecs[0], 1, None, 64)[0].key, 0);
    drop(index);
    let index = Index::open(&dir, 1).unwrap();
    assert_eq!(index.without_vectors(), missing[50..].to_vec());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// How clustered embeddings are varies by data; each segment calibrates its own probe budget, so
/// the default stays near exact on diffuse data, where a fixed budget isn't.
#[test]
fn calibrated_budget_keeps_diffuse_data_exact() {
    let dir = std::env::temp_dir().join(format!("hinny-vec-diffuse-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut rng = Rng(11);
    let dims = 64;
    let vecs: Vec<Vec<f32>> = (0..20_000).map(|_| (0..dims).map(|_| rng.normal()).collect()).collect();
    let mut ix = Index::open(&dir, 1).unwrap();
    for (k, v) in vecs.iter().enumerate() {
        ix.upsert(&Doc { key: k as u64, version: 1, passages: vec![], attrs: vec![], vector: Some(v.clone()) }).unwrap();
    }
    ix.checkpoint().unwrap();
    let queries: Vec<Vec<f32>> = (0..50).map(|_| (0..dims).map(|_| rng.normal()).collect()).collect();
    let recall = |nprobe: usize| {
        let mut sum = 0.0;
        for q in &queries {
            let want: HashSet<u64> = ix.nearest_exact(q, 10, None).iter().map(|f| f.key).collect();
            sum += ix.nearest(q, 10, None, nprobe).iter().filter(|f| want.contains(&f.key)).count() as f64 / 10.0;
        }
        sum / queries.len() as f64
    };
    let (default, fixed) = (recall(0), recall(16));
    assert!(default >= 0.97, "calibrated default: recall@10 {default}");
    assert!(fixed < default, "a fixed 16 ({fixed}) should do worse here than the calibrated budget ({default})");
    drop(ix);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn exact_filtered_searches_build_no_scan_caches() {
    let dir = std::env::temp_dir().join(format!("hinny-vectors-cold-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut rng = Rng(0xA076_1D64_78BD_642F);
    let (vectors, queries) = data(6000, &mut rng);
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.auto_merge = false;
    for (key, v) in vectors.iter().enumerate() {
        let attrs = vec![("rare".to_string(), Value::Int((key % 500 == 0) as i64))];
        ix.upsert(&Doc { key: key as u64, version: 1, passages: vec![vec!["x".into()]], attrs, vector: Some(v.clone()) }).unwrap();
        if key % 2000 == 1999 {
            ix.checkpoint().unwrap();
        }
    }
    drop(ix);
    // Reopened: nothing built yet.
    let ix = Index::open(&dir, 1).unwrap();
    assert_eq!((ix.stats().segments, ix.stats().vector_caches), (3, 0));
    let rare = Filter::Eq("rare".into(), Value::Int(1));
    let empty = Filter::Eq("rare".into(), Value::Int(7));
    for q in queries.iter().take(20) {
        assert_eq!(ix.nearest(q, 10, Some(&rare), 0), ix.nearest_exact(q, 10, Some(&rare)), "a few matches, compared exactly");
        assert!(ix.nearest(q, 10, Some(&empty), 0).is_empty());
    }
    let s = ix.stats();
    assert_eq!((s.vector_caches, s.vector_cache_bytes), (0, 0), "exact comparisons alone build no scan caches");
    // Nor does an unfiltered search: a segment stores its codes as a search reads them (since
    // format 8; before, the first search of a segment laid them out in memory).
    ix.nearest(&queries[0], 10, None, 0);
    let s = ix.stats();
    assert_eq!((s.vector_caches, s.vector_cache_bytes), (0, 0));
    drop(ix);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn vector_work_and_misses_are_counted() {
    let dir = std::env::temp_dir().join(format!("hinny-vectors-counted-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut rng = Rng(0x94D0_49BB_1331_11EB);
    let (vectors, queries) = data(12_000, &mut rng);
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.auto_merge = false;
    ix.flush_passages = 1_000_000;
    let put = |ix: &mut Index, key: usize| {
        let attrs = vec![("half".to_string(), Value::Int((key % 2) as i64))];
        // Every 10th without a vector.
        let vector = (key % 10 != 9).then(|| vectors[key].clone());
        ix.upsert(&Doc { key: key as u64, version: 1, passages: vec![vec!["x".into()]], attrs, vector }).unwrap();
    };
    // In memory only: every live vector compared once, filtered ones and deleted ones not at all.
    for key in 0..3000 {
        put(&mut ix, key);
    }
    for key in 0..300 {
        ix.delete(key as u64).unwrap();
    }
    let live_with = (300..3000).filter(|k| k % 10 != 9).count();
    let (_, _, reranked) = ix.nearest_with_stats(&queries[0], 10, None, 0);
    assert_eq!(reranked, live_with);
    let half = Filter::Eq("half".into(), Value::Int(1));
    let (_, _, reranked) = ix.nearest_with_stats(&queries[0], 10, Some(&half), 0);
    assert_eq!(reranked, (300..3000).filter(|k| k % 10 != 9 && k % 2 == 1).count());
    // Segments too, and memory: misses add up to what wasn't returned.
    ix.checkpoint().unwrap();
    for key in 3000..12_000 {
        put(&mut ix, key);
    }
    ix.checkpoint().unwrap(); // a segment of 9,000: clusters
    for key in 11_000..11_500 {
        put(&mut ix, key); // in memory again
    }
    let (mut probe, mut total) = (0, 0);
    for q in queries.iter().take(30) {
        for (k, nprobe) in [(10, 0), (100, 0), (10, 1)] {
            let truth: Vec<u64> = ix.nearest_exact(q, k, None).iter().map(|f| f.key).collect();
            let (found, m) = ix.nearest_misses(q, k, None, nprobe, &truth);
            assert_eq!(found, ix.nearest(q, k, None, nprobe), "tracing changes nothing");
            let missed = truth.iter().filter(|t| !found.iter().any(|f| f.key == **t)).count();
            assert_eq!(m.probe + m.prune + m.precision, missed, "k {k} nprobe {nprobe}: {m:?}");
            assert!(m.reranked >= found.len());
            if nprobe == 1 {
                probe += m.probe;
            }
            total += missed;
        }
    }
    assert!(probe > 0, "one cluster a segment misses some, in clusters it didn't scan");
    // (At their own budgets these searches may miss nothing: they did, on estimates, until the
    // bound was widened.)
    assert!(total >= probe, "{total} {probe}");
    drop(ix);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Filters of every share, with and against the query, and searches for more than ten: the
/// probe budget for each is the one measured for its depth when the clusters were trained, and
/// more where less passes near the query than everywhere. With the budget of the ten nearest
/// stretched by a fitted exponent, 100 results under a filter of a tenth found 0.98 of the true
/// ones on data like this, and a filter against the query 0.81 of ten.
#[test]
fn filtered_and_deep_searches_keep_their_recall() {
    let dir = std::env::temp_dir().join(format!("hinny-vectors-depth-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut rng = Rng(0xA076_1D64_78BD_642F);
    let (vectors, queries) = data(40_000, &mut rng);
    let n = vectors.len();
    // "rand": independent of the vectors, per mille. "side": a document's rank, per mille, along
    // a fixed direction, so a window of it around a query's own rank passes its neighbours more
    // than others, and a window around the opposite rank passes them less.
    let direction: Vec<f32> = (0..DIMS).map(|_| rng.normal()).collect();
    let along = |v: &[f32]| v.iter().zip(&direction).map(|(a, b)| a * b).sum::<f32>() / v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut sorted: Vec<f32> = vectors.iter().map(|v| along(v)).collect();
    sorted.sort_by(f32::total_cmp);
    let rank = |x: f32| ((sorted.partition_point(|&a| a < x) * 1000) / n).min(999) as i64;
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.auto_merge = false;
    ix.flush_passages = 1_000_000;
    ix.flush_bytes = 1 << 40;
    for (i, v) in vectors.iter().enumerate() {
        let attrs = vec![("rand".into(), Value::Int((rng.next() % 1000) as i64)), ("side".into(), Value::Int(rank(along(v))))];
        ix.upsert(&Doc { key: i as u64, version: 1, passages: vec![vec![String::new()]], attrs, vector: Some(v.clone()) }).unwrap();
    }
    ix.checkpoint().unwrap();
    assert_eq!(ix.stats().segments, 1);
    let window = |centre: i64, per_mille: i64| {
        let lo = (centre - per_mille / 2).clamp(0, 1000 - per_mille);
        Filter::Range("side".into(), Some(lo), Some(lo + per_mille - 1))
    };
    let cells: [(&str, i64); 9] = [("none", 1000), ("independent", 500), ("independent", 100), ("independent", 30), ("independent", 10), ("with", 100), ("with", 30), ("against", 300), ("against", 100)];
    let mut worst = (1.0f64, String::new());
    for (kind, per_mille) in cells {
        for k in [10, 100] {
            let mut sum = 0.0;
            for q in &queries {
                let f = match kind {
                    "none" => None,
                    "independent" => Some(Filter::Range("rand".into(), None, Some(per_mille - 1))),
                    "with" => Some(window(rank(along(q)), per_mille)),
                    _ => Some(window(999 - rank(along(q)), per_mille)),
                };
                let want = ix.nearest_exact(q, k, f.as_ref());
                let got = ix.nearest(q, k, f.as_ref(), 0);
                assert_eq!(got.len(), want.len(), "{kind} {per_mille}‰, k = {k}");
                sum += if want.is_empty() { 1.0 } else { recall(&got, &want) };
            }
            let mean = sum / queries.len() as f64;
            if mean < worst.0 {
                worst = (mean, format!("{kind} {per_mille}‰, k = {k}"));
            }
            assert!(mean >= 0.98, "{kind} {per_mille}‰, k = {k}: recall {mean:.3}");
        }
    }
    eprintln!("worst cell: {} at {:.4}", worst.1, worst.0);
    drop(ix);
    std::fs::remove_dir_all(&dir).unwrap();
}
