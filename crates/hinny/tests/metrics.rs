//! Vectors compared by Euclidean distance and by dot product: a search finds what scoring every
//! vector finds, on every path it can take (unflushed documents, a segment too small to cluster,
//! clustered lists at the calibrated budget, filters that are masked and filters compared
//! exactly, deletes, several segments of one generation, a merge, a reopening, a hybrid query),
//! and for dot product with vectors a hundred times as long as others. And the files say which
//! metric they are of: an index of one refuses another's.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use hinny::attrs::{Filter, Value};
use hinny::index::{Doc, Found, Index};
use hinny::{Database, Document, Fusion, Metric, Schema, attr};

const METRICS: [Metric; 2] = [Metric::Euclidean, Metric::Dot];

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
    /// Roughly normal (a sum of uniforms).
    fn normal(&mut self) -> f32 {
        (0..4).map(|_| self.unit()).sum::<f32>() - 2.0
    }
}

/// Clustered, like embeddings: `topics` centres (the same for every seed), each vector a centre
/// plus noise.
fn clustered(n: usize, dims: usize, topics: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut of_topics = Rng(7);
    let centres: Vec<Vec<f32>> = (0..topics).map(|_| (0..dims).map(|_| of_topics.normal()).collect()).collect();
    let mut rng = Rng(seed);
    (0..n).map(|_| centres[(rng.next() % topics as u64) as usize].iter().map(|c| c + 0.7 * rng.normal()).collect()).collect()
}

/// Each vector made longer or shorter, so that their lengths are spread over `factor` (evenly on
/// a logarithmic scale: at 100, the longest are a hundred times the shortest).
fn spread(vectors: &mut [Vec<f32>], factor: f32, seed: u64) {
    let mut rng = Rng(seed);
    for v in vectors {
        let by = factor.powf(rng.unit() - 0.5);
        v.iter_mut().for_each(|x| *x *= by);
    }
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("hinny-metrics-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// A vector as the index keeps it: in half precision.
fn kept(v: &[f32]) -> Vec<f32> {
    hinny::vector::to_f16(v).iter().map(|&h| hinny::vector::f16_to_f32(h)).collect()
}

/// A metric's score of `x` for the query `q`, worked out in double precision: the squared
/// distance negated, or the dot product.
fn score(metric: Metric, q: &[f32], x: &[f32]) -> f64 {
    match metric {
        Metric::Euclidean => -q.iter().zip(x).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>(),
        Metric::Dot => q.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum(),
        Metric::Cosine => unreachable!("cosine has its own tests"),
    }
}

/// What a query must return: every kept vector scored, the best `k`, best first.
struct Truth {
    metric: Metric,
    /// By key, the vector as kept.
    docs: HashMap<u64, Vec<f32>>,
}

impl Truth {
    fn new(metric: Metric) -> Truth {
        Truth { metric, docs: HashMap::new() }
    }

    fn add(&mut self, key: u64, v: &[f32]) {
        self.docs.insert(key, kept(v));
    }

    fn remove(&mut self, key: u64) {
        self.docs.remove(&key);
    }

    fn scores(&self, q: &[f32], keep: impl Fn(u64) -> bool) -> Vec<(u64, f64)> {
        let mut all: Vec<(u64, f64)> = self.docs.iter().filter(|(k, _)| keep(**k)).map(|(k, v)| (*k, score(self.metric, q, v))).collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all
    }

    /// `got` is the best `k` of what `keep` keeps: as many, in order, each with the score its
    /// document has. Two documents whose scores single precision can't tell apart may come in
    /// either order, which is all that "the same" leaves open.
    fn same(&self, got: &[Found], q: &[f32], k: usize, keep: impl Fn(u64) -> bool, label: &str) {
        let all = self.scores(q, keep);
        let of: HashMap<u64, f64> = all.iter().copied().collect();
        let want = &all[..k.min(all.len())];
        assert_eq!(got.len(), want.len(), "{label}: {} results of {}", got.len(), want.len());
        assert_eq!(got.iter().map(|f| f.key).collect::<HashSet<u64>>().len(), got.len(), "{label}: a document twice");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let close = |a: f64, b: f64| (a - b).abs() <= 1e-4 * (1.0 + a.abs().max(b.abs()));
            let own = *of.get(&g.key).unwrap_or_else(|| panic!("{label}: result {i} is {}, which may not be one", g.key));
            assert!(close(g.score as f64, own), "{label}: result {i}, document {}, has score {} but scores {own}", g.key, g.score);
            assert!(close(own, w.1), "{label}: result {i} is document {} at {own}; it should be {} at {}", g.key, w.0, w.1);
        }
    }
}

fn recall(got: &[Found], want: &[Found]) -> f64 {
    let w: HashSet<u64> = want.iter().map(|f| f.key).collect();
    got.iter().filter(|f| w.contains(&f.key)).count() as f64 / want.len().max(1) as f64
}

fn doc(key: u64, v: &[f32]) -> Doc {
    Doc { key, version: 1, passages: vec![vec![format!("doc {key}")]], attrs: vec![("group".into(), Value::Int((key % 50) as i64)), ("half".into(), Value::Int((key % 2) as i64)), ("rare".into(), Value::Int((key % 500) as i64))], vector: Some(v.to_vec()) }
}

fn open(d: &Path, metric: Metric) -> Index {
    let mut ix = Index::open(d, 1).unwrap();
    ix.set_metric(metric).unwrap();
    ix.auto_merge = false;
    ix.flush_passages = 10_000_000;
    ix
}

/// Sets too small to cluster are compared exactly, in memory and in a segment: the answers are
/// brute force's, for vectors as they come, spread over a hundred lengths, and with one of no
/// length among them; with filters and deletes.
#[test]
fn small_sets_are_brute_force() {
    for metric in METRICS {
        for (name, factor) in [("as generated", 1.0), ("lengths spread a hundredfold", 100.0)] {
            let label = format!("{}, {name}", metric.name());
            let d = dir(&format!("small-{}-{}", metric.name(), factor as u32));
            let dims = 40;
            let mut vectors = clustered(900, dims, 12, 11);
            spread(&mut vectors, factor, 5);
            // One with no length at all: a vector like any other to these metrics.
            vectors[17] = vec![0.0; dims];
            let mut queries = clustered(12, dims, 12, 99);
            spread(&mut queries, factor, 6);
            queries.push(vectors[3].iter().map(|x| 3.0 * x).collect());
            queries.push(vec![0.0; dims]);
            let mut truth = Truth::new(metric);
            let mut ix = open(&d, metric);
            let check = |ix: &Index, truth: &Truth, stage: &str| {
                for (i, q) in queries.iter().enumerate() {
                    for k in [1, 10, 37] {
                        let label = format!("{label}, {stage}, query {i}, {k} results");
                        truth.same(&ix.nearest(q, k, None, 0), q, k, |_| true, &label);
                        truth.same(&ix.nearest_exact(q, k, None), q, k, |_| true, &format!("{label}, exactly"));
                        let filter = Filter::Eq("group".into(), Value::Int(3));
                        truth.same(&ix.nearest(q, k, Some(&filter), 0), q, k, |key| key % 50 == 3, &format!("{label}, filtered"));
                    }
                }
            };
            for (i, v) in vectors[..600].iter().enumerate() {
                ix.upsert(&doc(i as u64, v)).unwrap();
                truth.add(i as u64, v);
            }
            check(&ix, &truth, "in memory");
            ix.checkpoint().unwrap();
            assert_eq!((ix.stats().segments, ix.stats().memory, ix.stats().vector_clusters), (1, 0, 0), "{label}: one segment, compared exactly");
            check(&ix, &truth, "in a segment");
            // More in memory, some replacing what the segment holds, and deletes in both.
            for (i, v) in vectors[600..].iter().enumerate() {
                let key = if i % 3 == 0 { i as u64 } else { 600 + i as u64 };
                ix.upsert(&doc(key, v)).unwrap();
                truth.add(key, v);
            }
            for key in [1, 50, 53, 601, 700] {
                ix.delete(key).unwrap();
                truth.remove(key);
            }
            check(&ix, &truth, "a segment, memory and deletes");
            assert!(ix.without_vectors().is_empty(), "{label}: every document has its vector, the one of no length too");
            drop(ix);
            std::fs::remove_dir_all(&d).unwrap();
        }
    }
}

/// How well and at what cost the calibrated default budget searches `vectors`: recall@10
/// against scoring every vector, and the share of the codes a search estimates.
fn calibrated(metric: Metric, name: &str, vectors: &[Vec<f32>], queries: &[Vec<f32>]) -> (f64, f64) {
    let d = dir(&format!("calibrated-{}-{name}", metric.name()));
    let mut ix = open(&d, metric);
    let mut truth = Truth::new(metric);
    for (i, v) in vectors.iter().enumerate() {
        ix.upsert(&Doc { key: i as u64, version: 1, passages: vec![], attrs: vec![], vector: Some(v.clone()) }).unwrap();
        truth.add(i as u64, v);
    }
    ix.checkpoint().unwrap();
    let s = ix.stats();
    assert_eq!((s.segments, s.memory, s.vector_generations), (1, 0, 1), "{name}");
    let (mut found, mut estimated) = (0.0, 0usize);
    for (i, q) in queries.iter().enumerate() {
        let want = ix.nearest_exact(q, 10, None);
        // The reference is right: scoring every vector in the index is scoring every vector here.
        if i < 3 {
            truth.same(&want, q, 10, |_| true, &format!("{} {name}: exact search, query {i}", metric.name()));
        }
        let (got, codes, _) = ix.nearest_with_stats(q, 10, None, 0);
        // What the lists find comes with its own score, best first, whether or not it is
        // everything there was to find.
        if i < 10 {
            let of: HashMap<u64, f64> = truth.scores(q, |_| true).into_iter().collect();
            assert_eq!(got.len(), 10);
            assert!(got.windows(2).all(|w| w[0].score >= w[1].score), "{} {name}: query {i} out of order", metric.name());
            for f in &got {
                assert!((f.score as f64 - of[&f.key]).abs() <= 1e-4 * (1.0 + of[&f.key].abs()), "{} {name}: document {} has score {} but scores {}", metric.name(), f.key, f.score, of[&f.key]);
            }
        }
        found += recall(&got, &want);
        estimated += codes;
    }
    drop(ix);
    std::fs::remove_dir_all(&d).unwrap();
    (found / queries.len() as f64, estimated as f64 / queries.len() as f64 / vectors.len() as f64)
}

/// 100,000 clustered vectors in one segment, searched with the budget it calibrated for itself:
/// recall@10 of 0.98 or more, from a small share of the codes. For dot product also with the
/// vectors' lengths spread over a factor of a hundred, queries spread the same and queries of
/// one length.
#[test]
fn the_calibrated_budget_finds_the_nearest_of_a_hundred_thousand() {
    let dims = 64;
    let vectors = clustered(100_000, dims, 300, 21);
    let queries = clustered(100, dims, 300, 22);
    for metric in METRICS {
        let (recall, codes) = calibrated(metric, "plain", &vectors, &queries);
        println!("{}: recall@10 {recall:.4}, {:.1}% of the codes", metric.name(), codes * 100.0);
        assert!(recall >= 0.98, "{}: recall@10 {recall:.4}", metric.name());
        assert!(codes <= 0.2, "{}: a search estimates {:.1}% of the codes", metric.name(), codes * 100.0);
    }
    // Dot product's hard case: the vectors of the greatest product are the long ones, which
    // lists by distance scatter and lists by direction keep together.
    let mut long = vectors.clone();
    spread(&mut long, 100.0, 31);
    let mut like = queries.clone();
    spread(&mut like, 100.0, 32);
    for (name, queries) in [("lengths spread a hundredfold", &like), ("spread, queries of one length", &queries)] {
        let (recall, codes) = calibrated(Metric::Dot, if name.starts_with("lengths") { "spread" } else { "spread-unit" }, &long, queries);
        println!("dot, {name}: recall@10 {recall:.4}, {:.1}% of the codes", codes * 100.0);
        assert!(recall >= 0.98, "dot, {name}: recall@10 {recall:.4}");
        assert!(codes <= 0.2, "dot, {name}: a search estimates {:.1}% of the codes", codes * 100.0);
    }
}

/// Lengths that go with the topic (every vector of one a hundred times as long as those of
/// another): what dot product finds is then the long topics' vectors, wherever the query
/// points, and the budget has to say so.
#[test]
fn dot_product_where_whole_topics_are_long() {
    let (dims, topics) = (64, 300);
    let mut of_topics = Rng(7);
    let centres: Vec<Vec<f32>> = (0..topics).map(|_| (0..dims).map(|_| of_topics.normal()).collect()).collect();
    let lengths: Vec<f32> = (0..topics).map(|_| 100f32.powf(of_topics.unit() - 0.5)).collect();
    let make = |n: usize, seed: u64| -> Vec<Vec<f32>> {
        let mut rng = Rng(seed);
        (0..n)
            .map(|_| {
                let t = (rng.next() % topics as u64) as usize;
                centres[t].iter().map(|c| lengths[t] * (c + 0.7 * rng.normal())).collect()
            })
            .collect()
    };
    let (vectors, queries) = (make(60_000, 41), make(60, 42));
    let (recall, codes) = calibrated(Metric::Dot, "topics", &vectors, &queries);
    println!("dot, lengths by topic: recall@10 {recall:.4}, {:.1}% of the codes", codes * 100.0);
    assert!(recall >= 0.98, "recall@10 {recall:.4}");
}

/// Filters of several shares over one clustered segment: half and a twentieth pass (probed,
/// the second under the filter's bits), one in five hundred (compared exactly). Every result
/// passes, there are as many as there can be, and they are the nearest that pass.
#[test]
fn filters_of_every_share() {
    for metric in METRICS {
        let d = dir(&format!("filters-{}", metric.name()));
        let dims = 64;
        let mut vectors = clustered(40_000, dims, 120, 51);
        let mut queries = clustered(40, dims, 120, 52);
        if metric == Metric::Dot {
            spread(&mut vectors, 100.0, 53);
            spread(&mut queries, 100.0, 54);
        }
        let mut ix = open(&d, metric);
        for (i, v) in vectors.iter().enumerate() {
            let attrs = vec![("half".into(), Value::Int((i % 2) as i64)), ("twentieth".into(), Value::Int((i % 20) as i64)), ("rare".into(), Value::Int((i % 500) as i64))];
            ix.upsert(&Doc { key: i as u64, version: 1, passages: vec![], attrs, vector: Some(v.clone()) }).unwrap();
        }
        ix.checkpoint().unwrap();
        assert_eq!(ix.stats().vector_generations, 1);
        // (name, the filter, which keys pass, whether its matches are compared exactly.)
        let filters: [(&str, Filter, u64, bool); 3] = [
            ("half", Filter::Eq("half".into(), Value::Int(1)), 2, false),
            ("a twentieth", Filter::Eq("twentieth".into(), Value::Int(7)), 20, false),
            ("one in five hundred", Filter::Eq("rare".into(), Value::Int(7)), 500, true),
        ];
        for (name, filter, modulus, exactly) in &filters {
            let passes = |key: u64| key % modulus == if *modulus == 2 { 1 } else { 7 };
            for k in [10, 100] {
                let label = format!("{}, {name} pass, {k} results", metric.name());
                let (mut found, mut estimated) = (0.0, 0usize);
                for q in &queries {
                    let want = ix.nearest_exact(q, k, Some(filter));
                    let (got, codes, _) = ix.nearest_with_stats(q, k, Some(filter), 0);
                    assert!(got.iter().all(|f| passes(f.key)), "{label}: a result outside the filter");
                    assert_eq!(got.len(), want.len(), "{label}: results missing");
                    assert_eq!(want.len(), k.min(40_000 / *modulus as usize), "{label}");
                    found += recall(&got, &want);
                    estimated += codes;
                }
                let r = found / queries.len() as f64;
                assert!(r >= 0.98, "{label}: recall {r:.4}");
                // Compared exactly, no code is estimated; probed, some are.
                assert_eq!(estimated == 0, *exactly, "{label}: {estimated} codes estimated");
                if *exactly {
                    assert_eq!(r, 1.0, "{label}: compared exactly");
                }
            }
        }
        // And without one: every list probed finds what exact search finds, a few probed finds
        // most of it from fewer codes.
        let (mut all, mut few, mut codes_all, mut codes_few) = (0.0, 0.0, 0usize, 0usize);
        for q in &queries {
            let want = ix.nearest_exact(q, 10, None);
            let (got, codes, _) = ix.nearest_with_stats(q, 10, None, usize::MAX);
            all += recall(&got, &want);
            codes_all += codes;
            let (got, codes, _) = ix.nearest_with_stats(q, 10, None, 1);
            few += recall(&got, &want);
            codes_few += codes;
        }
        let (all, few) = (all / queries.len() as f64, few / queries.len() as f64);
        assert!(all >= 0.999, "{}: every list probed, recall {all:.4}", metric.name());
        assert!(few <= all && codes_few * 10 < codes_all, "{}: one list probed: recall {few:.4}, {codes_few} codes of {codes_all}", metric.name());
        drop(ix);
        std::fs::remove_dir_all(&d).unwrap();
    }
}

/// Three segments of one generation and unflushed documents, with deletes in all of them:
/// searched as one, replayed from the log after a crash, merged (which copies the lists as
/// they are and changes no answer), and reopened.
#[test]
fn segments_memory_deletes_a_crash_a_merge_and_a_reopening() {
    for metric in METRICS {
        let d = dir(&format!("segments-{}", metric.name()));
        let dims = 64;
        let mut vectors = clustered(30_000, dims, 60, 61);
        let mut queries = clustered(50, dims, 60, 62);
        if metric == Metric::Dot {
            spread(&mut vectors, 100.0, 63);
            spread(&mut queries, 100.0, 64);
        }
        let mut ix = open(&d, metric);
        for (i, v) in vectors.iter().enumerate() {
            ix.upsert(&doc(i as u64, v)).unwrap();
            // Three segments, and the rest in memory.
            if i % 9000 == 8999 {
                ix.checkpoint().unwrap();
            }
        }
        for i in (0..30_000).step_by(7) {
            ix.delete(i as u64).unwrap();
        }
        let s = ix.stats();
        assert_eq!((s.segments, s.memory, s.vector_generations), (3, 3000, 1), "{}: {s:?}", metric.name());

        let check = |ix: &Index, label: &str| {
            let label = format!("{}, {label}", metric.name());
            let mut total = 0.0;
            for q in &queries {
                let want = ix.nearest_exact(q, 10, None);
                let got = ix.nearest(q, 10, None, 0);
                assert!(got.iter().all(|f| f.key % 7 != 0), "{label}: a deleted document came back");
                assert!(want.iter().all(|f| f.key % 7 != 0), "{label}: exact search returned a deleted document");
                total += recall(&got, &want);
            }
            let r = total / queries.len() as f64;
            assert!(r >= 0.98, "{label}: recall@10 {r:.4}");
            // One group in fifty (2%), and one document in five hundred.
            for (name, filter, modulus) in [("2%", Filter::Eq("group".into(), Value::Int(3)), 50), ("0.2%", Filter::Eq("rare".into(), Value::Int(3)), 500)] {
                let mut total = 0.0;
                for q in &queries {
                    let want = ix.nearest_exact(q, 10, Some(&filter));
                    let got = ix.nearest(q, 10, Some(&filter), 0);
                    assert_eq!(got.len(), want.len(), "{label}, {name}: filtered results missing");
                    assert!(got.iter().all(|h| h.key % modulus == 3 && h.key % 7 != 0), "{label}, {name}: a result outside the filter, or deleted");
                    total += recall(&got, &want);
                }
                let r = total / queries.len() as f64;
                assert!(r >= 0.98, "{label}, {name}: filtered recall@10 {r:.4}");
            }
        };
        check(&ix, "segments and memory");
        // What scoring every vector finds is what brute force finds here, over all of them.
        let mut truth = Truth::new(metric);
        for (i, v) in vectors.iter().enumerate().filter(|(i, _)| i % 7 != 0) {
            truth.add(i as u64, v);
        }
        for q in queries.iter().take(5) {
            truth.same(&ix.nearest_exact(q, 10, None), q, 10, |_| true, &format!("{}: exact search", metric.name()));
        }

        // A crash: replayed from the log, vectors as they came, by the metric the manifest names.
        ix.simulate_crash();
        let mut ix = Index::open(&d, 1).unwrap();
        assert_eq!(ix.metric(), metric);
        ix.auto_merge = false;
        ix.flush_passages = 10_000_000;
        assert_eq!(ix.stats().memory, 3000);
        check(&ix, "after replay");
        assert_eq!(ix.view().document(29_999).unwrap().vector.unwrap(), kept(&vectors[29_999]), "{}: a replayed vector is the one written", metric.name());

        ix.checkpoint().unwrap();
        assert_eq!((ix.stats().segments, ix.stats().vector_generations), (4, 1));
        let before: Vec<Vec<Found>> = queries.iter().map(|q| ix.nearest(q, 10, None, 0)).collect();
        assert!(ix.start_merge(true));
        ix.wait_merge().unwrap();
        assert_eq!((ix.stats().segments, ix.stats().vector_generations, ix.stats().deleted), (1, 1, 0));
        check(&ix, "merged");
        let after: Vec<Vec<Found>> = queries.iter().map(|q| ix.nearest(q, 10, None, 0)).collect();
        assert_eq!(after, before, "{}: a merge within a generation changes no answer", metric.name());
        drop(ix);
        let ix = Index::open(&d, 1).unwrap();
        assert_eq!(ix.metric(), metric);
        check(&ix, "reopened");
        let reopened: Vec<Vec<Found>> = queries.iter().map(|q| ix.nearest(q, 10, None, 0)).collect();
        assert_eq!(reopened, before, "{}: nor does reopening", metric.name());
        drop(ix);
        std::fs::remove_dir_all(&d).unwrap();
    }
}

/// A generation its vectors have outgrown trains the next, by the same metric, and a segment
/// of the earlier one moves to it when merged.
#[test]
fn generations_follow_the_metric() {
    for metric in METRICS {
        let d = dir(&format!("generations-{}", metric.name()));
        let dims = 48;
        let mut ix = open(&d, metric);
        let mut queries = clustered(40, dims, 80, 70);
        if metric == Metric::Dot {
            spread(&mut queries, 100.0, 71);
        }
        let put = |ix: &mut Index, first: u64, n: usize, seed: u64| {
            let mut vectors = clustered(n, dims, 80, seed);
            if metric == Metric::Dot {
                spread(&mut vectors, 100.0, seed + 100);
            }
            for (i, v) in vectors.iter().enumerate() {
                ix.upsert(&doc(first + i as u64, v)).unwrap();
            }
            ix.checkpoint().unwrap();
        };
        let recall_of = |ix: &Index| queries.iter().map(|q| recall(&ix.nearest(q, 10, None, 0), &ix.nearest_exact(q, 10, None))).sum::<f64>() / queries.len() as f64;
        put(&mut ix, 0, 4500, 72);
        let small = ix.stats().vector_clusters;
        put(&mut ix, 10_000, 20_000, 73);
        let s = ix.stats();
        assert_eq!((s.segments, s.vector_generations), (2, 2), "{}", metric.name());
        assert!(s.vector_clusters > small);
        let r = recall_of(&ix);
        assert!(r >= 0.98, "{}: two generations, recall@10 {r:.4}", metric.name());
        ix.start_merge(true);
        ix.wait_merge().unwrap();
        let s = ix.stats();
        assert_eq!((s.segments, s.vector_generations), (1, 1));
        let r = recall_of(&ix);
        assert!(r >= 0.98, "{}: merged into the later one, recall@10 {r:.4}", metric.name());
        drop(ix);
        std::fs::remove_dir_all(&d).unwrap();
    }
}

const WORDS: [&str; 8] = ["lease", "apartment", "invoice", "meeting", "holiday", "budget", "review", "contract"];

fn schema(dims: usize, metric: Metric) -> Schema {
    Schema::new().text("body", 1.0).int("year").vector_with(dims, metric)
}

/// Through a namespace: a vector query's scores are the metric's, a hybrid query fuses them
/// with text (they are higher for nearer, as fusion needs), with a filter and by rank; a
/// snapshot keeps its answers; a fork and a reopening keep the metric, and nothing changes it.
#[test]
fn namespaces_hybrid_queries_snapshots_and_forks() {
    for metric in METRICS {
        let root = dir(&format!("hybrid-{}", metric.name()));
        let dims = 32;
        let mut vectors = clustered(5_000, dims, 20, 81);
        if metric == Metric::Dot {
            spread(&mut vectors, 100.0, 82);
        }
        let db = Database::open(&root).unwrap();
        let ns = db.namespace("docs", schema(dims, metric)).unwrap();
        assert_eq!(ns.schema().metric, metric);
        let mut truth = Truth::new(metric);
        for (i, v) in vectors.iter().enumerate() {
            let body = format!("{} {} note {i}", WORDS[i % WORDS.len()], WORDS[(i / 3) % WORDS.len()]);
            ns.upsert(Document::new(i as u64).text("body", &body).int("year", 2000 + (i % 25) as i64).vector(v.clone())).unwrap();
            truth.add(i as u64, v);
            // A segment (too small to cluster: compared exactly) and the rest in memory.
            if i == 3_500 {
                ns.flush().unwrap();
            }
        }
        let found = |hits: &[hinny::Hit]| -> Vec<Found> { hits.iter().map(|h| Found { key: h.key, passage: 0, score: h.vector_score.unwrap() }).collect() };
        for (i, q) in vectors.iter().step_by(701).enumerate() {
            let label = format!("{}, query {i}", metric.name());
            // The vector alone: brute force's answer, and the score is the vector's.
            let hits = ns.query().vector(q.clone()).limit(10).run().unwrap();
            truth.same(&found(&hits), q, 10, |_| true, &format!("{label}, vector alone"));
            assert!(hits.iter().all(|h| h.score == h.vector_score.unwrap() && h.text_score.is_none()), "{label}");
            if metric == Metric::Euclidean {
                assert!(hits.iter().all(|h| h.score <= 0.0), "{label}: a squared distance negated is never above 0");
            }
            // Filtered.
            let hits = ns.query().vector(q.clone()).filter(attr("year").gte(2020)).limit(10).run().unwrap();
            truth.same(&found(&hits), q, 10, |key| key % 25 >= 20, &format!("{label}, filtered"));
            // With text: every hit the vector search found carries the vector's own score, and
            // with all the weight on the vectors the ranking is theirs.
            let (hits, stats) = ns.query().text("apartment lease").vector(q.clone()).limit(10).explain().unwrap();
            let stats = stats.expect("a hybrid query says how it fused");
            assert_eq!(hits.len(), 10, "{label}");
            assert!(hits.iter().any(|h| h.text_score.is_some()) && hits.iter().any(|h| h.vector_score.is_some()), "{label}: both retrievers contribute");
            let of: HashMap<u64, f64> = truth.scores(q, |_| true).into_iter().collect();
            for h in hits.iter().filter(|h| h.vector_score.is_some()) {
                let (got, own) = (h.vector_score.unwrap() as f64, of[&h.key]);
                assert!((got - own).abs() <= 1e-4 * (1.0 + own.abs()), "{label}: document {} has vector score {got}, scores {own}", h.key);
            }
            let (low, high) = stats.vector.scores.unwrap();
            let best = truth.scores(q, |_| true);
            assert!((high as f64 - best[0].1).abs() <= 1e-4 * (1.0 + best[0].1.abs()) && low <= high, "{label}: the vector side's scores run from {low} to {high}");
            let by_vector = ns.query().text("apartment lease").vector(q.clone()).text_weight(0.0).limit(10).run().unwrap();
            let alone = ns.query().vector(q.clone()).limit(10).run().unwrap();
            assert_eq!(by_vector.iter().map(|h| h.key).collect::<Vec<_>>(), alone.iter().map(|h| h.key).collect::<Vec<_>>(), "{label}: all the weight on the vectors");
            // By rank, and filtered.
            let hits = ns.query().text("budget review").vector(q.clone()).fusion(Fusion::Rrf { k: 60.0 }).filter(attr("year").gte(2020)).limit(20).run().unwrap();
            assert_eq!(hits.len(), 20, "{label}");
            assert!(hits.iter().all(|h| matches!(h.attr("year"), Some(Value::Int(y)) if *y >= 2020)), "{label}: a hit outside the filter");
        }
        // A snapshot answers as of when it was taken; the namespace sees what came since: the
        // longest vector is its own best match by either metric, and once it is gone a copy a
        // little longer is the best there is.
        let longest = (0..vectors.len()).max_by(|a, b| score(Metric::Dot, &vectors[*a], &vectors[*a]).total_cmp(&score(Metric::Dot, &vectors[*b], &vectors[*b]))).unwrap();
        let q = vectors[longest].clone();
        let snapshot = ns.snapshot();
        let before: Vec<u64> = snapshot.query().vector(q.clone()).limit(5).run().unwrap().iter().map(|h| h.key).collect();
        assert_eq!(before[0], longest as u64);
        ns.upsert(Document::new(900_000).text("body", "new").int("year", 2030).vector(q.iter().map(|x| 1.01 * x).collect())).unwrap();
        ns.delete(before[0]).unwrap();
        assert_eq!(snapshot.query().vector(q.clone()).limit(5).run().unwrap().iter().map(|h| h.key).collect::<Vec<_>>(), before, "{}: a snapshot", metric.name());
        let now: Vec<u64> = ns.query().vector(q.clone()).limit(5).run().unwrap().iter().map(|h| h.key).collect();
        assert!(now.contains(&900_000) && !now.contains(&before[0]), "{}: {now:?}", metric.name());
        drop(snapshot);

        // A fork is of the same metric and answers the same.
        let fork = ns.fork("copy").unwrap();
        assert_eq!(fork.schema().metric, metric);
        let keys = |ns: &hinny::Namespace| -> Vec<(u64, u32)> { ns.query().vector(q.clone()).limit(10).run().unwrap().iter().map(|h| (h.key, h.score.to_bits())).collect() };
        assert_eq!(keys(&fork), keys(&ns), "{}: a fork", metric.name());
        drop(fork);

        // Reopened as it is, and read-only; under another metric it is refused, cosine included.
        let want = keys(&ns);
        drop(ns);
        let ns = db.open_namespace("docs").unwrap();
        assert_eq!(ns.schema().metric, metric);
        assert_eq!(keys(&ns), want, "{}: reopened", metric.name());
        let reader = db.open_read_only("docs").unwrap();
        assert_eq!(reader.schema().metric, metric);
        assert_eq!(keys(&reader), want, "{}: read-only", metric.name());
        drop((ns, reader));
        for other in [Metric::Cosine, Metric::Euclidean, Metric::Dot].into_iter().filter(|m| *m != metric) {
            let e = db.namespace("docs", schema(dims, other)).err().expect("another metric is another schema");
            assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput, "{e}");
            assert!(e.to_string().contains("has another schema"), "{e}");
        }
        assert!(db.namespace("docs", Schema::new().text("body", 1.0).int("year").vector(dims)).is_err(), "{}: `vector` is cosine", metric.name());
        std::fs::remove_dir_all(&root).unwrap();
    }
}

fn version_of(segment: &Path) -> u32 {
    u32::from_le_bytes(std::fs::read(segment).unwrap()[4..8].try_into().unwrap())
}

fn segments_in(d: &Path) -> Vec<PathBuf> {
    let mut all: Vec<PathBuf> = std::fs::read_dir(d).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "swt")).collect();
    all.sort();
    all
}

/// What the files say. A cosine namespace is written as it was before there were metrics
/// (schema 2, manifest 7, segments of version 8, no metric named), so an earlier release
/// still reads it; another metric is named in a schema of version 5, a manifest of format 8
/// and segments of version 9, each of which an earlier release refuses by its number.
#[test]
fn files_name_the_metric_and_cosine_files_are_what_they_were() {
    let root = dir("files");
    let db = Database::open(&root).unwrap();
    let vectors = clustered(200, 8, 5, 91);
    for (name, metric) in [("cosine", Metric::Cosine), ("euclidean", Metric::Euclidean), ("dot", Metric::Dot)] {
        let ns = db.namespace(name, if metric == Metric::Cosine { Schema::new().text("body", 1.0).vector(8) } else { Schema::new().text("body", 1.0).vector_with(8, metric) }).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            ns.upsert(Document::new(i as u64).text("body", "words").vector(v.clone())).unwrap();
        }
        ns.flush().unwrap();
        drop(ns);
        let d = root.join(name);
        let schema = std::fs::read_to_string(d.join("schema")).unwrap();
        let manifest = std::fs::read_to_string(d.join("manifest")).unwrap();
        let versions: Vec<u32> = segments_in(&d).iter().map(|s| version_of(s)).collect();
        if metric == Metric::Cosine {
            assert!(schema.starts_with("fugudb schema 2\n") && schema.contains("\nvector 8\n"), "{schema}");
            assert!(manifest.starts_with("fugudb-index 7\n") && !manifest.contains("metric"), "{manifest}");
            assert_eq!(versions, [8]);
        } else {
            assert!(schema.starts_with("fugudb schema 5\n") && schema.contains(&format!("\nvector 8 {name}\n")), "{schema}");
            assert!(manifest.starts_with("fugudb-index 8\n") && manifest.contains(&format!("\nmetric {name}\n")), "{manifest}");
            assert_eq!(versions, [9]);
        }
    }
    // Documents without vectors make a segment that is of no metric: version 8 in any namespace.
    let ns = db.namespace("none", Schema::new().text("body", 1.0).vector_with(8, Metric::Dot)).unwrap();
    ns.upsert(Document::new(1).text("body", "words")).unwrap();
    ns.flush().unwrap();
    drop(ns);
    assert_eq!(segments_in(&root.join("none")).iter().map(|s| version_of(s)).collect::<Vec<_>>(), [8]);

    // An index refuses a segment of another metric than its own: the dot product namespace's
    // segment in the cosine namespace's place.
    let (cosine, dot) = (segments_in(&root.join("cosine")), segments_in(&root.join("dot")));
    std::fs::copy(&dot[0], &cosine[0]).unwrap();
    for e in [Index::open(&root.join("cosine"), 1).err().unwrap(), Index::open_read_only(&root.join("cosine"), 1).err().unwrap()] {
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{e}");
        assert!(e.to_string().contains("compared by dot") && e.to_string().contains("by cosine"), "{e}");
    }
    // And a schema beside an index of another metric is refused, read-only too.
    std::fs::copy(root.join("dot").join("schema"), root.join("euclidean").join("schema")).unwrap();
    let e = db.open_namespace("euclidean").err().unwrap();
    assert!(e.to_string().contains("euclidean") && e.to_string().contains("dot"), "{e}");
    let e = db.open_read_only("euclidean").err().unwrap();
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{e}");
    std::fs::remove_dir_all(&root).unwrap();
}

/// A metric is set while an index holds no vectors and not after, is kept through a crash
/// before anything is flushed (the log is replayed by it), and a read-only index can't set one.
#[test]
fn a_metric_is_set_once() {
    let d = dir("once");
    let mut ix = Index::open(&d, 1).unwrap();
    assert_eq!(ix.metric(), Metric::Cosine);
    // Documents without vectors don't fix it.
    ix.upsert(&Doc { key: 1, version: 1, passages: vec![vec!["text".into()]], attrs: vec![], vector: None }).unwrap();
    ix.set_metric(Metric::Euclidean).unwrap();
    ix.set_metric(Metric::Dot).unwrap();
    ix.upsert(&Doc { key: 2, version: 1, passages: vec![], attrs: vec![], vector: Some(vec![3.0, 4.0, 0.0]) }).unwrap();
    ix.set_metric(Metric::Dot).unwrap();
    for other in [Metric::Cosine, Metric::Euclidean] {
        let e = ix.set_metric(other).err().unwrap();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
        assert!(e.to_string().contains("compared by dot") && e.to_string().contains(other.name()), "{e}");
    }
    assert_eq!(ix.nearest(&[1.0, 1.0, 0.0], 1, None, 0)[0].score, 7.0);
    // Killed with the vector only in the log.
    ix.simulate_crash();
    let ix = Index::open(&d, 1).unwrap();
    assert_eq!(ix.metric(), Metric::Dot);
    assert_eq!(ix.nearest(&[1.0, 1.0, 0.0], 1, None, 0)[0].score, 7.0, "replayed as it was written, not normalized");
    assert_eq!(ix.view().document(2).unwrap().vector.unwrap(), [3.0, 4.0, 0.0]);
    drop(ix);
    let mut reader = Index::open_read_only(&d, 1).unwrap();
    assert_eq!(reader.metric(), Metric::Dot);
    assert_eq!(reader.set_metric(Metric::Cosine).err().unwrap().kind(), std::io::ErrorKind::PermissionDenied);
    drop(reader);
    std::fs::remove_dir_all(&d).unwrap();
}

/// Scores are what the metric says, exactly, on vectors small enough to check by hand; and
/// what each metric takes and refuses.
#[test]
fn scores_and_what_a_vector_may_be() {
    let vectors = [vec![3.0f32, 4.0], vec![0.3, 0.4], vec![-1.0, 2.0], vec![0.0, 0.0]];
    let q = [1.0f32, 2.0];
    for (metric, want) in [
        // Squared distances negated: 8, 3.05, 4, 5.
        (Metric::Euclidean, vec![(1u64, -3.05f32), (2, -4.0), (3, -5.0), (0, -8.0)]),
        // Products: 11, 1.1, 3, 0.
        (Metric::Dot, vec![(0, 11.0), (2, 3.0), (1, 1.1), (3, 0.0)]),
    ] {
        let d = dir(&format!("scores-{}", metric.name()));
        let mut ix = open(&d, metric);
        for (key, v) in vectors.iter().enumerate() {
            ix.upsert(&Doc { key: key as u64, version: 1, passages: vec![], attrs: vec![], vector: Some(v.clone()) }).unwrap();
        }
        for stage in ["in memory", "in a segment"] {
            let got = ix.nearest(&q, 4, None, 0);
            assert_eq!(got.iter().map(|f| f.key).collect::<Vec<_>>(), want.iter().map(|w| w.0).collect::<Vec<_>>(), "{} {stage}", metric.name());
            for (g, w) in got.iter().zip(&want) {
                assert!((g.score - w.1).abs() < 1e-3, "{} {stage}: document {} scores {}, not {}", metric.name(), g.key, g.score, w.1);
            }
            ix.checkpoint().unwrap();
        }
        // A query of no length is one like any other: by distance the vectors nearest the
        // origin, by product every vector alike (0).
        let got = ix.nearest(&[0.0, 0.0], 2, None, 0);
        assert_eq!(got.len(), 2);
        match metric {
            Metric::Euclidean => assert_eq!(got.iter().map(|f| f.key).collect::<Vec<_>>(), [3, 1]),
            _ => assert!(got.iter().all(|f| f.score == 0.0)),
        }
        // Kept as they come, in half precision: a component past its end is refused, with the
        // document and the limit named; one that isn't a number, as ever.
        let e = ix.upsert(&Doc { key: 9, version: 1, passages: vec![], attrs: vec![], vector: Some(vec![70_000.0, 1.0]) }).err().unwrap();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
        assert!(e.to_string().contains("document 9") && e.to_string().contains("65504"), "{e}");
        assert!(ix.upsert(&Doc { key: 9, version: 1, passages: vec![], attrs: vec![], vector: Some(vec![f32::NAN, 1.0]) }).is_err());
        ix.upsert(&Doc { key: 9, version: 1, passages: vec![], attrs: vec![], vector: Some(vec![-65_504.0, 65_504.0]) }).unwrap();
        assert!(ix.nearest(&q, 5, None, 0).iter().all(|f| f.score.is_finite()));
        assert!(ix.nearest(&[f32::INFINITY, 0.0], 5, None, 0).is_empty(), "a query that isn't finite finds nothing");
        drop(ix);
        std::fs::remove_dir_all(&d).unwrap();
    }
    // Cosine takes vectors of any size (it stores them at length 1) and none of no length.
    let d = dir("scores-cosine");
    let mut ix = Index::open(&d, 1).unwrap();
    ix.upsert(&Doc { key: 1, version: 1, passages: vec![], attrs: vec![], vector: Some(vec![300_000.0, 400_000.0]) }).unwrap();
    ix.upsert(&Doc { key: 2, version: 1, passages: vec![], attrs: vec![], vector: Some(vec![0.0, 0.0]) }).unwrap();
    assert_eq!(ix.without_vectors(), [2]);
    let got = ix.nearest(&[3.0, 4.0], 2, None, 0);
    assert_eq!(got.len(), 1);
    assert!((got[0].score - 1.0).abs() < 1e-4, "{}", got[0].score);
    drop(ix);
    std::fs::remove_dir_all(&d).unwrap();
}

/// An index from before there were metrics is cosine's: it opens as that, and a schema that
/// asks for another metric over its vectors is refused.
#[test]
fn an_earlier_index_is_cosine() {
    let from = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v0.3");
    let to = dir("earlier");
    fn walk(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for e in std::fs::read_dir(from).unwrap().flatten() {
            if e.file_type().unwrap().is_dir() {
                walk(&e.path(), &to.join(e.file_name()));
            } else {
                std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
            }
        }
    }
    walk(&from, &to);
    let reader = Index::open_read_only(&to.join("docs"), 2).unwrap();
    assert_eq!(reader.metric(), Metric::Cosine);
    drop(reader);
    let db = Database::open(&to).unwrap();
    let old = Schema::new().text("title", 2.0).text("body", 1.0).int("year").string("kind");
    let e = db.namespace("docs", old.clone().vector_with(8, Metric::Dot)).err().expect("its vectors are cosine's");
    assert!(e.to_string().contains("has another schema"), "{e}");
    let ns = db.namespace("docs", old.vector(8)).unwrap();
    assert_eq!(ns.schema().metric, Metric::Cosine);
    drop(ns);
    std::fs::remove_dir_all(&to).unwrap();
}
