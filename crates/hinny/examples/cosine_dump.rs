//! Every result of a fixed set of cosine searches, as text: run before and after a change to the
//! vector search, the two files must be the same (the same documents in the same order with the
//! same scores to the bit).
//!
//!   cargo run --release -p hinny --example cosine_dump -- OUT.txt

use std::fmt::Write as _;

use hinny::attrs::{Filter, Value};
use hinny::index::{Doc, Found, Index};
use hinny::{Database, Document, Fusion, Schema, attr};

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

fn clustered(n: usize, dims: usize, topics: usize, noise: f32, rng: &mut Rng) -> Vec<Vec<f32>> {
    let centres: Vec<Vec<f32>> = (0..topics).map(|_| (0..dims).map(|_| rng.normal()).collect()).collect();
    (0..n)
        .map(|_| {
            let t = &centres[(rng.next() % topics as u64) as usize];
            t.iter().map(|c| c + noise * rng.normal()).collect()
        })
        .collect()
}

fn line(out: &mut String, label: &str, q: usize, found: &[Found]) {
    write!(out, "{label} q{q}:").unwrap();
    for f in found {
        write!(out, " {}/{}:{:08x}", f.key, f.passage, f.score.to_bits()).unwrap();
    }
    out.push('\n');
}

fn searches(out: &mut String, label: &str, ix: &Index, queries: &[Vec<f32>]) {
    let filters: Vec<(&str, Filter)> = vec![
        ("2%", Filter::Eq("group".into(), Value::Int(3))),
        ("10%", Filter::Eq("bucket".into(), Value::Int(1))),
        ("50%", Filter::Eq("half".into(), Value::Int(0))),
        ("0.1%", Filter::Eq("rare".into(), Value::Int(5))),
    ];
    for (q, v) in queries.iter().enumerate() {
        for (k, nprobe) in [(10, 0), (100, 0), (10, 1), (10, 16), (1000, 0), (10, usize::MAX)] {
            line(out, &format!("{label} k{k} nprobe{nprobe}"), q, &ix.nearest(v, k, None, nprobe));
        }
        line(out, &format!("{label} exact"), q, &ix.nearest_exact(v, 10, None));
        for (name, f) in &filters {
            for (k, nprobe) in [(10, 0), (100, 0), (10, 8)] {
                line(out, &format!("{label} filter{name} k{k} nprobe{nprobe}"), q, &ix.nearest(v, k, Some(f), nprobe));
            }
            line(out, &format!("{label} filter{name} exact"), q, &ix.nearest_exact(v, 10, Some(f)));
        }
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: cosine_dump OUT.txt");
    let mut out = String::new();
    let root = std::env::temp_dir().join(format!("hinny-cosine-dump-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);

    // Segments of one generation, unflushed documents, deletes; then merged; then reopened.
    {
        let dims = 128;
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let vectors = clustered(60_000, dims, 60, 0.6, &mut rng);
        let queries = clustered(40, dims, 60, 0.6, &mut Rng(0x1234_5678_9ABC_DEF1));
        let queries: Vec<Vec<f32>> = queries.into_iter().chain(vectors.iter().step_by(4001).cloned()).collect();
        let dir = root.join("a");
        let mut ix = Index::open(&dir, 1).unwrap();
        ix.auto_merge = false;
        ix.flush_passages = 10_000_000;
        for (i, v) in vectors.iter().enumerate() {
            let attrs = vec![("group".into(), Value::Int((i % 50) as i64)), ("bucket".into(), Value::Int((i % 10) as i64)), ("half".into(), Value::Int((i % 2) as i64)), ("rare".into(), Value::Int((i % 1000) as i64))];
            ix.upsert(&Doc { key: i as u64, version: 1, passages: vec![vec![format!("doc {i}")]], attrs, vector: Some(v.clone()) }).unwrap();
            if i % 18_000 == 17_999 {
                ix.checkpoint().unwrap();
            }
        }
        for i in (0..60_000).step_by(7) {
            ix.delete(i as u64).unwrap();
        }
        writeln!(out, "a: {} segments, {} in memory", ix.stats().segments, ix.stats().memory).unwrap();
        searches(&mut out, "a live", &ix, &queries);
        ix.checkpoint().unwrap();
        assert!(ix.start_merge(true));
        ix.wait_merge().unwrap();
        writeln!(out, "a merged: {} segments", ix.stats().segments).unwrap();
        searches(&mut out, "a merged", &ix, &queries);
        drop(ix);
        let ix = Index::open(&dir, 1).unwrap();
        searches(&mut out, "a reopened", &ix, &queries);
    }

    // One large segment with its calibrated budget (the benchmark's data, smaller).
    {
        let dims = 256;
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let vectors = clustered(100_200, dims, 1000, 0.8, &mut rng);
        let (docs, queries) = vectors.split_at(100_000);
        let dir = root.join("b");
        let mut ix = Index::open(&dir, 1).unwrap();
        ix.auto_merge = false;
        ix.flush_passages = 10_000_000;
        for (i, v) in docs.iter().enumerate() {
            ix.upsert(&Doc { key: i as u64, version: 1, passages: vec![], attrs: vec![], vector: Some(v.clone()) }).unwrap();
        }
        ix.checkpoint().unwrap();
        writeln!(out, "b: {} segments, {:?} generations, {} clusters", ix.stats().segments, ix.stats().vector_generations, ix.stats().vector_clusters).unwrap();
        for (q, v) in queries.iter().enumerate() {
            for (k, nprobe) in [(10, 0), (100, 0), (10, 1), (10, 4), (10, usize::MAX)] {
                line(&mut out, &format!("b k{k} nprobe{nprobe}"), q, &ix.nearest(v, k, None, nprobe));
            }
        }
    }

    // Through the namespace: vector alone, hybrid, filtered.
    {
        let dims = 64;
        let mut rng = Rng(77);
        let vectors = clustered(6_000, dims, 20, 0.5, &mut rng);
        let words = ["lease", "apartment", "invoice", "meeting", "holiday", "budget", "review", "contract"];
        let db = Database::open(root.join("c")).unwrap();
        let ns = db.namespace("docs", Schema::new().text("body", 1.0).int("year").vector(dims)).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            let body = format!("{} {} note {i}", words[i % words.len()], words[(i / 3) % words.len()]);
            ns.upsert(Document::new(i as u64).text("body", &body).int("year", 2000 + (i % 25) as i64).vector(v.clone())).unwrap();
            if i == 4_500 {
                ns.flush().unwrap();
            }
        }
        for (q, v) in vectors.iter().step_by(601).enumerate() {
            let runs = [
                ns.query().vector(v.clone()).limit(10).run().unwrap(),
                ns.query().text("apartment lease").vector(v.clone()).limit(10).run().unwrap(),
                ns.query().text("budget review").vector(v.clone()).filter(attr("year").gte(2020)).limit(20).run().unwrap(),
                ns.query().text("contract").vector(v.clone()).fusion(Fusion::Rrf { k: 60.0 }).limit(10).run().unwrap(),
            ];
            for (r, hits) in runs.iter().enumerate() {
                write!(out, "c run{r} q{q}:").unwrap();
                for h in hits {
                    write!(out, " {}:{:08x}:{:08x}", h.key, h.score.to_bits(), h.vector_score.map_or(0, f32::to_bits)).unwrap();
                }
                out.push('\n');
            }
        }
    }

    std::fs::write(&path, &out).unwrap();
    // KEEP=DIR: the files it wrote, moved there, to compare with another build's.
    if let Ok(keep) = std::env::var("KEEP") {
        std::fs::rename(&root, &keep).unwrap();
        println!("files kept in {keep}");
        return;
    }
    std::fs::remove_dir_all(&root).unwrap();
    println!("{} lines to {path}", out.lines().count());
}
