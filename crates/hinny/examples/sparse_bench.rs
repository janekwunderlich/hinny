//! Learned sparse queries: latency, and with `--features bench` what pruning leaves to do.
//!
//!   cargo run --release -p hinny --features bench --example sparse_bench -- CORPUS.jsonl QUERIES.jsonl
//!
//! Each line of both files is a {term: weight} object (benchmarks/beir/embed_sparse.py). Prints
//! numbers only.

use std::time::Instant;

use hinny::{Database, Document, Schema};

fn read(path: &str) -> Vec<Vec<(String, f32)>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| {
            let v: serde_json::Map<String, serde_json::Value> = serde_json::from_str(l).unwrap();
            v.into_iter().map(|(t, w)| (t, w.as_f64().unwrap() as f32)).collect()
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // OLD_RULE=1: term at a time whenever the lists cover the segment, as before strong lists
    // were counted (bench feature), to compare.
    #[cfg(feature = "bench")]
    if std::env::var("OLD_RULE").is_ok() {
        hinny::search::TERM_AT_A_TIME.store(3, std::sync::atomic::Ordering::Relaxed);
    }
    let (docs, queries) = (read(&args[1]), read(&args[2]));
    let dir = std::env::temp_dir().join(format!("hinny-sparse-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let ns = Database::open(&dir).unwrap().namespace("s", Schema::new().store_text(false).sparse("terms")).unwrap();
    let t = Instant::now();
    ns.upsert_many(docs.iter().enumerate().map(|(i, terms)| Document::new(i as u64).sparse("terms", terms))).unwrap();
    ns.optimize().unwrap();
    let postings: usize = docs.iter().map(Vec::len).sum();
    println!("{} documents, {:.0} terms each, built in {:.1} s; {} queries, {:.1} terms each", docs.len(), postings as f64 / docs.len() as f64, t.elapsed().as_secs_f64(), queries.len(), queries.iter().map(Vec::len).sum::<usize>() as f64 / queries.len() as f64);
    for k in [10, 100] {
        let mut ms = Vec::new();
        #[cfg(feature = "bench")]
        let before = [&hinny::search::DECODED_BLOCKS, &hinny::search::CANDIDATES, &hinny::search::PROBES].map(|c| c.load(std::sync::atomic::Ordering::Relaxed));
        for q in &queries {
            let t = Instant::now();
            std::hint::black_box(ns.query().sparse("terms", q.iter().map(|(t, w)| (t.as_str(), *w))).limit(k).with_text(false).snippet(0).run().unwrap());
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        ms.sort_by(f64::total_cmp);
        print!("top {k}: p50 {:.3} ms  p99 {:.3} ms", ms[ms.len() / 2], ms[ms.len() * 99 / 100]);
        #[cfg(feature = "bench")]
        {
            let after = [&hinny::search::DECODED_BLOCKS, &hinny::search::CANDIDATES, &hinny::search::PROBES].map(|c| c.load(std::sync::atomic::Ordering::Relaxed));
            let per = |i: usize| (after[i] - before[i]) as f64 / queries.len() as f64;
            print!("  per query: {:.0} blocks decoded, {:.0} candidates, {:.0} probes", per(0), per(1), per(2));
        }
        println!();
    }
    drop(ns);
    std::fs::remove_dir_all(dir).unwrap();
}
