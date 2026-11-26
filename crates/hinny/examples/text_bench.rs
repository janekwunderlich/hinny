//! Full-text search on a BEIR dataset: latency, and with `--features bench` the same queries with
//! term-at-a-time scoring forced off and on, to see which wins where.
//!
//!   cargo run --release -p hinny --features bench --example text_bench -- DATA/fiqa/fiqa [english] [plans.jsonl]
//!
//! Reads corpus.jsonl (title + text), queries.jsonl and qrels/test.tsv (SPLIT=dev for another split).
//! K=200 asks for 200 results (a hybrid query's text side) instead of 11. With a third argument, writes
//! each query's lists (length, bound) and its time both ways, one JSON line each. Prints numbers only.

use std::time::Instant;

use hinny::{Analyzer, Database, Document, Language, Schema};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let english = args.get(2).is_some_and(|a| a == "english");
    let corpus: Vec<String> = std::fs::read_to_string(dir.join("corpus.jsonl"))
        .unwrap()
        .lines()
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            format!("{} {}", v["title"].as_str().unwrap_or(""), v["text"].as_str().unwrap_or("")).trim().to_string()
        })
        .collect();
    // SPLIT=dev (or train): that split's queries instead of the test ones.
    let split = std::env::var("SPLIT").unwrap_or_else(|_| "test".into());
    let test: std::collections::HashSet<String> = std::fs::read_to_string(dir.join(format!("qrels/{split}.tsv"))).unwrap().lines().skip(1).map(|l| l.split('\t').next().unwrap().to_string()).collect();
    let queries: Vec<String> = std::fs::read_to_string(dir.join("queries.jsonl"))
        .unwrap()
        .lines()
        .filter_map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            test.contains(v["_id"].as_str().unwrap()).then(|| v["text"].as_str().unwrap().to_string())
        })
        .collect();
    let tmp = std::env::temp_dir().join(format!("hinny-text-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let schema = if english { Schema::new().store_text(false).text_with("text", 1.0, Analyzer::language(Language::English)) } else { Schema::new().store_text(false).text("text", 1.0) };
    let ns = Database::open(&tmp).unwrap().namespace("t", schema).unwrap();
    ns.upsert_many(corpus.iter().enumerate().map(|(i, t)| Document::new(i as u64).text("text", t))).unwrap();
    ns.optimize().unwrap();
    println!("{} documents, {} queries, {} analyzer", corpus.len(), queries.len(), if english { "English" } else { "plain" });

    let k: usize = std::env::var("K").ok().and_then(|k| k.parse().ok()).unwrap_or(11);
    let time = |q: &str| {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            std::hint::black_box(ns.query().text(q).limit(k).with_text(false).snippet(0).run().unwrap());
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
        }
        best
    };
    let p50 = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        (v[v.len() / 2], v[v.len() * 99 / 100])
    };
    for q in queries.iter().take(20) {
        time(q);
    }
    let rule: Vec<f64> = queries.iter().map(|q| time(q)).collect();
    let (a, b) = p50(rule.clone());
    println!("as decided:          p50 {a:.3} ms  p99 {b:.3} ms");
    #[cfg(feature = "bench")]
    {
        use std::sync::atomic::Ordering;
        use hinny::search::{PLANS, TERM_AT_A_TIME};
        let mut runs = Vec::new();
        for mode in [1u8, 2, 3, 4] {
            TERM_AT_A_TIME.store(mode, Ordering::Relaxed);
            let t: Vec<f64> = queries.iter().map(|q| time(q)).collect();
            let (a, b) = p50(t.clone());
            println!("{}: p50 {a:.3} ms  p99 {b:.3} ms", ["", "never term at a time", "always term at a time", "the old rule", "strong lists only"][mode as usize]);
            runs.push(t);
        }
        TERM_AT_A_TIME.store(0, Ordering::Relaxed);
        let best: Vec<f64> = runs[0].iter().zip(&runs[1]).map(|(a, b)| a.min(*b)).collect();
        let (a, b) = p50(best);
        println!("the faster per query: p50 {a:.3} ms  p99 {b:.3} ms");
        if let Some(out) = args.get(3) {
            let mut lines = String::new();
            for (i, q) in queries.iter().enumerate() {
                PLANS.lock().unwrap().clear();
                time(q);
                let plan = PLANS.lock().unwrap().first().cloned().unwrap_or_default();
                lines.push_str(&serde_json::json!({"docs": plan.0, "lists": plan.1, "never": runs[0][i], "always": runs[1][i], "rule": rule[i]}).to_string());
                lines.push('\n');
            }
            std::fs::write(out, lines).unwrap();
        }
    }
    drop(ns);
    std::fs::remove_dir_all(tmp).unwrap();
}
