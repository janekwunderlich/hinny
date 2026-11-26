//! Retrieval quality of text, vector and hybrid queries through the public API, on a benchmark
//! of questions about a mailbox (a hit is the email asked about or another in its thread).
//!
//!   cargo run --release -p hinny --example hybrid_eval -- DIR
//!
//! DIR has corpus.tsv (id, thread, date, text), vectors.npy and doc-ids.npy (document vectors and
//! their ids), query-vectors.npy, and queries.json ([{keywords, target}]). Prints metrics only.

use std::collections::HashMap;

use hinny::{Database, Document, Schema};

fn npy_f32(path: &str) -> (Vec<f32>, usize) {
    let b = std::fs::read(path).unwrap();
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let h = std::str::from_utf8(&b[10..10 + hl]).unwrap();
    let dims: usize = h[h.find("'shape': (").unwrap() + 10..].split(',').nth(1).unwrap().trim().trim_end_matches(')').parse().unwrap();
    (b[10 + hl..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(), dims)
}

fn npy_i64(path: &str) -> Vec<i64> {
    let b = std::fs::read(path).unwrap();
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    b[10 + hl..].chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn main() {
    let dir = std::env::args().nth(1).expect("DIR");
    let p = |f: &str| format!("{dir}/{f}");
    let (vectors, dims) = npy_f32(&p("vectors.npy"));
    let ids = npy_i64(&p("doc-ids.npy"));
    let vector_of: HashMap<i64, &[f32]> = ids.iter().enumerate().map(|(i, &id)| (id, &vectors[i * dims..(i + 1) * dims])).collect();
    let (qvecs, _) = npy_f32(&p("query-vectors.npy"));
    let queries: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(p("queries.json")).unwrap()).unwrap();

    let store = std::env::temp_dir().join(format!("hinny-hybrid-eval-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&store);
    let db = Database::open(&store).unwrap();
    let ns = db.namespace("mail", Schema::new().text("text", 1.0).int("thread").int("date").vector(dims)).unwrap();
    let mut thread: HashMap<u64, i64> = HashMap::new();
    let corpus = std::fs::read_to_string(p("corpus.tsv")).unwrap();
    let t = std::time::Instant::now();
    let mut docs = Vec::new();
    for line in corpus.lines() {
        let mut f = line.splitn(4, '\t');
        let (id, th, date, text) = (f.next().unwrap().parse::<u64>().unwrap(), f.next().unwrap().parse::<i64>().unwrap(), f.next().unwrap().parse::<i64>().unwrap(), f.next().unwrap_or(""));
        thread.insert(id, th);
        let mut d = Document::new(id).text("text", text).int("thread", th).int("date", date);
        if let Some(v) = vector_of.get(&(id as i64)) {
            d = d.vector(v.to_vec());
        }
        docs.push(d);
    }
    ns.upsert_many(docs).unwrap();
    ns.flush().unwrap();
    println!("{} emails indexed (text and vectors) in {:.1} s", thread.len(), t.elapsed().as_secs_f64());

    let hit = |key: u64, target: u64| key == target || (thread.get(&key).is_some_and(|&t| t != 0 && Some(&t) == thread.get(&target)));
    let mut ranks: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut times: HashMap<&str, Vec<f64>> = HashMap::new();
    for (i, q) in queries.iter().enumerate() {
        let keywords = q["keywords"].as_str().unwrap_or("");
        let target = q["target"].as_u64().unwrap();
        let qv = qvecs[i * dims..(i + 1) * dims].to_vec();
        for (name, query) in [("text", ns.query().text(keywords)), ("vector", ns.query().vector(qv.clone())), ("hybrid", ns.query().text(keywords).vector(qv.clone()))] {
            let t = std::time::Instant::now();
            let hits = query.limit(10).run().unwrap();
            times.entry(name).or_default().push(t.elapsed().as_secs_f64() * 1000.0);
            ranks.entry(name).or_default().push(hits.iter().position(|h| hit(h.key, target)).map_or(usize::MAX, |r| r + 1));
        }
    }
    println!("{} questions", queries.len());
    for name in ["text", "vector", "hybrid"] {
        let r = &ranks[name];
        let at = |k: usize| r.iter().filter(|&&x| x <= k).count() as f64 / r.len() as f64 * 100.0;
        let mrr = r.iter().map(|&x| if x <= 10 { 1.0 / x as f64 } else { 0.0 }).sum::<f64>() / r.len() as f64;
        let mut t = times[name].clone();
        t.sort_by(f64::total_cmp);
        println!("{name:<7} R@1 {:5.1}  R@3 {:5.1}  R@6 {:5.1}  R@10 {:5.1}  MRR@10 {mrr:.3}  p50 {:.2} ms  p99 {:.2} ms", at(1), at(3), at(6), at(10), t[t.len() / 2], t[t.len() * 99 / 100]);
    }
    drop(ns);
    std::fs::remove_dir_all(store).unwrap();
}
