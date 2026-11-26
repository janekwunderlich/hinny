//! Writes while querying: one thread writes documents one at a time, another runs hybrid queries
//! without pause. Prints latency percentiles for both.
//!
//!   cargo run --release -p hinny --example mixed -- [N] [DIMS]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use hinny::{Database, Document, Schema};

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

fn report(name: &str, mut ms: Vec<f64>) {
    ms.sort_by(f64::total_cmp);
    let at = |p: f64| ms[((ms.len() as f64 * p) as usize).min(ms.len() - 1)];
    println!("{name}: p50 {:.3} ms  p99 {:.3} ms  p99.9 {:.2} ms  max {:.1} ms  ({})", at(0.5), at(0.99), at(0.999), ms[ms.len() - 1], ms.len());
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(80_000);
    let dims: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(384);
    let dir = std::env::temp_dir().join(format!("hinny-mixed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let ns = Database::open(&dir)?.namespace("m", Schema::new().text("text", 1.0).vector(dims))?;
    let mut rng = Rng(7);
    let vecs: Vec<Vec<f32>> = (0..n).map(|_| (0..dims).map(|_| rng.normal()).collect()).collect();
    let vecs = Arc::new(vecs);
    let words = ["lease", "invoice", "flight", "refund", "contract", "meeting", "report", "travel"];
    let done = Arc::new(AtomicBool::new(false));
    let reader = {
        let (ns, vecs, done) = (ns.clone(), vecs.clone(), done.clone());
        std::thread::spawn(move || {
            let mut ms = Vec::new();
            let mut i = 0;
            while !done.load(Ordering::Relaxed) {
                let t = Instant::now();
                ns.query().text("lease refund").vector(vecs[i % 1000].clone()).limit(10).run().unwrap();
                ms.push(t.elapsed().as_secs_f64() * 1000.0);
                i += 1;
            }
            ms
        })
    };
    let start = Instant::now();
    let mut ms = Vec::with_capacity(n);
    for (i, v) in vecs.iter().enumerate() {
        let t = Instant::now();
        ns.upsert(Document::new(i as u64).text("text", format!("{} {} item {i}", words[i % 8], words[(i * 3) % 8])).vector(v.clone()))?;
        ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let total = start.elapsed().as_secs_f64();
    done.store(true, Ordering::Relaxed);
    let q = reader.join().unwrap();
    let mut slow: Vec<(f64, usize)> = ms.iter().copied().enumerate().filter(|(_, t)| *t > 5.0).map(|(i, t)| (t, i)).collect();
    slow.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("writes over 5 ms: {} — slowest at {:?}", slow.len(), slow.iter().take(6).map(|(t, i)| format!("#{i} {t:.0} ms")).collect::<Vec<_>>());
    report("write", ms);
    report("query", q);
    println!("{n} writes in {total:.1} s");
    drop(ns);
    std::fs::remove_dir_all(dir)
}
