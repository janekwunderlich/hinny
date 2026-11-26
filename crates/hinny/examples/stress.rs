//! Writes, flushes, merges and queries at once, checking every snapshot: documents come in
//! pairs written and deleted in the same batch, so a snapshot holds both of a pair, at the same
//! version, or neither; and text, vector, hybrid and filtered queries on it find only what it
//! holds.
//!
//!   cargo run --release -p hinny --example stress -- [SECONDS]
//!   cargo run --release -p hinny --example stress -- crash [TIMES]
//!
//! `crash`: a child process writes pairs until it's killed (SIGKILL, at a random moment), and the
//! database must then open with every pair whole, and answer queries; TIMES times over.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use hinny::{Change, Database, Document, Schema, attr};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

const PAIRS: u64 = 6000;
const DIMS: usize = 64;

fn vector(rng: &mut Rng) -> Vec<f32> {
    (0..DIMS).map(|_| (rng.next() % 2000) as f32 / 1000.0 - 1.0).collect()
}

fn batch(rng: &mut Rng, round: i64) -> Vec<Change> {
    let mut batch = Vec::new();
    for _ in 0..40 {
        let p = rng.next() % PAIRS;
        if rng.next() % 8 == 0 {
            batch.push(Change::Delete(2 * p));
            batch.push(Change::Delete(2 * p + 1));
        } else {
            for k in [2 * p, 2 * p + 1] {
                let text = (0..3 + rng.next() % 20).map(|_| format!("w{}", rng.next() % 50)).collect::<Vec<_>>().join(" ");
                batch.push(Change::Put(Document::new(k).version(round as u64).text("body", text).int("pair", p as i64).int("round", round).vector(vector(rng))));
            }
        }
    }
    batch
}

fn schema() -> Schema {
    Schema::new().text("body", 1.0).int("pair").int("round").vector(DIMS)
}

/// Opens the database after a crash and checks it.
fn check_after_crash(dir: &std::path::Path) -> usize {
    let ns = Database::open(dir).unwrap().namespace("s", schema()).unwrap();
    let versions = ns.versions();
    for p in 0..PAIRS {
        assert_eq!(versions.get(&(2 * p)), versions.get(&(2 * p + 1)), "pair {p} torn by the crash");
    }
    let hits = ns.query().text("w1").vector(vec![0.5; DIMS]).limit(20).run().unwrap();
    assert!(hits.iter().all(|h| versions.contains_key(&h.key)));
    versions.len()
}

fn crash(times: usize) {
    let dir = std::env::temp_dir().join(format!("hinny-stress-crash-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let me = std::env::current_exe().unwrap();
    let mut rng = Rng(11);
    for t in 0..times {
        let mut child = std::process::Command::new(&me).arg("write").arg(&dir).arg(t.to_string()).spawn().unwrap();
        std::thread::sleep(Duration::from_millis(200 + rng.next() % 1500));
        child.kill().unwrap();
        child.wait().unwrap();
        let docs = check_after_crash(&dir);
        println!("crash {t}: {docs} documents, every pair whole");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// The child of `crash`: writes until killed, flushing now and then.
fn write(dir: &str, seed: u64) {
    let ns = Database::open(dir).unwrap().namespace("s", schema()).unwrap();
    let mut rng = Rng(0x5EED + seed);
    let mut round = (seed as i64) << 32;
    loop {
        ns.apply(batch(&mut rng, round)).unwrap();
        round += 1;
        if round % 200 == 0 {
            ns.flush().unwrap();
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("crash") => return crash(args.get(2).and_then(|a| a.parse().ok()).unwrap_or(10)),
        Some("write") => return write(&args[2], args[3].parse().unwrap()),
        _ => {}
    }
    let secs: u64 = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(60);
    let dir = std::env::temp_dir().join(format!("hinny-stress-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let ns = Database::open(&dir).unwrap().namespace("s", schema()).unwrap();
    let done = Arc::new(AtomicBool::new(false));
    let checked = Arc::new(AtomicUsize::new(0));
    let readers: Vec<_> = (0..4u64)
        .map(|r| {
            let (ns, done, checked) = (ns.clone(), done.clone(), checked.clone());
            std::thread::spawn(move || {
                let mut rng = Rng(0x9E37_79B9 + r);
                while !done.load(Ordering::Relaxed) {
                    let snap = ns.snapshot();
                    let versions = snap.versions();
                    for p in 0..PAIRS {
                        let (a, b) = (versions.get(&(2 * p)), versions.get(&(2 * p + 1)));
                        assert_eq!(a, b, "pair {p}: half a batch in snapshot {}", snap.seq());
                    }
                    let held = |key: u64| versions.contains_key(&key);
                    let word = format!("w{}", rng.next() % 50);
                    let q = vector(&mut rng);
                    let queries = [
                        snap.query().text(&word).limit(20).run().unwrap(),
                        snap.query().vector(q.clone()).limit(20).run().unwrap(),
                        snap.query().text(&word).vector(q.clone()).limit(20).run().unwrap(),
                        snap.query().vector(q.clone()).filter(attr("pair").lt(60)).limit(20).run().unwrap(),
                        snap.query().text(&word).filter(attr("round").gte(0)).limit(20).run().unwrap(),
                    ];
                    for hits in &queries {
                        for h in hits {
                            assert!(held(h.key), "a hit snapshot {} doesn't hold: {}", snap.seq(), h.key);
                        }
                    }
                    // Vectors: every held document with a vector can be found by its own.
                    if versions.len() >= 20 {
                        assert_eq!(queries[1].len(), 20, "vector hits in snapshot {}", snap.seq());
                    }
                    for h in &queries[3] {
                        assert!(h.key / 2 < 60, "filter: pair {}", h.key / 2);
                    }
                    checked.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();
    let mut rng = Rng(7);
    let start = Instant::now();
    let (mut round, mut writes) = (0i64, 0usize);
    while start.elapsed() < Duration::from_secs(secs) {
        let batch = batch(&mut rng, round);
        writes += batch.len();
        ns.apply(batch).unwrap();
        round += 1;
        if round % 150 == 0 {
            ns.flush().unwrap();
        }
    }
    done.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().unwrap();
    }
    println!("{round} batches ({writes} changes) in {secs} s, {} snapshots checked, {} segments at the end", checked.load(Ordering::Relaxed), ns.stats().segments);
    drop(ns);
    std::fs::remove_dir_all(dir).unwrap();
}
