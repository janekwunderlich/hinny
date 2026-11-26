//! Attribute-only queries (no text, no vector): time and peak scratch memory per query, by filter
//! selectivity, order and limit, on generated documents (never user data).
//!
//!     cargo run --release -p hinny --offline --example query_bench -- [--docs N] [--dates random|sorted|skewed] [--seed S]
//!
//! Prints one JSON line per case: p50/p95 milliseconds over 30 runs and the most memory one query
//! allocated beyond what it returned with. With `--vectors DIMS`, documents get random vectors
//! instead, and it times the first vector query with a 0.01% filter after reopening the index
//! (the file cache warm, the process cold), then the second.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use hinny::attrs::{Filter, Value};
use hinny::index::{Doc, Index, Op};

/// Counts live bytes and their high-water mark.
struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_string());
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("query_bench [--docs N (1000000)] [--dates random|sorted|skewed (random)] [--seed S (1)] [--vectors DIMS: instead, the first filtered vector query after reopening]");
        return;
    }
    let n: u64 = arg("--docs", "1000000").parse().expect("--docs");
    let dims: usize = arg("--vectors", "0").parse().expect("--vectors");
    let dates = arg("--dates", "random");
    let mut s: u64 = arg("--seed", "1").parse::<u64>().expect("--seed") | 1;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let dir = std::env::temp_dir().join(format!("hinny-query-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut ix = Index::open(&dir, 1).unwrap();
    let t = Instant::now();
    for start in (0..n).step_by(10_000) {
        let docs: Vec<Doc> = (start..(start + 10_000).min(n))
            .map(|key| {
                let date = match dates.as_str() {
                    "sorted" => key as i64,
                    // Most documents recent, a long tail of old ones.
                    "skewed" => n as i64 - ((rnd() % 1000) as f64 / 1000.0).powi(8).mul_add(n as f64, 0.0) as i64,
                    _ => (rnd() % n) as i64,
                };
                // "group" g{0..99}: 1% each; "rare": 0.01%.
                let attrs = vec![("date".to_string(), Value::Int(date)), ("group".to_string(), Value::Str(format!("g{}", rnd() % 100))), ("rare".to_string(), Value::Int((rnd() % 10_000 == 0) as i64))];
                let vector = (dims > 0).then(|| (0..dims).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect());
                Doc { key, version: 1, passages: vec![vec!["x".to_string()]], attrs, vector }
            })
            .collect();
        let ops: Vec<Op> = docs.iter().map(Op::Put).collect();
        ix.write(&ops).unwrap();
    }
    ix.checkpoint().unwrap();
    while ix.wait_merge().unwrap() {}
    eprintln!("{n} documents in {:.1} s, {} segments", t.elapsed().as_secs_f64(), ix.stats().segments);
    if dims > 0 {
        drop(ix);
        let ix = Index::open(&dir, 1).unwrap();
        let rare = Filter::Eq("rare".into(), Value::Int(1));
        let q: Vec<f32> = (0..dims).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect();
        let mut times = Vec::new();
        for _ in 0..2 {
            let before = LIVE.load(Ordering::Relaxed);
            PEAK.store(before, Ordering::Relaxed);
            let t = Instant::now();
            let found = ix.nearest(&q, 10, Some(&rare), 0);
            times.push((t.elapsed().as_secs_f64() * 1000.0, PEAK.load(Ordering::Relaxed) - before, LIVE.load(Ordering::Relaxed) as i64 - before as i64));
            std::hint::black_box(found);
        }
        println!("{{\"docs\":{n},\"dims\":{dims},\"first_ms\":{:.2},\"first_peak_bytes\":{},\"retained_bytes\":{},\"second_ms\":{:.3}}}", times[0].0, times[0].1, times[0].2, times[1].0);
        drop(ix);
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    let filters: [(&str, Option<Filter>); 3] = [("all", None), ("1%", Some(Filter::Eq("group".into(), Value::Str("g7".into())))), ("0.01%", Some(Filter::Eq("rare".into(), Value::Int(1))))];
    let view = ix.view();
    for (selectivity, f) in &filters {
        for order in [None, Some(("date", true))] {
            for limit in [10, 1000] {
                let mut ms = Vec::new();
                let mut scratch = 0;
                for _ in 0..30 {
                    let before = LIVE.load(Ordering::Relaxed);
                    PEAK.store(before, Ordering::Relaxed);
                    let t = Instant::now();
                    let found = view.query(f.as_ref(), order, limit);
                    ms.push(t.elapsed().as_secs_f64() * 1000.0);
                    let kept = std::mem::size_of_val(found.as_slice());
                    scratch = scratch.max(PEAK.load(Ordering::Relaxed).saturating_sub(before + kept));
                    std::hint::black_box(found);
                }
                ms.sort_by(f64::total_cmp);
                println!(
                    "{{\"docs\":{n},\"dates\":\"{dates}\",\"filter\":\"{selectivity}\",\"order\":\"{}\",\"limit\":{limit},\"p50_ms\":{:.3},\"p95_ms\":{:.3},\"scratch_bytes\":{scratch}}}",
                    order.map_or("key", |_| "date desc"),
                    ms[ms.len() / 2],
                    ms[ms.len() * 95 / 100]
                );
            }
        }
    }
    drop((view, ix));
    let _ = std::fs::remove_dir_all(&dir);
}
