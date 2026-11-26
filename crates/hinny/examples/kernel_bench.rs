//! The vector scan's kernels on their own, in nanoseconds: scoring a block of 32 codes against a
//! query's lookup tables, the lower bounds of a block, a full-precision distance, a query's
//! tables for one cluster, and the distance to a centroid.
//!
//!   cargo run --release -p hinny --example kernel_bench -- [DIMS]

use std::hint::black_box;
use std::time::Instant;

use hinny::vector::{Blocked, Pre, QueryCode, dist2, dist2_f16, to_f16};

fn main() {
    let dims: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let p = dims.next_power_of_two().max(64);
    let words = p / 64;
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let n = 32 * 4096;
    let codes: Vec<u64> = (0..n * words).map(|_| next()).collect();
    let blocked = Blocked::new(&codes, &[0, n as u32], words);
    let pre: Vec<Pre> = (0..n).map(|_| Pre { inv_ip: 1.25, err: 0.75, norm: 0.9 + (next() % 100) as f32 / 1000.0, ones: (p / 2) as f32 }).collect();
    let unit = |x: u64| (x % 20_001) as f32 / 10_000.0 - 1.0;
    let q: Vec<f32> = (0..p).map(|_| unit(next())).collect();
    let c: Vec<f32> = (0..p).map(|_| 0.2 * unit(next())).collect();
    let mut qc = QueryCode::new(&q, &c);
    let blocks = n / 32;
    let rounds = 200;

    let mut weighted = [0u32; 32];
    // A processor that was idle takes a moment to reach its full clock rate.
    let t = Instant::now();
    while t.elapsed().as_millis() < 300 {
        qc.weighted_block(&blocked.bytes[..blocked.block], &mut weighted);
        black_box(&weighted);
    }
    let t = Instant::now();
    let mut sum = 0u64;
    for _ in 0..rounds {
        for b in 0..blocks {
            qc.weighted_block(&blocked.bytes[b * blocked.block..(b + 1) * blocked.block], &mut weighted);
            sum += weighted[7] as u64;
        }
    }
    println!("{dims} dimensions ({p} padded)");
    println!("score a block of 32 codes:        {:>7.1} ns", t.elapsed().as_nanos() as f64 / (rounds * blocks) as f64);
    black_box(sum);

    for (name, threshold) in [("none pass", 0.0f32), ("all pass", f32::INFINITY)] {
        let t = Instant::now();
        let mut passed = 0usize;
        for _ in 0..rounds {
            for b in 0..blocks {
                weighted[b % 32] = (b % 1000) as u32;
                qc.below(black_box(&weighted), &pre[b * 32..(b + 1) * 32], threshold, u32::MAX, |_, _, _| passed += 1);
            }
        }
        println!("bounds of a block, {name:<10}:    {:>7.1} ns", t.elapsed().as_nanos() as f64 / (rounds * blocks) as f64);
        black_box(passed);
    }

    let rows: Vec<f32> = (0..1024 * dims).map(|_| unit(next())).collect();
    let halves = to_f16(&rows);
    let t = Instant::now();
    let mut total = 0f32;
    for _ in 0..rounds * 20 {
        for r in 0..1024 {
            total += dist2_f16(&q[..dims], &halves[r * dims..(r + 1) * dims]);
        }
    }
    println!("distance to a half-precision row: {:>7.1} ns", t.elapsed().as_nanos() as f64 / (rounds * 20 * 1024) as f64);
    let cents: Vec<f32> = (0..1024 * p).map(|_| unit(next())).collect();
    let t = Instant::now();
    for _ in 0..rounds * 20 {
        for r in 0..1024 {
            total += dist2(&q, &cents[r * p..(r + 1) * p]);
        }
    }
    println!("distance to a centroid:           {:>7.1} ns", t.elapsed().as_nanos() as f64 / (rounds * 20 * 1024) as f64);
    let t = Instant::now();
    for r in 0..rounds * 500 {
        qc.set(&q, &cents[(r % 1024) * p..(r % 1024 + 1) * p]);
    }
    println!("a query's tables for a cluster:   {:>7.1} ns", t.elapsed().as_nanos() as f64 / (rounds * 500) as f64);
    black_box(total);
}
