//! Approximate vector search against exact search on real queries, by cell: k, filter (none;
//! independent of the query; correlated with it, positively or negatively) and selectivity, in an
//! index of several segments, unflushed vectors and deletions. For calibrating on one set of
//! queries and checking, once, on another.
//!
//!   cargo run --release -p hinny --offline --example ann_calibration -- CORPUS.npy QUERIES.npy \
//!       [--name NAME] [--queries N] [--segments 4] [--memory N/20, at most 2000] [--deleted 0.05] [--nprobe 0]
//!
//! Recall is against two references: exact search over the original f32 vectors, and exact search
//! over what the index keeps (f16, normalized), which the engine's `nearest_exact` gives. Misses are
//! put down to probing (a cluster not scanned), pruning (scanned, near enough, skipped on its
//! estimate) or precision (f16 ranks it below the k-th). Prints one JSON line per cell: mean
//! recall, its 5th percentile, the share of queries below 0.9, misses and work per query, and
//! p50/p99 milliseconds.

use std::collections::HashSet;
use std::time::Instant;

use hinny::attrs::{Filter, Value};
use hinny::index::{Doc, Index, Op};
use rayon::prelude::*;

fn read_npy(path: &str) -> (Vec<f32>, usize) {
    let bytes = std::fs::read(path).expect("read .npy");
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not a .npy file");
    let (header_len, start) = if bytes[6] == 1 { (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10) } else { (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12) };
    let header = std::str::from_utf8(&bytes[start..start + header_len]).unwrap();
    assert!(header.contains("'<f4'") && header.contains("'fortran_order': False"), "float32, C order only: {header}");
    let shape = &header[header.find("'shape': (").unwrap() + 10..];
    let dims: usize = shape.split(',').nth(1).unwrap().trim().trim_end_matches(')').parse().unwrap();
    let data = bytes[start + header_len..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    (data, dims)
}

fn normalized(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

const KS: [usize; 3] = [10, 100, 1000];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.iter().any(|a| a == "--help") {
        println!("ann_calibration CORPUS.npy QUERIES.npy [--name NAME] [--queries N] [--segments 4] [--memory N/20, at most 2000] [--deleted 0.05] [--nprobe 0]");
        return;
    }
    let arg = |name: &str, default: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_string());
    let name = arg("--name", "data");
    let max_queries: usize = arg("--queries", "500").parse().unwrap();
    let segments: usize = arg("--segments", "4").parse().unwrap();
    let memory: Option<usize> = args.iter().position(|a| a == "--memory").map(|i| args[i + 1].parse().unwrap());
    let deleted: f64 = arg("--deleted", "0.05").parse().unwrap();
    let nprobe: usize = arg("--nprobe", "0").parse().unwrap();
    // For calibrating the depth exponent (a bench build): `--exponent X` overrides it.
    #[cfg(feature = "bench")]
    if let Some(x) = args.iter().position(|a| a == "--exponent").map(|i| args[i + 1].parse::<f64>().unwrap()) {
        hinny::vector::DEPTH_EXPONENT_OVERRIDE.store(x.to_bits().max(1), std::sync::atomic::Ordering::Relaxed);
    }
    let (corpus, dims) = read_npy(&args[1]);
    let (queries, qdims) = read_npy(&args[2]);
    assert_eq!(dims, qdims, "queries and corpus differ in dimensions");
    let n = corpus.len() / dims;
    let docs: Vec<Vec<f32>> = (0..n).map(|i| normalized(&corpus[i * dims..(i + 1) * dims])).collect();
    let nq = (queries.len() / dims).min(max_queries);
    let qs: Vec<Vec<f32>> = (0..nq).map(|i| normalized(&queries[i * dims..(i + 1) * dims])).collect();

    // "rand": independent of everything (per mille). "side": the document's rank, per mille, along
    // a fixed random direction; a query's own rank there gives filters correlated with it.
    let direction: Vec<f32> = normalized(&(0..dims).map(|i| (mix(i as u64 + 1) % 2001) as f32 / 1000.0 - 1.0).collect::<Vec<_>>());
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let mut along: Vec<f32> = docs.iter().map(|d| dot(d, &direction)).collect();
    along.sort_by(f32::total_cmp);
    let rank = |x: f32| ((along.partition_point(|&a| a < x) * 1000) / n).min(999) as i64;
    let side: Vec<i64> = docs.iter().map(|d| rank(dot(d, &direction))).collect();
    let rand: Vec<i64> = (0..n).map(|i| (mix(i as u64) % 1000) as i64).collect();

    // Segments of equal size, the last `memory` vectors unflushed, a share deleted (chosen by hash).
    let dir = std::env::temp_dir().join(format!("hinny-ann-calibration-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.auto_merge = false;
    ix.flush_passages = 1 << 40;
    ix.flush_bytes = 1 << 40;
    // Unflushed: a 20th, at most 2,000 (as much as a memtable keeps while writing, at most).
    let memory = memory.unwrap_or((n / 20).min(2000));
    let flushed = n.saturating_sub(memory);
    let per = flushed.div_ceil(segments.max(1)).max(1);
    let t = Instant::now();
    let doc = |i: usize| Doc { key: i as u64, version: 1, passages: vec![vec![String::new()]], attrs: vec![("rand".into(), Value::Int(rand[i])), ("side".into(), Value::Int(side[i]))], vector: Some(docs[i].clone()) };
    let mut ranges: Vec<(usize, usize, bool)> = (0..flushed).step_by(per).map(|a| (a, (a + per).min(flushed), true)).collect();
    ranges.push((flushed, n, false));
    for (a, b, flush) in ranges {
        for start in (a..b).step_by(5000) {
            let batch: Vec<Doc> = (start..(start + 5000).min(b)).map(doc).collect();
            let ops: Vec<Op> = batch.iter().map(Op::Put).collect();
            ix.write(&ops).unwrap();
        }
        if flush {
            ix.checkpoint().unwrap();
        }
    }
    let gone: Vec<u64> = (0..n as u64).filter(|&i| (mix(i ^ 0x5eed) % 10_000) as f64 / 10_000.0 < deleted).collect();
    for chunk in gone.chunks(5000) {
        let ops: Vec<Op> = chunk.iter().map(|&k| Op::Delete(k)).collect();
        ix.write(&ops).unwrap();
    }
    let live: Vec<bool> = {
        let mut l = vec![true; n];
        for &k in &gone {
            l[k as usize] = false;
        }
        l
    };
    let s = ix.stats();
    eprintln!("{name}: {n} vectors of {dims}, {} segments + {} in memory, {} deleted, built in {:.1} s; {nq} queries", s.segments, s.memory, gone.len(), t.elapsed().as_secs_f64());
    assert_eq!((s.segments, s.memory), (segments, n - flushed), "the layout asked for");

    // Cells: (label, selectivity per mille, kind). Correlated filters are windows on "side":
    // around the query's own rank (positive) or around the opposite one (negative).
    let cells: Vec<(&str, i64)> = vec![("none", 1000), ("independent", 500), ("independent", 100), ("independent", 30), ("independent", 10), ("independent", 3), ("independent", 1), ("positive", 100), ("positive", 10), ("negative", 100), ("negative", 10)];
    let filter_for = |kind: &str, per_mille: i64, q: &[f32]| -> Option<Filter> {
        let window = |centre: i64| {
            let lo = (centre - per_mille / 2).clamp(0, 1000 - per_mille);
            Filter::Range("side".into(), Some(lo), Some(lo + per_mille - 1))
        };
        match kind {
            "none" => None,
            "independent" => Some(Filter::Range("rand".into(), None, Some(per_mille - 1))),
            "positive" => Some(window(rank(dot(q, &direction)))),
            _ => Some(window(999 - rank(dot(q, &direction)))),
        }
    };
    let passes = |kind: &str, per_mille: i64, q: &[f32], i: usize| -> bool {
        match filter_for(kind, per_mille, q) {
            None => true,
            Some(Filter::Range(attr, lo, hi)) => {
                let v = if attr == "rand" { rand[i] } else { side[i] };
                lo.is_none_or(|lo| v >= lo) && hi.is_none_or(|hi| v <= hi)
            }
            _ => unreachable!(),
        }
    };

    // f32 truth: every live vector's distance, ranked once per query; each cell takes its passing
    // prefix.
    let truth32: Vec<Vec<Vec<u64>>> = qs
        .par_iter()
        .map(|q| {
            let mut ranked: Vec<(f32, u32)> = (0..n).filter(|&i| live[i]).map(|i| (-dot(q, &docs[i]), i as u32)).collect();
            ranked.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            cells.iter().map(|(kind, p)| ranked.iter().filter(|(_, i)| passes(kind, *p, q, *i as usize)).take(1000).map(|(_, i)| *i as u64).collect()).collect()
        })
        .collect();

    for (c, (kind, per_mille)) in cells.iter().enumerate() {
        for k in KS {
            let mut r32 = Vec::with_capacity(nq);
            let mut r16 = Vec::with_capacity(nq);
            let (mut probe, mut prune, mut precision, mut estimated, mut reranked) = (0usize, 0usize, 0usize, 0usize, 0usize);
            let mut ms = Vec::with_capacity(nq);
            for (qi, q) in qs.iter().enumerate() {
                let f = filter_for(kind, *per_mille, q);
                let truth16: Vec<u64> = ix.nearest_exact(q, k, f.as_ref()).iter().map(|h| h.key).collect();
                let (found, m) = ix.nearest_misses(q, k, f.as_ref(), nprobe, &truth16);
                let got: HashSet<u64> = found.iter().map(|h| h.key).collect();
                let t32: Vec<u64> = truth32[qi][c].iter().take(k).copied().collect();
                let recall = |t: &[u64]| if t.is_empty() { 1.0 } else { t.iter().filter(|x| got.contains(x)).count() as f64 / t.len() as f64 };
                r32.push(recall(&t32));
                r16.push(recall(&truth16));
                (probe, prune, precision, estimated, reranked) = (probe + m.probe, prune + m.prune, precision + m.precision, estimated + m.estimated, reranked + m.reranked);
                let t = Instant::now();
                std::hint::black_box(ix.nearest(q, k, f.as_ref(), nprobe));
                ms.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let stats = |r: &mut Vec<f64>| {
                r.sort_by(f64::total_cmp);
                (r.iter().sum::<f64>() / r.len() as f64, r[r.len() / 20], r.iter().filter(|&&x| x < 0.9).count() as f64 / r.len() as f64)
            };
            let (m32, p5_32, bad32) = stats(&mut r32);
            let (m16, p5_16, bad16) = stats(&mut r16);
            ms.sort_by(f64::total_cmp);
            let per = |x: usize| x as f64 / nq as f64;
            println!(
                "{{\"dataset\":\"{name}\",\"filter\":\"{kind}\",\"per_mille\":{per_mille},\"k\":{k},\"nprobe\":{nprobe},\"queries\":{nq},\"recall_f32\":{m32:.4},\"p5_f32\":{p5_32:.3},\"below_0.9_f32\":{bad32:.3},\"recall_f16\":{m16:.4},\"p5_f16\":{p5_16:.3},\"below_0.9_f16\":{bad16:.3},\"probe_misses\":{:.3},\"prune_misses\":{:.3},\"precision_misses\":{:.3},\"estimated\":{:.0},\"reranked\":{:.0},\"p50_ms\":{:.3},\"p99_ms\":{:.3}}}",
                per(probe),
                per(prune),
                per(precision),
                per(estimated),
                per(reranked),
                ms[ms.len() / 2],
                ms[ms.len() * 99 / 100]
            );
        }
    }
    drop(ix);
    let _ = std::fs::remove_dir_all(&dir);
}
