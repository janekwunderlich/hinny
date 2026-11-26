//! Vector search: recall against exact search, latency, and how much is read at full precision.
//!
//!   cargo run --release -p hinny --example vector_bench -- npy FILE.npy [QUERIES.npy]   (float32 rows)
//!   cargo run --release -p hinny --example vector_bench -- synthetic N DIMS [METRIC [SPREAD]]
//!
//! METRIC: cosine (the default), euclidean or dot. SPREAD: the synthetic vectors' lengths are
//! spread over that factor (100: the longest are a hundred times the shortest), which is what
//! makes dot product differ from cosine: each vector's by itself, or with SPREAD_BY=topic every
//! vector of a topic by the same factor (dot product's hard case: what it finds is then the
//! long topics, wherever the query points). Queries are spread like the documents, or all of
//! one length with QUERY_LENGTH=1.
//!
//! With one .npy file, 200 rows are held out as queries: queries like the documents. Real
//! questions are usually unlike the documents they look for (a question against an abstract), so
//! give them as a second file to measure what users get. Prints numbers only.

use std::collections::HashSet;
use std::time::Instant;

use hinny::Metric;
use hinny::index::{Doc, Index};

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

/// Clustered, like embeddings: 1,000 topics, each vector a topic centre plus noise. With the
/// topic of each.
fn synthetic(n: usize, dims: usize) -> (Vec<f32>, Vec<usize>) {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let topics: Vec<Vec<f32>> = (0..1000).map(|_| (0..dims).map(|_| rng.normal()).collect()).collect();
    let mut out = Vec::with_capacity(n * dims);
    let mut of = Vec::with_capacity(n);
    for _ in 0..n {
        let topic = (rng.next() % 1000) as usize;
        of.push(topic);
        out.extend(topics[topic].iter().map(|c| c + 0.8 * rng.normal()));
    }
    (out, of)
}

/// Each row made 1/√spread to √spread times as long (evenly on a logarithmic scale): each by
/// itself, or (SPREAD_BY=topic) every row of a topic by the same factor.
fn spread_lengths(data: &mut [f32], dims: usize, spread: f32, topics: &[usize]) {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let by_topic = std::env::var("SPREAD_BY").is_ok_and(|by| by == "topic");
    let of_topic: Vec<f32> = (0..1000).map(|_| (rng.next() % 1_000_000) as f32 / 1_000_000.0).collect();
    for (row, &topic) in data.chunks_exact_mut(dims).zip(topics) {
        let place = if by_topic { of_topic[topic] } else { (rng.next() % 1_000_000) as f32 / 1_000_000.0 };
        let factor = spread.powf(place - 0.5);
        row.iter_mut().for_each(|x| *x *= factor);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let held_out = args.get(1).map(String::as_str) == Some("npy") && args.get(3).is_none();
    let mut topics = Vec::new();
    let (mut data, dims) = match args.get(1).map(String::as_str) {
        Some("npy") => read_npy(&args[2]),
        Some("synthetic") => {
            let (n, d) = (args[2].parse().unwrap(), args[3].parse().unwrap());
            let (data, of) = synthetic(n, d);
            topics = of;
            (data, d)
        }
        _ => panic!("usage: vector_bench npy FILE | synthetic N DIMS [METRIC [SPREAD]]"),
    };
    let synthetic_run = args.get(1).map(String::as_str) == Some("synthetic");
    let metric = args.get(4).filter(|_| synthetic_run).map_or(Metric::Cosine, |m| Metric::from_name(m).expect("a metric: cosine, euclidean or dot"));
    if let Some(spread) = args.get(5).filter(|_| synthetic_run) {
        spread_lengths(&mut data, dims, spread.parse().expect("a factor"), &topics);
    }
    let mut rows = data.len() / dims;
    let mut queries = if held_out || args.get(1).map(String::as_str) == Some("synthetic") { 200.min(rows / 10) } else { 0 };
    let n = rows - queries;
    if let Some(qfile) = args.get(3).filter(|_| args[1] == "npy") {
        let (q, qd) = read_npy(qfile);
        assert_eq!(qd, dims, "queries of another dimension");
        data.extend_from_slice(&q);
        queries = q.len() / dims;
        rows = n + queries;
    }
    let dir = std::env::temp_dir().join(format!("hinny-vector-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.set_metric(metric).unwrap();
    // LIVE=1: segments as writing leaves them (the defaults: flushes as memory fills, merges in
    // the background, no final merge into one), which is what a query meets while data arrives.
    let live = std::env::var("LIVE").is_ok();
    if !live {
        ix.flush_passages = 200_000;
        ix.auto_merge = false;
    }
    let t = Instant::now();
    for i in 0..n {
        ix.upsert(&Doc { key: i as u64, version: 1, passages: vec![], attrs: vec![], vector: Some(data[i * dims..(i + 1) * dims].to_vec()) }).unwrap();
    }
    let logged = t.elapsed().as_secs_f64();
    ix.checkpoint().unwrap();
    if live {
        ix.settle().unwrap();
    } else if ix.stats().segments > 1 {
        ix.start_merge(true);
        ix.wait_merge().unwrap();
    }
    let built = t.elapsed().as_secs_f64();
    let size: u64 = std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().ends_with(".swt")).map(|e| e.metadata().unwrap().len()).sum();
    println!("{n} vectors × {dims} dimensions, {queries} queries; written in {logged:.1} s, indexed in {:.1} s ({} segment(s)); segment {:.0} MB ({:.0} MB of that full-precision vectors, in f16)", built - logged, ix.stats().segments, size as f64 / 1e6, (n * dims * 2) as f64 / 1e6);

    // QUERY_LENGTH=L: every query of that length, whatever the documents' lengths.
    if let Some(length) = std::env::var("QUERY_LENGTH").ok().and_then(|l| l.parse::<f32>().ok()) {
        for q in data[n * dims..].chunks_exact_mut(dims) {
            let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            q.iter_mut().for_each(|x| *x *= length / norm);
        }
    }
    let qs: Vec<&[f32]> = (n..rows).map(|i| &data[i * dims..(i + 1) * dims]).collect();
    // Exact answers, and how long scoring everything takes.
    let t = Instant::now();
    let exact: Vec<HashSet<u64>> = qs.iter().map(|q| ix.nearest_exact(q, 10, None).iter().map(|f| f.key).collect()).collect();
    let exact_ms = t.elapsed().as_secs_f64() * 1000.0 / qs.len() as f64;
    println!("exact (every vector at full precision): {exact_ms:.2} ms per query");
    // PROFILE=SECONDS: query with every cluster probed for that long first, for a profiler to watch.
    if let Some(secs) = std::env::var("PROFILE").ok().and_then(|s| s.parse::<f64>().ok()) {
        eprintln!("profiling {secs} s, pid {}", std::process::id());
        let t = Instant::now();
        while t.elapsed().as_secs_f64() < secs {
            for q in &qs {
                std::hint::black_box(ix.nearest_with_stats(q, 10, None, usize::MAX));
            }
        }
    }
    for nprobe in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, usize::MAX] {
        let mut times = Vec::new();
        let (mut recall, mut estimated, mut reranked) = (0.0, 0usize, 0usize);
        for (q, want) in qs.iter().zip(&exact) {
            let t = Instant::now();
            let (got, e, r) = ix.nearest_with_stats(q, 10, None, nprobe);
            times.push(t.elapsed().as_secs_f64() * 1000.0);
            recall += got.iter().filter(|f| want.contains(&f.key)).count() as f64 / want.len().max(1) as f64;
            estimated += e;
            reranked += r;
        }
        times.sort_by(f64::total_cmp);
        let q = qs.len() as f64;
        println!(
            "nprobe {:>4}: recall@10 {:.3}  p50 {:.3} ms  p99 {:.3} ms  codes scanned {:.1}%  read at full precision {:.2}% of the collection",
            match nprobe { 0 => "auto".to_string(), usize::MAX => "all".to_string(), n => n.to_string() },
            recall / q,
            times[times.len() / 2],
            times[(times.len() * 99 / 100).min(times.len() - 1)],
            estimated as f64 / q / n as f64 * 100.0,
            reranked as f64 / q / n as f64 * 100.0
        );
    }
    // Searches at once, each thread with the same view: how many a second, and whether one more
    // thread is one more thread's worth.
    let view = ix.view();
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut threads = vec![1, 2, 4, 8, 16, 32];
    threads.retain(|&t| t <= cores);
    if !threads.contains(&cores) {
        threads.push(cores);
    }
    for t in threads {
        let rounds = 20;
        let start = Instant::now();
        std::thread::scope(|scope| {
            for i in 0..t {
                let (view, qs) = (&view, &qs);
                scope.spawn(move || {
                    for r in 0..rounds {
                        for j in 0..qs.len() {
                            std::hint::black_box(view.nearest(qs[(i * 7 + r + j) % qs.len()], 10, None, 0));
                        }
                    }
                });
            }
        });
        let secs = start.elapsed().as_secs_f64();
        let done = (t * rounds * qs.len()) as f64;
        println!("{t:>2} thread(s): {:>7.0} searches a second ({:.3} ms each)", done / secs, secs * 1000.0 * t as f64 / done);
    }
    drop(view);
    drop(ix);
    std::fs::remove_dir_all(dir).unwrap();
}
