//! hinny-names: build, search, benchmark and watch the Hinny name index.
//!
//!   hinny-names build [--root DIR] [--out FILE]   walk everything under DIR (default ~) into FILE
//!   hinny-names search QUERY [--paths] [--limit N]  top results (kinds and scores; paths with --paths)
//!   hinny-names bench [--freshness]                 latency per query and per keystroke; freshness
//!   hinny-names watch                               apply file system changes live, saving as it goes
//!
//! Output never includes file names unless asked (--paths): results are counts, kinds and timings.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use hinny_names::fast::{Path as SearchPath, Searcher};
use hinny_names::power;
use hinny_names::starts::WordStarts;
use hinny_names::store::{Builder, Index, Traits};
use hinny_names::watch::{self, FSEventsGetCurrentEventId};
use hinny_names::search;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let home = PathBuf::from(std::env::var_os("HOME").expect("HOME"));
    let out = value("--out").map(PathBuf::from).unwrap_or_else(Index::default_path);

    match args.first().map(String::as_str) {
        Some("build") => build(&value("--root").map(PathBuf::from).unwrap_or(home), &out),
        Some("search") => {
            let query = args.get(1).cloned().unwrap_or_default();
            let limit = value("--limit").and_then(|v| v.parse().ok()).unwrap_or(10);
            search_command(&out, &query, limit, flag("--paths"));
        }
        Some("bench") => bench(&out, flag("--freshness")),
        Some("cpu") => cpu(&out),
        Some("compare") => compare(&out),
        Some("count") => {
            // Entries under a folder (and how many rank as ~/Library), without printing names.
            let index = Index::open(&out).expect("open index");
            let dir = PathBuf::from(args.get(1).expect("count DIR"));
            let Some(root) = index.lookup(&dir) else { return println!("not indexed: {}", dir.display()) };
            let (mut n, mut lib, mut stack) = (0usize, 0usize, vec![root]);
            while let Some(e) = stack.pop() {
                for c in index.children(e) {
                    if index.is_deleted(c) { continue; }
                    n += 1;
                    if index.traits(c).has(Traits::LIBRARY) { lib += 1; }
                    stack.push(c);
                }
            }
            println!("{n} entries under it, {lib} ranked as ~/Library");
        }
        Some("starts") => build_starts(&out),
        Some("export") => {
            // For comparing with other engines (SQLite): one line per entry, in index order.
            let index = Index::open(&out).expect("open index");
            let dest = PathBuf::from(args.get(1).expect("export DEST"));
            let mut w = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&dest).expect("create"));
            use std::io::Write;
            for i in 0..index.count {
                let name = index.folded_name(i);
                if name.contains(&b'\t') || name.contains(&b'\n') {
                    continue;
                }
                w.write_all(name).unwrap();
                writeln!(w, "\t{}\t{}\t{}", index.depth(i), index.traits(i).0, index.modified(i)).unwrap();
            }
            println!("exported {} entries", index.count);
        }
        Some("compact") => {
            let index = Index::open(&out).expect("open index");
            let (cpu0, t0) = (cpu_seconds(), Instant::now());
            let tmp = out.with_extension("compact-test");
            let n = index.compact_to(&tmp).expect("compact");
            let size = std::fs::metadata(&tmp).map_or(0, |m| m.len());
            let _ = std::fs::remove_file(&tmp);
            println!("compacted {n} entries in {:.1} s ({:.1} s of CPU), wrote {:.0} MB", t0.elapsed().as_secs_f64(), cpu_seconds() - cpu0, size as f64 / 1e6);
        }
        Some("watch") => watch_command(&out),
        _ => eprintln!("usage: hinny-names build|search|bench|watch (see src/main.rs)"),
    }
}

fn now() -> u32 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as u32)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn peak_rss_mb() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    usage.ru_maxrss as f64 / 1e6 // bytes on macOS
}

fn build(root: &std::path::Path, out: &std::path::Path) {
    // The event ID from before the walk: whatever changes during it is replayed by `watch`.
    let since = unsafe { FSEventsGetCurrentEventId() };
    let battery = power::on_battery();
    let (cpu0, start) = (cpu_seconds(), Instant::now());
    // On battery the walk runs at background priority: efficiency cores, throttled disk reads.
    let builder = power::build_pool().install(|| Builder::walk(root));
    let walked = start.elapsed();
    builder.write(out, root, since).expect("write index");
    // A fresh build has no changes to replay.
    let _ = std::fs::remove_file(hinny_names::store::Journal::path_for(out));
    println!("{} mode: {:.0} s of CPU", if battery { "battery (background priority)" } else { "power" }, cpu_seconds() - cpu0);
    build_starts(out);
    let size = std::fs::metadata(out).map_or(0, |m| m.len());
    println!(
        "{} entries under {} in {:.1} s ({:.0} entries/s), written in {:.1} s: {:.0} MB; peak memory {:.0} MB",
        builder.len(),
        root.display(),
        walked.as_secs_f64(),
        builder.len() as f64 / walked.as_secs_f64(),
        (start.elapsed() - walked).as_secs_f64(),
        size as f64 / 1e6,
        peak_rss_mb()
    );
}

/// The word-start lists for one- and two-letter queries, from the index as written.
fn build_starts(out: &std::path::Path) {
    let index = Index::open(out).expect("open index");
    let t = Instant::now();
    let path = WordStarts::path_for(out);
    let n = power::build_pool().install(|| WordStarts::build(&index, &path, now())).expect("write word starts");
    let size = std::fs::metadata(&path).map_or(0, |m| m.len());
    println!("word starts: {n} list entries in {:.1} s, {:.1} MB", t.elapsed().as_secs_f64(), size as f64 / 1e6);
}

fn describe(index: &Index, hit: &search::Hit, paths: bool) -> String {
    let t = index.traits(hit.index);
    let kind = if t.has(Traits::APPLICATION) {
        "app"
    } else if t.has(Traits::DIRECTORY) {
        "folder"
    } else if t.has(Traits::DOCUMENT) {
        "document"
    } else if t.has(Traits::CODE_FILE) {
        "code file"
    } else {
        "file"
    };
    let mut where_ = Vec::new();
    for (bit, label) in [(Traits::LIBRARY, "Library"), (Traits::CODE, "code project"), (Traits::IN_PACKAGE, "in a bundle"), (Traits::HIDDEN, "hidden")] {
        if t.has(bit) {
            where_.push(label);
        }
    }
    let full = index.path(hit.index);
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    if full.starts_with(home.join("Library/Mobile Documents/com~apple~CloudDocs")) || full.starts_with(home.join("Library/CloudStorage")) {
        where_.push("cloud drive");
    }
    let place = if where_.is_empty() { String::new() } else { format!(" ({})", where_.join(", ")) };
    let path = if paths { format!("  {}", index.path(hit.index).display()) } else { String::new() };
    format!("{:7.1}  {}{} depth {}{}", hit.score, kind, place, index.depth(hit.index), path)
}

fn search_command(file: &std::path::Path, query: &str, limit: usize, paths: bool) {
    let start = Instant::now();
    let index = Index::open(file).expect("open index (run `hinny-names build` first)");
    let opened = start.elapsed();
    let mut searcher = Searcher::for_index(&index, file);
    let t = Instant::now();
    let hits = searcher.search(&index, query, limit, now());
    let first = t.elapsed();
    let mut fresh = Searcher::for_index(&index, file);
    let t = Instant::now();
    let _ = fresh.search(&index, query, limit, now());
    println!("\"{query}\": {} results; open {:.2} ms, first search {:.2} ms, again {:.2} ms", hits.len(), ms(opened), ms(first), ms(t.elapsed()));
    for hit in &hits {
        println!("  {}", describe(&index, hit, paths));
    }
}

// MARK: Benchmark

/// Words anyone might type (not taken from this Mac's files), plus multi-word and typo-free prefixes.
const QUERIES: &[&str] = &["lease", "invoice", "screenshot", "passport", "resume", "readme", "report", "notes", "photo", "tax", "budget", "contract", "2025", "config", "index", "sys set", "read me", "a", "s", "ab", "the"];
/// Typed a key at a time, as in the launcher.
const TYPED: &[&str] = &["screenshot", "invoice", "system settings", "boarding pass"];

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
}

fn time_query(index: &Index, q: &str, runs: usize) -> (f64, f64, usize) {
    let pool = power::search_pool();
    let _ = pool.install(|| search::search(index, q, 10, now())); // warm
    let mut times: Vec<f64> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            let hits = pool.install(|| search::search(index, q, 10, now()));
            std::hint::black_box(hits.len());
            ms(t.elapsed())
        })
        .collect();
    times.sort_by(f64::total_cmp);
    (percentile(&times, 0.5), percentile(&times, 0.99), pool.install(|| search::search(index, q, 10, now())).len())
}

fn bench(file: &std::path::Path, freshness: bool) {
    let t = Instant::now();
    let index = Index::open(file).expect("open index (run `hinny-names build` first)");
    let open_ms = ms(t.elapsed());
    let t = Instant::now();
    let _ = search::search(&index, "lease", 10, now());
    println!("index: {} entries, {:.0} MB of folded names; open {:.2} ms, first query {:.1} ms (file in the page cache)", index.count, index.folded_bytes as f64 / 1e6, open_ms, ms(t.elapsed()));
    println!("{} mode: {} search threads", if power::on_battery() { "battery" } else { "power" }, power::search_pool().current_num_threads());

    println!("\nper query (20 runs):");
    let mut all = Vec::new();
    for q in QUERIES {
        let (p50, p99, n) = time_query(&index, q, 20);
        all.push(p99);
        println!("  {:<14} p50 {:6.2} ms  p99 {:6.2} ms  ({} results)", format!("\"{q}\""), p50, p99, n);
    }
    println!("\ntyped a key at a time (each prefix, 20 runs):");
    for word in TYPED {
        let mut worst = 0f64;
        let mut sum = 0f64;
        let chars: Vec<char> = word.chars().collect();
        for n in 1..=chars.len() {
            let prefix: String = chars[..n].iter().collect();
            let (p50, p99, _) = time_query(&index, &prefix, 20);
            worst = worst.max(p99);
            sum += p50;
            all.push(p99);
        }
        println!("  {:<18} {} keystrokes, median {:.2} ms each, worst p99 {:.2} ms", format!("\"{word}\""), chars.len(), sum / chars.len() as f64, worst);
    }
    all.sort_by(f64::total_cmp);
    println!("\nall queries and keystrokes: p99 of p99s {:.2} ms; peak memory {:.0} MB", percentile(&all, 0.99), peak_rss_mb());

    if freshness {
        drop(index);
        bench_freshness(file);
    }
}

fn cpu_seconds() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let t = |tv: libc::timeval| tv.tv_sec as f64 + tv.tv_usec as f64 / 1e6;
    t(usage.ru_utime) + t(usage.ru_stime)
}

/// CPU time (all threads) and wall time per search, at the thread count rayon was given
/// (RAYON_NUM_THREADS): the energy a keystroke costs, against how fast it answers.
fn cpu(file: &std::path::Path) {
    let index = Index::open(file).expect("open index");
    for q in QUERIES {
        let _ = search::search(&index, q, 10, now()); // pages in, warm
    }
    let runs = 30;
    let (cpu0, t0) = (cpu_seconds(), Instant::now());
    for _ in 0..runs {
        for q in QUERIES {
            std::hint::black_box(search::search(&index, q, 10, now()));
        }
    }
    let n = (runs * QUERIES.len()) as f64;
    let cpu_ms = (cpu_seconds() - cpu0) * 1000.0 / n;
    let wall_ms = ms(t0.elapsed()) / n;
    println!("threads {:>2}: {:.2} ms per search, {:.1} ms of CPU time per search ({:.1} cores busy while searching)", rayon::current_num_threads(), wall_ms, cpu_ms, cpu_ms / wall_ms);
}

fn bench_freshness(file: &std::path::Path) {
    println!("\nfreshness (a file created and removed under ~/Library/Caches/dev.hinny.bench):");
    let index = Arc::new(RwLock::new(Index::open(file).expect("open index")));
    let (tx, rx) = std::sync::mpsc::channel::<bool>();
    let t = Instant::now();
    let _watcher = watch::watch_index(index.clone(), move |applied, _| {
        if applied.history_done {
            let _ = tx.send(true);
        }
    })
    .expect("start FSEvents");
    // Catching up on everything since the index was written.
    let _ = rx.recv_timeout(Duration::from_secs(120));
    {
        let ix = index.read().unwrap();
        println!("  caught up on changes since the build in {:.0} ms ({} entries added since)", ms(t.elapsed()), ix.added.len());
    }
    let dir = PathBuf::from(std::env::var_os("HOME").unwrap()).join("Library/Caches/dev.hinny.bench");
    std::fs::create_dir_all(&dir).unwrap();
    std::thread::sleep(Duration::from_millis(300));
    for round in 0..5 {
        let token = format!("zqxfresh{}{}", std::process::id(), round);
        let path = dir.join(format!("{token}.txt"));
        let t = Instant::now();
        std::fs::write(&path, b"x").unwrap();
        let found = wait_for(&index, &token, true);
        let added_ms = ms(t.elapsed());
        let t = Instant::now();
        std::fs::remove_file(&path).unwrap();
        let gone = wait_for(&index, &token, false);
        println!("  round {}: searchable after {:.0} ms{}, gone after {:.0} ms{}", round + 1, added_ms, if found { "" } else { " (timed out)" }, ms(t.elapsed()), if gone { "" } else { " (timed out)" });
    }
    let _ = std::fs::remove_dir(&dir);
}

fn wait_for(index: &RwLock<Index>, token: &str, present: bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        let found = {
            let ix = index.read().unwrap();
            !search::search(&ix, token, 1, now()).is_empty()
        };
        if found == present {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

fn watch_command(file: &std::path::Path) {
    let service = hinny_names::service::Service::open(file).expect("open index (run `hinny-names build` first)");
    println!("opened: {} entries, {} replayed from the journal", service.index.read().unwrap().count, service.index.read().unwrap().added.len());
    if !service.watch(|line| println!("{line}")) {
        eprintln!("couldn't start FSEvents");
        return;
    }
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}

// MARK: Old engine vs new

fn same_top(a: &[search::Hit], b: &[search::Hit]) -> bool {
    let mut x: Vec<usize> = a.iter().map(|h| h.index).collect();
    let mut y: Vec<usize> = b.iter().map(|h| h.index).collect();
    x.sort_unstable();
    y.sort_unstable();
    x == y
}

/// Same top 10 up to ties: the same scores (entries with equal scores at the cut-off can come
/// from either engine).
fn same_scores(a: &[search::Hit], b: &[search::Hit]) -> bool {
    let key = |h: &[search::Hit]| {
        let mut s: Vec<i64> = h.iter().map(|x| (x.score * 100.0).round() as i64).collect();
        s.sort_unstable();
        s
    };
    key(a) == key(b)
}

/// The original engine (rayon, a full scan per keystroke) against the new one (`fast::Searcher`):
/// independent queries, typing a key at a time, CPU per keystroke, and whether the top 10 agree.
fn compare(file: &std::path::Path) {
    let index = Index::open(file).expect("open index");
    let probe = Searcher::for_index(&index, file);
    println!("{} entries; {} mode; new engine: {} workers (GCD), word-start lists {}", index.count, if power::on_battery() { "battery" } else { "power" }, hinny_names::pool::workers(), if probe.has_word_starts() { "present" } else { "MISSING (run `hinny-names starts`)" });
    let old_pool = power::search_pool();
    // Warm both and page everything in.
    for q in QUERIES {
        let _ = old_pool.install(|| search::search(&index, q, 10, now()));
        let _ = Searcher::for_index(&index, file).search(&index, q, 10, now());
    }

    println!("\nindependent queries (no earlier keystroke to build on), 20 runs each:");
    println!("  {:<12} {:>17} {:>17}  {:<10} agree", "query", "old p50 / p99", "new p50 / p99", "new path");
    let mut agree = 0;
    let (mut old_q_cpu, mut new_q_cpu) = (0.0f64, 0.0f64);
    for q in QUERIES {
        let mut old_t = Vec::new();
        let mut new_t = Vec::new();
        let mut path = SearchPath::Scan;
        let (mut old_hits, mut new_hits) = (Vec::new(), Vec::new());
        for _ in 0..20 {
            let c0 = cpu_seconds();
            let t = Instant::now();
            old_hits = old_pool.install(|| search::search(&index, q, 10, now()));
            old_t.push(ms(t.elapsed()));
            old_q_cpu += cpu_seconds() - c0;
            let mut s = Searcher::for_index(&index, file);
            let c0 = cpu_seconds();
            let t = Instant::now();
            new_hits = s.search(&index, q, 10, now());
            new_t.push(ms(t.elapsed()));
            new_q_cpu += cpu_seconds() - c0;
            path = s.last;
        }
        old_t.sort_by(f64::total_cmp);
        new_t.sort_by(f64::total_cmp);
        let same = same_top(&old_hits, &new_hits);
        agree += same as usize;
        if !same && !same_scores(&old_hits, &new_hits) {
            let fmt = |h: &[search::Hit]| h.iter().map(|x| format!("{:.1}", x.score)).collect::<Vec<_>>().join(" ");
            println!("    scores differ: old [{}] new [{}]", fmt(&old_hits), fmt(&new_hits));
        }
        println!("  {:<12} {:7.3} / {:6.3} ms {:7.3} / {:6.3} ms  {:<10} {}", format!("\"{q}\""), percentile(&old_t, 0.5), percentile(&old_t, 0.99), percentile(&new_t, 0.5), percentile(&new_t, 0.99), path_name(path), if same { "yes" } else { "NO" });
    }
    println!("  top 10 identical for {agree} of {} queries", QUERIES.len());
    let n = (QUERIES.len() * 20) as f64;
    println!("  CPU per query: old {:.2} ms, new {:.2} ms", old_q_cpu * 1000.0 / n, new_q_cpu * 1000.0 / n);

    // Typing: the whole word key by key, 20 times, each time with a fresh searcher (only that
    // word's earlier keystrokes to build on); CPU measured over all keystrokes.
    println!("\ntyped a key at a time (20 times each):");
    let mut old_all = Vec::new();
    let mut new_all = Vec::new();
    let (mut keystrokes, mut agree_keys, mut checked) = (0usize, 0usize, 0usize);
    let (mut old_cpu, mut new_cpu) = (0.0f64, 0.0f64);
    let mut paths: std::collections::BTreeMap<&str, usize> = Default::default();
    for word in TYPED {
        let chars: Vec<char> = word.chars().collect();
        let prefixes: Vec<String> = (1..=chars.len()).map(|n| chars[..n].iter().collect()).collect();
        let (mut o, mut n) = (Vec::new(), Vec::new());
        for run in 0..20 {
            let c0 = cpu_seconds();
            for p in &prefixes {
                let t = Instant::now();
                let _ = old_pool.install(|| search::search(&index, p, 10, now()));
                o.push(ms(t.elapsed()));
            }
            old_cpu += cpu_seconds() - c0;
            let mut s = Searcher::for_index(&index, file);
            let c0 = cpu_seconds();
            for p in &prefixes {
                let t = Instant::now();
                let hits = s.search(&index, p, 10, now());
                n.push(ms(t.elapsed()));
                if run == 0 {
                    *paths.entry(path_name(s.last)).or_default() += 1;
                    let old = old_pool.install(|| search::search(&index, p, 10, now()));
                    checked += 1;
                    agree_keys += same_top(&old, &hits) as usize;
                    if !same_scores(&old, &hits) {
                        let fmt = |h: &[search::Hit]| h.iter().map(|x| format!("{:.1}", x.score)).collect::<Vec<_>>().join(" ");
                        println!("    \"{p}\" ({}): scores differ: old [{}] new [{}]", path_name(s.last), fmt(&old), fmt(&hits));
                    }
                }
            }
            new_cpu += cpu_seconds() - c0;
            keystrokes += prefixes.len();
        }
        let (mut os, mut ns) = (o.clone(), n.clone());
        os.sort_by(f64::total_cmp);
        ns.sort_by(f64::total_cmp);
        println!("  {:<18} old median {:.3} ms, p99 {:.3} ms   new median {:.3} ms, p99 {:.3} ms", format!("\"{word}\""), percentile(&os, 0.5), percentile(&os, 0.99), percentile(&ns, 0.5), percentile(&ns, 0.99));
        old_all.extend(o);
        new_all.extend(n);
    }
    old_all.sort_by(f64::total_cmp);
    new_all.sort_by(f64::total_cmp);
    println!("  all keystrokes: old median {:.3} ms, p99 {:.3} ms; new median {:.3} ms, p99 {:.3} ms", percentile(&old_all, 0.5), percentile(&old_all, 0.99), percentile(&new_all, 0.5), percentile(&new_all, 0.99));
    let per_key = |cpu: f64| cpu * 1000.0 / keystrokes as f64;
    println!("  CPU per keystroke: old {:.2} ms, new {:.3} ms ({:.0}× less)", per_key(old_cpu), per_key(new_cpu), old_cpu / new_cpu.max(1e-9));
    println!("  new engine's paths: {}", paths.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", "));
    println!("  top 10 identical for {agree_keys} of {checked} keystrokes");
}

fn path_name(p: SearchPath) -> &'static str {
    match p {
        SearchPath::Repeat => "repeat",
        SearchPath::WordStarts => "starts",
        SearchPath::Refined { .. } => "refined",
        SearchPath::Scan => "scan",
    }
}

