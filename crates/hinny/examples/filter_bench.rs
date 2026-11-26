//! How long conditions on attributes take: `filter_bench [DOCUMENTS]` (a million by default).
//!
//! Documents as a file search has them: a kind (a dozen strings, some common), an owner (a
//! thousand), a path (every one different), a size, a time that grows with the key, as it
//! does when files are indexed as they change, and labels (tags: a group of a thousand, one
//! three in ten have, one two in a hundred have, and up to two of fifty topics). One segment, so a time is the columns' own.
//! Prints, for each condition, how many documents pass it, how long finding them in the segment
//! takes (a bit per document), and how long a query for the ten first takes (which looks at
//! every match), alone and with a word in front; then counts, the ten first in an order, and
//! how long a fork of the namespace takes.

use std::time::Instant;

use hinny::{Database, Document, Filter, Schema, attr};

const KINDS: [&str; 12] = ["pdf", "txt", "jpg", "png", "docx", "xlsx", "md", "html", "mp4", "zip", "eml", "key"];

fn main() {
    let n: u64 = std::env::args().nth(1).map(|a| a.parse().expect("a number of documents")).unwrap_or(1_000_000);
    let dir = std::env::temp_dir().join(format!("hinny-filter-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let schema = || Schema::new().text("body", 1.0).string("kind").string("owner").string("path").int("size").int("modified").tags("labels");
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let t = Instant::now();
    {
        let db = Database::open(&dir).unwrap();
        let ns = db.namespace("files", schema()).unwrap();
        let mut batch = Vec::with_capacity(10_000);
        for k in 0..n {
            let r = next();
            // The first kinds are the common ones: a half, a quarter, an eighth, ...
            let kind = KINDS[((r as u16).leading_zeros() as usize).min(KINDS.len() - 1)];
            let owner = (r >> 16) % 1000;
            let name = if (r >> 40) % 500 == 0 { "report" } else { "file" };
            let s = next();
            let mut labels = vec![format!("g{}", s % 1000)];
            if (s >> 10) % 10 < 3 {
                labels.push("shared".to_string());
            }
            if (s >> 20) % 50 == 0 {
                labels.push("starred".to_string());
            }
            labels.extend((0..(s >> 30) % 3).map(|i| format!("topic{}", (s >> (34 + 8 * i)) % 50)));
            batch.push(
                Document::new(k)
                    .text("body", if k % 4 == 0 { "alpha" } else { "beta" })
                    .string("kind", kind)
                    .string("owner", format!("user{owner}"))
                    .string("path", format!("/home/user{owner}/{}/{name}{k}.{kind}", ["work", "private", "old"][(r >> 30) as usize % 3]))
                    .int("size", ((r >> 20) % 4_000_000_000) as i64)
                    .int("modified", 1_700_000_000 + k as i64 * 30 + ((r >> 8) % 600) as i64)
                    .tags("labels", labels),
            );
            if batch.len() == 10_000 {
                ns.upsert_many(batch.drain(..)).unwrap();
            }
        }
        ns.upsert_many(batch).unwrap();
        ns.optimize().unwrap();
    }
    let db = Database::open(&dir).unwrap();
    let ns = db.namespace("files", schema()).unwrap();
    let files = |ext: &str| -> u64 { std::fs::read_dir(dir.join("files")).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().ends_with(ext)).map(|e| e.metadata().unwrap().len()).sum() };
    println!("{n} documents in {} segment(s), written in {:.1} s; segments {:.1} MB", ns.stats().segments, t.elapsed().as_secs_f64(), files(".swt") as f64 / 1e6);

    let day = 86_400;
    let from = 1_700_000_000 + n as i64 * 30 / 2;
    let conditions: Vec<(&str, Filter)> = vec![
        ("kind = pdf", attr("kind").eq("pdf")),
        ("kind = eml", attr("kind").eq("eml")),
        ("kind one of four", attr("kind").one_of(["zip", "eml", "key", "mp4"])),
        ("owner = user417", attr("owner").eq("user417")),
        ("owner one of five", attr("owner").one_of(["user1", "user22", "user333", "user417", "user999"])),
        ("owner starts with user41", attr("owner").prefix("user41")),
        ("path starts with /home/user417/", attr("path").prefix("/home/user417/")),
        ("path contains report", attr("path").contains("report")),
        ("path = one file", attr("path").eq("/home/nobody/file.txt")),
        ("modified in one day", attr("modified").between(from, from + day)),
        ("modified in the last tenth", attr("modified").gte(1_700_000_000 + n as i64 * 27)),
        ("size above 3.96 GB", attr("size").gt(3_960_000_000)),
        ("size one of three", attr("size").one_of([5i64, 77, 1234])),
        ("kind = pdf and modified in one day", attr("kind").eq("pdf") & attr("modified").between(from, from + day)),
        ("owner = user417 and not kind = pdf", attr("owner").eq("user417") & attr("kind").ne("pdf")),
        ("kind = eml or size above 3.96 GB", attr("kind").eq("eml") | attr("size").gt(3_960_000_000)),
        ("labels has g417", attr("labels").has("g417")),
        ("labels has starred", attr("labels").has("starred")),
        ("labels has shared", attr("labels").has("shared")),
        ("labels has one of five groups", attr("labels").has_any(["g1", "g22", "g333", "g417", "g999"])),
        ("labels has shared and starred", attr("labels").has_all(["shared", "starred"])),
        ("labels has one starting with g41", attr("labels").prefix("g41")),
        ("labels has one containing opic4", attr("labels").contains("opic4")),
        ("labels has g417 and kind = pdf", attr("labels").has("g417") & attr("kind").eq("pdf")),
        ("no label", !attr("labels").exists()),
    ];
    let file = std::fs::read_dir(dir.join("files")).unwrap().flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|x| x == "swt")).unwrap();
    let segment = hinny::segment::Segment::open(&file).unwrap();
    // The best of five rounds: the machine does other things too.
    let best = |run: &dyn Fn()| {
        run();
        (0..5)
            .map(|_| {
                let t = Instant::now();
                for _ in 0..20 {
                    run();
                }
                t.elapsed().as_secs_f64() * 1000.0 / 20.0
            })
            .fold(f64::MAX, f64::min)
    };
    println!("{:<40} {:>9} {:>12} {:>12} {:>16}", "condition", "pass", "found, ms", "alone, ms", "with a word, ms");
    for (name, f) in &conditions {
        let pass = ns.query().filter(f.clone()).limit(n as usize).run().unwrap().len();
        let bound = f.bind(&segment.attr_columns());
        let found = best(&|| {
            std::hint::black_box(segment.matching(&bound));
        });
        let query = |text: bool| {
            best(&|| {
                let q = ns.query().filter(f.clone()).limit(10);
                std::hint::black_box(if text { q.text("alpha") } else { q }.run().unwrap());
            })
        };
        println!("{name:<40} {pass:>9} {found:>12.3} {:>12.3} {:>16.3}", query(false), query(true));
    }
    drop(segment);
    // Counts: of everything, of what a condition passes, and by an attribute's values.
    println!("{:<40} {:>12}", "counts", "ms");
    for (name, f) in [("every document", None), ("kind = pdf", Some(attr("kind").eq("pdf"))), ("owner = user417", Some(attr("owner").eq("user417")))] {
        let ms = best(&|| {
            std::hint::black_box(ns.count(f.as_ref()));
        });
        println!("{name:<40} {ms:>12.3}");
        for by in ["kind", "owner", "labels"] {
            let ms = best(&|| {
                std::hint::black_box(ns.count_by(by, f.as_ref(), 10));
            });
            println!("{:<40} {ms:>12.3}", format!("  by {by}, of {name}"));
        }
    }
    // The ten first in an order, of everything and of what a condition passes.
    println!("{:<40} {:>12}", "the ten first", "ms");
    let some: Vec<(&str, Option<Filter>)> = vec![("", None), (", kind = pdf", Some(attr("kind").eq("pdf"))), (", owner = user417", Some(attr("owner").eq("user417")))];
    for (order, by) in [("by key", None), ("newest (modified)", Some(("modified", true))), ("oldest (modified)", Some(("modified", false))), ("largest (size)", Some(("size", true)))] {
        for (name, f) in &some {
            let ms = best(&|| {
                let q = ns.query().limit(10);
                let q = match f {
                    Some(f) => q.filter(f.clone()),
                    None => q,
                };
                let q = match by {
                    Some((column, descending)) => q.order_by(column, descending),
                    None => q,
                };
                std::hint::black_box(q.run().unwrap());
            });
            println!("{:<40} {ms:>12.3}", format!("{order}{name}"));
        }
    }
    // A fork: a namespace of its own with what this one holds, made of links to its files.
    let mut forks = Vec::new();
    for i in 0..5 {
        let t = Instant::now();
        let fork = ns.fork(&format!("fork-{i}")).unwrap();
        forks.push((t.elapsed().as_secs_f64() * 1000.0, fork.count(None)));
    }
    forks.sort_by(|a, b| a.0.total_cmp(&b.0));
    println!("{:<40} {:>12.3}  ({} documents in it; the slowest of five {:.3})", "a fork, ms", forks[0].0, forks[0].1, forks[4].0);
    // And one after writes that are still in memory: they are written out first.
    ns.upsert_many((n..n + 10_000).map(|k| Document::new(k).text("body", "gamma").string("kind", "pdf"))).unwrap();
    let t = Instant::now();
    let fork = ns.fork("fork-after-writes").unwrap();
    println!("{:<40} {:>12.3}  ({} documents in it)", "a fork with 10,000 unflushed, ms", t.elapsed().as_secs_f64() * 1000.0, fork.count(None));
    drop(fork);
    drop(ns);
    drop(db);
    std::fs::remove_dir_all(&dir).unwrap();
}
