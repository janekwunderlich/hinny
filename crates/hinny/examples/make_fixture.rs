//! Writes a small database with this version's formats, for tests that newer versions still read
//! it (`tests/formats.rs`).   cargo run -p hinny --example make_fixture -- DIR
//!
//! Everything a database holds: two text fields, int and string attributes, vectors, segments
//! (one merged), deletions, and changes still only in the write-ahead log. From 0.3 a second
//! namespace, "more": an English field, a sparse field, and batches of puts and deletes.

use hinny::{Analyzer, Change, Database, Document, Schema};

pub fn vector(k: u64) -> Vec<f32> {
    (0..8).map(|i| (((k * 7 + i * 13) % 17) as f32 - 8.0) / 8.0).collect()
}

fn main() -> std::io::Result<()> {
    let dir = std::env::args().nth(1).expect("usage: make_fixture DIR");
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(&dir)?;
    let ns = db.namespace("docs", Schema::new().text("title", 2.0).text("body", 1.0).int("year").string("kind").vector(8))?;
    let doc = |k: u64| {
        Document::new(k)
            .version(k + 1)
            .text("title", format!("Title {k}"))
            .text("body", format!("body of document {k} with word{} and Größe", k % 5))
            .int("year", 2000 + k as i64)
            .string("kind", if k % 2 == 0 { "even" } else { "odd" })
            .vector(vector(k))
    };
    // Two segments, merged into one; a third segment; deletions; then the log alone.
    ns.upsert_many((0..40).map(doc))?;
    ns.flush()?;
    ns.upsert_many((40..80).map(doc))?;
    ns.optimize()?;
    ns.upsert_many((80..100).map(doc))?;
    ns.flush()?;
    for k in (0..100).step_by(7) {
        ns.delete(k)?;
    }
    ns.flush()?;
    ns.upsert_many((100..110).map(doc))?;
    ns.delete(101)?;
    // 0.3: analyzers, sparse fields, mixed batches.
    let more = db.namespace("more", Schema::new().text_with("body", 1.0, Analyzer::english()).sparse("terms"))?;
    let m = |k: u64| Document::new(k).text("body", format!("running reports number {k}")).sparse("terms", &[(format!("t{}", k % 3), 1.0 + k as f32 / 10.0), ("all".to_string(), 0.5)]);
    more.apply((0..20).map(|k| Change::Put(m(k))))?;
    more.flush()?;
    more.apply([Change::Delete(4), Change::Put(m(20)), Change::Delete(5)])?;
    // Leave the last changes in the log: no checkpoint on the way out.
    std::process::exit(0)
}
