//! Data written by earlier versions still reads: each directory in tests/fixtures was written by
//! `examples/make_fixture.rs` at that version, and must stay readable (and writable on top of)
//! by this one. A version that drops one changes what it promises (DESIGN.md, "Formats").

use std::path::{Path, PathBuf};

use hinny::index::Index;
use hinny::{Database, Document, Schema, Value};

fn vector(k: u64) -> Vec<f32> {
    (0..8).map(|i| (((k * 7 + i * 13) % 17) as f32 - 8.0) / 8.0).collect()
}

fn schema() -> Schema {
    Schema::new().text("title", 2.0).text("body", 1.0).int("year").string("kind").vector(8)
}

/// A copy, so tests never change the committed files.
fn copy(fixture: &str, name: &str) -> PathBuf {
    let from = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(fixture);
    let to = std::env::temp_dir().join(format!("hinny-format-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&to);
    fn walk(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for e in std::fs::read_dir(from).unwrap().flatten() {
            if e.file_type().unwrap().is_dir() {
                walk(&e.path(), &to.join(e.file_name()));
            } else {
                std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
            }
        }
    }
    walk(&from, &to);
    to
}

fn live(k: u64) -> bool {
    k < 110 && !(k < 100 && k % 7 == 0) && k != 101
}

fn check(dir: &Path, label: &str, extra: &[u64]) {
    let db = Database::open(dir).unwrap();
    let ns = db.namespace("docs", schema()).unwrap();
    let want: Vec<u64> = (0..110).filter(|&k| live(k)).chain(extra.iter().copied()).collect();
    let mut have: Vec<u64> = ns.versions().into_keys().collect();
    have.sort_unstable();
    assert_eq!(have, want, "{label}: documents");
    assert_eq!(ns.versions()[&43], 44, "{label}: versions");
    // Full text, from segments and from the log.
    let mut word3: Vec<u64> = ns.query().text("word3").limit(100).run().unwrap().iter().map(|h| h.key).collect();
    word3.sort_unstable();
    assert_eq!(word3, want.iter().copied().filter(|k| k % 5 == 3 || extra.contains(k)).collect::<Vec<_>>(), "{label}: text");
    // Attributes, filters and stored text.
    let h = ns.query().text("document 52").limit(1).run().unwrap().remove(0);
    assert_eq!(h.key, 52, "{label}");
    assert_eq!(h.attr("year"), Some(&Value::Int(2052)), "{label}: attributes");
    assert_eq!(h.attr("kind"), Some(&Value::Str("even".into())), "{label}");
    assert_eq!(h.text("title"), Some("Title 52"), "{label}: stored text");
    let (_, s) = h.snippet.as_ref().expect("a snippet");
    assert!(s.highlights.iter().any(|r| &s.text[r.clone()] == "52"), "{label}: snippet {s:?}");
    let odd = ns.query().filter(hinny::attr("kind").eq("odd")).limit(200).run().unwrap();
    assert_eq!(odd.len(), want.iter().filter(|k| *k % 2 == 1 && !extra.contains(k)).count(), "{label}: filter");
    // Every kind of condition: on columns as that version wrote them, in the log, and (after a
    // merge) as this version writes them.
    let keys = |f: hinny::Filter| {
        let mut k: Vec<u64> = ns.query().filter(f).limit(500).run().unwrap().iter().map(|h| h.key).collect();
        k.sort_unstable();
        k
    };
    let pick = |p: &dyn Fn(u64) -> bool| want.iter().copied().filter(|&k| p(k)).collect::<Vec<_>>();
    let kind = |k: u64| if extra.contains(&k) { "" } else if k % 2 == 0 { "even" } else { "odd" };
    assert_eq!(keys(hinny::attr("year").between(2010, 2049)), pick(&|k| (10..=49).contains(&k)), "{label}: between");
    assert_eq!(keys(hinny::attr("year").one_of([2003i64, 2052, 2500, 1999])), pick(&|k| [3, 52, 500].contains(&k)), "{label}: one of");
    assert_eq!(keys(hinny::attr("year").gt(2104)), pick(&|k| k > 104), "{label}: above");
    assert_eq!(keys(hinny::attr("year").lt(2002)), pick(&|k| k < 2), "{label}: below");
    assert_eq!(keys(hinny::attr("year").exists()), want, "{label}: has a year");
    assert_eq!(keys(hinny::attr("kind").exists()), pick(&|k| !kind(k).is_empty()), "{label}: has a kind");
    assert_eq!(keys(hinny::attr("kind").prefix("EV")), pick(&|k| kind(k) == "even"), "{label}: prefix");
    assert_eq!(keys(hinny::attr("kind").contains("d")), pick(&|k| kind(k) == "odd"), "{label}: contains");
    assert_eq!(keys(hinny::attr("kind").one_of(["odd", "even", "neither"])), pick(&|k| !kind(k).is_empty()), "{label}: one of the kinds");
    assert_eq!(keys(hinny::attr("kind").ne("odd")), pick(&|k| kind(k) != "odd"), "{label}: not");
    assert_eq!(keys(hinny::attr("kind").eq("even") & hinny::attr("year").between(2040, 2060)), pick(&|k| kind(k) == "even" && (40..=60).contains(&k)), "{label}: both");
    // Vectors (vector(k) repeats every 17 keys).
    for k in [3u64, 55, 104] {
        let top = &ns.query().vector(vector(k)).limit(1).run().unwrap()[0];
        assert!((top.score - 1.0).abs() < 1e-4 && top.key % 17 == k % 17, "{label}: vector {k} → {}", top.key);
    }
}

#[test]
fn reads_v0_1() {
    let dir = copy("v0.1", "read");
    check(&dir, "as written by 0.1", &[]);
    // A read-only snapshot needs no write access and changes nothing.
    let before: Vec<_> = std::fs::read_dir(dir.join("docs")).unwrap().flatten().map(|e| (e.file_name(), e.metadata().unwrap().len())).collect();
    let r = Index::open_read_only(&dir.join("docs"), 2).unwrap();
    assert_eq!(r.versions().len(), 94);
    drop(r);
    let after: Vec<_> = std::fs::read_dir(dir.join("docs")).unwrap().flatten().map(|e| (e.file_name(), e.metadata().unwrap().len())).collect();
    assert_eq!(before.len(), after.len());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn reads_v0_2() {
    let dir = copy("v0.2", "read2");
    check(&dir, "as written by 0.2", &[]);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn reads_v0_3() {
    let dir = copy("v0.3", "read3");
    check(&dir, "as written by 0.3", &[]);
    // Analyzers, a sparse field, and batches, the last ones only in the log.
    let db = Database::open(&dir).unwrap();
    let more = db.namespace("more", Schema::new().text_with("body", 1.0, hinny::Analyzer::english()).sparse("terms")).unwrap();
    let mut keys: Vec<u64> = more.versions().into_keys().collect();
    keys.sort_unstable();
    assert_eq!(keys, (0..21).filter(|k| *k != 4 && *k != 5).collect::<Vec<_>>());
    assert_eq!(more.query().text("run report").limit(30).run().unwrap().len(), 19, "stemmed");
    let t2: Vec<u64> = more.query().sparse("terms", [("t2", 1.0)]).limit(30).run().unwrap().iter().map(|h| h.key).collect();
    assert_eq!(t2, vec![20, 17, 14, 11, 8, 2], "by weight");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn writes_on_top_of_v0_1() {
    let dir = copy("v0.1", "write");
    {
        let db = Database::open(&dir).unwrap();
        let ns = db.namespace("docs", schema()).unwrap();
        ns.upsert(Document::new(500).text("title", "New").text("body", "written by this version word3").int("year", 2500).vector(vector(500))).unwrap();
        // Old and new segments merged together.
        ns.optimize().unwrap();
        assert_eq!(ns.stats().segments, 1);
    }
    let db = Database::open(&dir).unwrap();
    let ns = db.namespace("docs", schema()).unwrap();
    assert_eq!(ns.versions().len(), 95);
    assert_eq!(ns.query().text("written").run().unwrap()[0].text("title"), Some("New"));
    drop(ns);
    // Everything else as before.
    std::fs::remove_dir_all(dir.join("docs").join("lock")).ok();
    check(&dir, "after writing", &[500]);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn refuses_newer_formats() {
    let dir = copy("v0.1", "newer");
    let docs = dir.join("docs");
    std::fs::write(docs.join("manifest"), "fugudb-index 99\nnext 5\nsegment 3\nsegment 4\n").unwrap();
    for e in [Index::open(&docs, 2).err().unwrap(), Index::open_read_only(&docs, 2).err().unwrap()] {
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
        assert!(e.to_string().contains("format 99, written by a newer hinny"), "{e}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn upgrading_keeps_older_deletion_files_and_seals_the_manifest() {
    // 0.3 wrote raw deletion bitmaps and an unchecked manifest. Writing on top seals the manifest
    // and writes new deletions with checksums; the untouched raw files must keep reading.
    let dir = copy("v0.3", "upgrade4");
    let docs = dir.join("docs");
    let raw: Vec<_> = std::fs::read_dir(&docs).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "del")).collect();
    let db = Database::open(&dir).unwrap();
    let ns = db.namespace("docs", schema()).unwrap();
    let before = ns.versions().len();
    let gone = *ns.versions().keys().next().unwrap();
    ns.delete(gone).unwrap();
    ns.flush().unwrap();
    drop(ns);
    let manifest = std::fs::read_to_string(docs.join("manifest")).unwrap();
    assert!(manifest.starts_with("fugudb-index 7\n") && manifest.lines().last().unwrap().starts_with("crc32 "), "{manifest}");
    let sealed = std::fs::read_dir(&docs).unwrap().flatten().filter(|e| std::fs::read(e.path()).is_ok_and(|b| b.starts_with(b"fugu-del"))).count();
    assert!(sealed >= 1, "the new deletion is written with a checksum");
    let still_raw = raw.iter().filter(|p| std::fs::read(p).is_ok_and(|b| !b.starts_with(b"fugu-del"))).count();
    std::fs::remove_dir_all(docs.join("lock")).ok();
    let ns = Database::open(&dir).unwrap().namespace("docs", schema()).unwrap();
    assert_eq!(ns.versions().len(), before - 1);
    assert!(ns.get(gone).is_none());
    assert!(still_raw >= 1, "an untouched raw deletion file must have been read beside the sealed ones");
    drop(ns);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// 5,000 clustered vectors as 0.3 wrote them (segment format 7: the codes one after another and
/// the rows in document order, which a search lays out in memory the first time): searched as
/// they are, reported as written before there were checksums, and rewritten by a merge the way
/// this version stores them, with the same answers.
#[test]
fn searches_clustered_vectors_written_by_0_3() {
    // The fixture's vectors: 40 topics, and every 760th document the same vector again.
    let clustered = |k: u64| -> Vec<f32> {
        let t = k % 40;
        (0..16u64).map(|i| (((t * 31 + i * 17) % 23) as f32 - 11.0) / 11.0 + (((k * 7 + i * 13) % 19) as f32 - 9.0) / 60.0).collect()
    };
    let dir = copy("v0.3-vectors", "vectors3");
    let db = Database::open(&dir).unwrap();
    let ns = db.namespace("vecs", Schema::new().int("n").vector(16)).unwrap();
    assert_eq!(ns.versions().len(), 4998, "two deleted");
    let probes: Vec<u64> = (0..5000).step_by(97).collect();
    let nearest = || -> Vec<Vec<u64>> { probes.iter().map(|&k| ns.query().vector(clustered(k)).limit(5).run().unwrap().iter().map(|h| h.key).collect()).collect() };
    let before = nearest();
    // What it finds is as near as the five nearest of every live document, compared one by one.
    let unit = |k: u64| -> Vec<f32> {
        let v = clustered(k);
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    };
    let dist = |a: &[f32], k: u64| -> f32 { a.iter().zip(unit(k)).map(|(x, y)| (x - y) * (x - y)).sum() };
    for (&k, hits) in probes.iter().zip(&before) {
        let q = unit(k);
        let mut all: Vec<f32> = (0..5000).filter(|d| *d != 7 && *d != 4999).map(|d| dist(&q, d)).collect();
        all.sort_by(f32::total_cmp);
        let found: Vec<f32> = hits.iter().map(|&h| dist(&q, h)).collect();
        assert_eq!(hits.len(), 5);
        assert!(found.iter().zip(&all).all(|(f, a)| (f - a).abs() < 1e-4) && !hits.contains(&7) && !hits.contains(&4999), "{k}: {hits:?} at {found:?}, the nearest at {:?}", &all[..5]);
    }
    let s = ns.stats();
    assert!(s.vector_caches == 1 && s.vector_cache_bytes > 5000, "a format 7 segment is laid out in memory: {s:?}");
    let v = ns.verify().unwrap();
    assert_eq!((v.segments, v.unchecked_segments), (0, 1));
    // Rewritten: stored as a search reads it, with checksums.
    ns.optimize().unwrap();
    assert_eq!(nearest(), before);
    let s = ns.stats();
    assert_eq!((s.segments, s.vector_caches, s.vector_cache_bytes), (1, 0, 0));
    let v = ns.verify().unwrap();
    assert_eq!((v.segments, v.unchecked_segments), (1, 0));
    drop(ns);
    drop(db);
    std::fs::remove_dir_all(&dir).unwrap();
}
