//! Limits an embedded index needs to keep its caller safe: batches that couldn't be logged
//! faithfully, or are larger than allowed, are refused before anything is written; memory flushes by
//! bytes as well as passages; what snapshots pin is visible; queries stop at a deadline or when
//! cancelled, with an error rather than partial results.

use std::io::ErrorKind;
use std::time::Duration;

use hinny::index::{Doc, Index, Op};
use hinny::{Cancel, Database, Document, Schema, Value};

fn dir(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("hinny-budgets-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn doc(key: u64, text: &str) -> Doc {
    Doc { key, version: 1, passages: vec![vec![text.to_string()]], attrs: Vec::new(), vector: None }
}

#[test]
fn batches_over_the_limit_are_refused_before_anything_is_written() {
    let path = dir("admit");
    let mut ix = Index::open(&path, 1).unwrap();
    ix.put(1, 1, &[vec!["kept".into()]]).unwrap();
    // The exact logged size of a batch of two: what one more byte of limit allows.
    let batch = [doc(2, &"a".repeat(1000)), doc(3, "b")];
    let ops: Vec<Op> = batch.iter().map(Op::Put).collect();
    let before = ix.stats().log_bytes;
    ix.write(&ops).unwrap();
    let logged = (ix.stats().log_bytes - before) as usize - 8; // the record, without its length and checksum
    let again = [doc(4, &"a".repeat(1000)), doc(5, "b")];
    let ops: Vec<Op> = again.iter().map(Op::Put).collect();
    ix.max_batch_bytes = logged - 1;
    let e = ix.write(&ops).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidInput, "{e}");
    assert!(ix.get(4).is_none() && ix.get(5).is_none(), "nothing of a refused batch is applied");
    ix.max_batch_bytes = logged;
    ix.write(&ops).unwrap();
    assert!(ix.get(4).is_some() && ix.get(5).is_some(), "a refused batch doesn't poison the writer");
    // Lengths the log writes in 16 bits are refused, not wrapped.
    let many = Doc { attrs: (0..=u16::MAX as usize).map(|i| (format!("a{i}"), Value::Int(1))).collect(), ..doc(6, "x") };
    assert_eq!(ix.write(&[Op::Put(&many)]).unwrap_err().kind(), ErrorKind::InvalidInput);
    let long = Doc { attrs: vec![("n".repeat(u16::MAX as usize + 1), Value::Int(1))], ..doc(7, "x") };
    assert_eq!(ix.write(&[Op::Put(&long)]).unwrap_err().kind(), ErrorKind::InvalidInput);
    ix.simulate_crash();
    let ix = Index::open(&path, 1).unwrap();
    assert_eq!(ix.versions().len(), 5, "only the accepted batches were logged");
    drop(ix);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn memory_flushes_by_bytes_as_well_as_passages() {
    let path = dir("bytes");
    let mut ix = Index::open(&path, 1).unwrap();
    ix.flush_passages = 1_000_000;
    ix.flush_bytes = 64 << 10;
    ix.auto_merge = false; // so the segments count the flushes
    let big = "word ".repeat(4_000); // 20 kB a document
    for k in 0..40 {
        ix.put(k, 1, &[vec![big.clone()]]).unwrap();
        let s = ix.stats();
        // While a flush runs, the active memtable fills to twice the limit before writes wait for
        // it; then it's frozen whole and written out while the next fills: at most about four
        // times the limit, each memtable past its mark by one document.
        assert!(s.memtable_bytes <= 4 * (64 << 10) + 2 * big.len(), "memory stays bounded by the byte limit: {} bytes", s.memtable_bytes);
    }
    ix.checkpoint().unwrap();
    assert!(ix.stats().segments >= 2, "a few large documents flushed by bytes, not passages");
    assert_eq!(ix.versions().len(), 40);
    drop(ix);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn segments_a_snapshot_pins_after_a_merge_are_counted_until_it_goes() {
    let (path, db) = {
        let p = dir("pinned");
        (p.clone(), Database::open(&p).unwrap())
    };
    let ns = db.namespace("docs", Schema::new().text("body", 1.0)).unwrap();
    for round in 0..3u64 {
        for k in 0..50 {
            ns.upsert(Document::new(round * 100 + k).text("body", &format!("item {k} of round {round}"))).unwrap();
        }
        ns.flush().unwrap();
    }
    let snapshot = ns.snapshot();
    ns.optimize().unwrap();
    let s = ns.stats();
    assert!(s.pinned_segments >= 2 && s.pinned_bytes > 0, "{s:?}");
    assert_eq!(snapshot.query().text("item").limit(500).run().unwrap().len(), 150, "the pinned snapshot still reads");
    drop(snapshot);
    assert_eq!(ns.stats().pinned_segments, 0);
    drop(ns);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn queries_stop_at_a_deadline_or_when_cancelled() {
    let path = dir("deadline");
    let db = Database::open(&path).unwrap();
    let ns = db.namespace("docs", Schema::new().text("body", 1.0).vector(4)).unwrap();
    for k in 0..2_000u64 {
        let v = vec![(k % 7) as f32, (k % 11) as f32, (k % 13) as f32, 1.0];
        ns.upsert(Document::new(k).text("body", &format!("common words and item {k}")).vector(v)).unwrap();
    }
    ns.flush().unwrap();
    let q = || ns.query().text("common words").vector(vec![1.0, 2.0, 3.0, 1.0]).snippet(0);
    assert_eq!(q().deadline(Duration::ZERO).run().unwrap_err().kind(), ErrorKind::TimedOut);
    let cancel = Cancel::new();
    cancel.cancel();
    assert_eq!(q().cancel(&cancel).run().unwrap_err().kind(), ErrorKind::Interrupted);
    assert_eq!(ns.query().vector(vec![1.0, 2.0, 3.0, 1.0]).cancel(&cancel).run().unwrap_err().kind(), ErrorKind::Interrupted);
    // A deadline that isn't reached changes nothing.
    let plain = q().run().unwrap();
    let timed = q().deadline(Duration::from_secs(60)).cancel(&Cancel::new()).run().unwrap();
    assert_eq!(plain.iter().map(|h| h.key).collect::<Vec<_>>(), timed.iter().map(|h| h.key).collect::<Vec<_>>());
    // Runaway queries are refused.
    let huge = (0..5000).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
    assert_eq!(ns.query().text(&huge).run().unwrap_err().kind(), ErrorKind::InvalidInput);
    drop(ns);
    std::fs::remove_dir_all(path).unwrap();
}
