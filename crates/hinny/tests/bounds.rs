//! Inputs at the edges of what's representable: integer ranges past i64's ends, a limit of 0,
//! a first batch whose vectors disagree, weights that pruning can't bound, and high-dimensional
//! vectors whose scan sums outgrow 16 bits. Each is refused or answered exactly, never wrapped.

use std::io::ErrorKind;

use hinny::index::{Doc, Index, Op};
use hinny::{Database, Document, Schema, attr};

fn dir(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("hinny-bounds-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

#[test]
fn ranges_past_the_integers_ends_match_nothing() {
    let path = dir("ranges");
    let db = Database::open(&path).unwrap();
    let ns = db.namespace("docs", Schema::new().text("body", 1.0).int("year")).unwrap();
    ns.upsert(Document::new(1).text("body", "a").int("year", 0)).unwrap();
    ns.upsert(Document::new(2).text("body", "a").int("year", i64::MAX)).unwrap();
    ns.upsert(Document::new(3).text("body", "a").int("year", i64::MIN + 1)).unwrap();
    ns.upsert(Document::new(4).text("body", "a")).unwrap();
    for at in ["memory", "disk"] {
        let keys = |f| ns.query().filter(f).limit(10).run().unwrap().iter().map(|h| h.key).collect::<Vec<_>>();
        assert_eq!(keys(attr("year").gt(i64::MAX)), [], "{at}");
        assert_eq!(keys(attr("year").lt(i64::MIN)), [], "{at}");
        assert_eq!(keys(attr("year").gt(i64::MAX - 1)), [2], "{at}");
        assert_eq!(keys(attr("year").lt(i64::MIN + 2)), [3], "{at}");
        // Composed: nothing or'd away, negated into everything with a year.
        assert_eq!(keys(attr("year").gt(i64::MAX) | attr("year").eq(0i64)), [1], "{at}");
        assert_eq!(keys(attr("year").gt(i64::MAX) & attr("year").exists()), [], "{at}");
        assert_eq!(keys(!attr("year").gt(i64::MAX)), [1, 2, 3, 4], "{at}");
        // With text, too.
        assert_eq!(ns.query().text("a").filter(attr("year").lt(i64::MIN)).run().unwrap().len(), 0, "{at}");
        ns.flush().unwrap();
    }
    drop(ns);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn a_limit_of_zero_returns_nothing() {
    let path = dir("zero");
    let db = Database::open(&path).unwrap();
    let ns = db.namespace("docs", Schema::new().text("body", 1.0).vector(3)).unwrap();
    for k in 0..50u64 {
        ns.upsert(Document::new(k).text("body", "word").vector(vec![1.0, k as f32, 2.0])).unwrap();
    }
    for _ in 0..2 {
        assert!(ns.query().text("word").limit(0).run().unwrap().is_empty());
        assert!(ns.query().vector(vec![1.0, 1.0, 1.0]).limit(0).run().unwrap().is_empty());
        assert!(ns.query().text("word").vector(vec![1.0, 1.0, 1.0]).limit(0).run().unwrap().is_empty());
        assert!(ns.query().limit(0).run().unwrap().is_empty());
        ns.flush().unwrap();
    }
    let mut ix = Index::open(&path.join("raw"), 1).unwrap();
    let doc = Doc { key: 1, version: 1, passages: vec![vec!["x".into()]], attrs: Vec::new(), vector: Some(vec![1.0, 2.0]) };
    ix.write(&[Op::Put(&doc)]).unwrap();
    for _ in 0..2 {
        assert!(ix.nearest_exact(&[1.0, 1.0], 0, None).is_empty());
        assert!(ix.nearest(&[1.0, 1.0], 0, None, 0).is_empty());
        ix.checkpoint().unwrap();
    }
    drop((ns, ix));
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn a_first_batch_with_vectors_of_different_lengths_is_refused_whole() {
    let path = dir("dims");
    let mut ix = Index::open(&path, 1).unwrap();
    let doc = |key, v: Vec<f32>| Doc { key, version: 1, passages: vec![vec!["x".into()]], attrs: Vec::new(), vector: Some(v) };
    let (a, b) = (doc(1, vec![1.0, 0.0]), doc(2, vec![1.0, 0.0, 0.0]));
    assert_eq!(ix.write(&[Op::Put(&a), Op::Put(&b)]).unwrap_err().kind(), ErrorKind::InvalidInput);
    assert_eq!(ix.versions().len(), 0, "nothing of it applied");
    assert_eq!(ix.dims(), 0);
    ix.write(&[Op::Put(&b)]).unwrap();
    assert_eq!(ix.dims(), 3);
    ix.simulate_crash();
    let ix = Index::open(&path, 1).unwrap();
    assert_eq!(ix.versions().into_keys().collect::<Vec<_>>(), [2], "only the accepted batch was logged");
    drop(ix);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn weights_pruning_cant_bound_are_refused() {
    let path = dir("weights");
    let db = Database::open(&path).unwrap();
    for b in [f32::NAN, f32::INFINITY, -1.0] {
        assert_eq!(db.namespace("docs", Schema::new().text("body", b)).err().map(|e| e.kind()), Some(ErrorKind::InvalidInput), "boost {b}");
    }
    let ns = db.namespace("docs", Schema::new().text("body", 0.0).text("title", 2.0).sparse("terms")).unwrap();
    ns.upsert(Document::new(1).text("title", "a").sparse("terms", &[("t", 1.0), ("big", 5e7)])).unwrap();
    for w in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(ns.query().sparse("terms", [("t", w)]).run().unwrap_err().kind(), ErrorKind::InvalidInput, "weight {w}");
    }
    // Weights of 0 or less are left out, as before.
    assert_eq!(ns.query().sparse("terms", [("t", 0.0), ("t", -1.0)]).run().unwrap().len(), 0);
    assert_eq!(ns.query().sparse("terms", [("t", 1.0)]).run().unwrap().len(), 1);
    // A document's weight past the range is left out, not wrapped.
    assert_eq!(ns.query().sparse("terms", [("big", 1.0)]).run().unwrap().len(), 0);
    drop(ns);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn high_dimensional_vectors_find_the_same_as_exact_search() {
    let path = dir("wide");
    let mut ix = Index::open(&path, 1).unwrap();
    // Random queries sum to about 7.5 a dimension: past 16 bits from about 8,700.
    let dims = 16384;
    let mut s = 7u64;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % 2000) as f32 / 1000.0 - 1.0
    };
    let docs: Vec<Doc> = (0..300u64).map(|key| Doc { key, version: 1, passages: vec![vec!["x".into()]], attrs: Vec::new(), vector: Some((0..dims).map(|_| rnd()).collect()) }).collect();
    let ops: Vec<Op> = docs.iter().map(Op::Put).collect();
    ix.write(&ops).unwrap();
    ix.checkpoint().unwrap();
    for q in 0..5 {
        // Near a stored vector, so the nearest are clear.
        let mut v = docs[q * 60].vector.clone().unwrap();
        v.iter_mut().for_each(|x| *x += 0.1 * rnd());
        let exact: Vec<u64> = ix.nearest_exact(&v, 10, None).iter().map(|f| f.key).collect();
        let ann: Vec<u64> = ix.nearest(&v, 10, None, 0).iter().map(|f| f.key).collect();
        assert_eq!(ann[0], (q * 60) as u64);
        let common = ann.iter().filter(|k| exact.contains(k)).count();
        assert!(common >= 9, "query {q}: {common} of 10 in common");
    }
    drop(ix);
    std::fs::remove_dir_all(path).unwrap();
}
