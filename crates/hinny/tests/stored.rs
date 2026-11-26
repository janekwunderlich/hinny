//! Stored text in blob files: merges copy references and leave text where it is, rewrite it out of
//! mostly dead blob files, and delete those no segment points into; readers keep what they mapped.

use std::collections::BTreeSet;
use std::path::Path;

use hinny::index::{Doc, Index};

fn dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("hinny-stored-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn doc(k: u64) -> Doc {
    Doc { key: k, version: 1, passages: vec![vec![format!("title {k}"), format!("the body of document {k}, Größe ").repeat(1 + (k % 9) as usize)]], attrs: vec![], vector: None }
}

fn blobs(d: &Path) -> BTreeSet<u64> {
    std::fs::read_dir(d).unwrap().flatten().filter_map(|e| hinny::blob::id_of(&e.file_name().to_string_lossy())).collect()
}

fn open(d: &Path) -> Index {
    let mut ix = Index::open(d, 2).unwrap();
    ix.store_text = true;
    ix.auto_merge = false;
    ix
}

#[test]
fn merges_copy_text_and_rewrite_only_dead_blobs() {
    let d = dir("merge");
    let mut ix = open(&d);
    for k in 0..300 {
        ix.upsert(&doc(k)).unwrap();
    }
    ix.checkpoint().unwrap();
    for k in 300..600 {
        ix.upsert(&doc(k)).unwrap();
    }
    ix.checkpoint().unwrap();
    let two = blobs(&d);
    assert_eq!(two.len(), 2, "a blob file per checkpoint");
    let check = |ix: &Index, label: &str, live: &dyn Fn(u64) -> bool| {
        for k in [0u64, 1, 299, 300, 451, 599] {
            let want = live(k).then(|| doc(k).passages[0].clone());
            assert_eq!(ix.text(k), want, "{label}: {k}");
        }
    };
    check(&ix, "two segments", &|_| true);

    // A merge without much deleted: the merged segment points into both old blob files.
    for k in (0..600).step_by(10) {
        ix.delete(k).unwrap();
    }
    assert!(ix.start_merge(false) || { ix.merge_factor = 2; ix.start_merge(false) });
    ix.wait_merge().unwrap();
    assert_eq!(ix.stats().segments, 1);
    assert_eq!(blobs(&d), two, "text stays in place");
    check(&ix, "merged, copied", &|k| k % 10 != 0);

    // A reader's snapshot, from before the rewrite.
    let reader = Index::open_read_only(&d, 2).unwrap();

    // Most of the first blob file dead: the next merge moves its live text and deletes it.
    for k in 0..300 {
        if k % 10 != 0 && k % 4 != 0 {
            ix.delete(k).unwrap();
        }
    }
    ix.checkpoint().unwrap();
    assert!(ix.start_merge(true));
    ix.wait_merge().unwrap();
    let now = blobs(&d);
    let first = *two.iter().next().unwrap();
    assert!(!now.contains(&first), "the mostly dead blob file is gone: {now:?}");
    let live = |k: u64| k % 10 != 0 && (k >= 300 || k % 4 == 0);
    check(&ix, "rewritten", &live);
    // The reader still reads from the file it mapped, though it's deleted.
    assert_eq!(reader.text(1), Some(doc(1).passages[0].clone()));
    drop(reader);

    // After a restart, from segments alone.
    drop(ix);
    let ix = open(&d);
    check(&ix, "reopened", &live);
    drop(ix);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn crash_leftovers_are_removed_and_text_survives_the_log() {
    let d = dir("crash");
    let mut ix = open(&d);
    for k in 0..50 {
        ix.upsert(&doc(k)).unwrap();
    }
    ix.checkpoint().unwrap();
    for k in 50..60 {
        ix.upsert(&doc(k)).unwrap();
    }
    ix.simulate_crash();
    // A blob file no segment names (a crash between writing it and its segment).
    std::fs::write(d.join("blob-00000099.fzt"), b"leftover").unwrap();
    let ix = open(&d);
    assert!(!blobs(&d).contains(&99));
    assert_eq!(ix.text(55), Some(doc(55).passages[0].clone()), "from the log");
    assert_eq!(ix.text(5), Some(doc(5).passages[0].clone()), "from a blob file");
    drop(ix);
    // An index that doesn't store text has none, and writes no blob files.
    let d2 = dir("bare");
    let mut bare = Index::open(&d2, 2).unwrap();
    bare.upsert(&doc(1)).unwrap();
    bare.checkpoint().unwrap();
    assert_eq!(bare.text(1), None);
    assert!(blobs(&d2).is_empty());
    drop(bare);
    std::fs::remove_dir_all(&d).unwrap();
    std::fs::remove_dir_all(&d2).unwrap();
}
