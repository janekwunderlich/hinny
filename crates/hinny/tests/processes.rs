//! One writer, many readers: a second writer is refused, readers can't write, and snapshots
//! opened while a writer checkpoints and merges are always whole.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use hinny::index::{Doc, Index};

fn dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("hinny-proc-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn doc(key: u64) -> Doc {
    Doc { key, version: 1, passages: vec![vec![format!("item number{key} common")]], attrs: vec![], vector: None }
}

#[test]
fn one_writer_and_read_only_snapshots() {
    let d = dir("lock");
    let mut w = Index::open(&d, 1).unwrap();
    // The lock is per open file, so this holds within a process too.
    assert_eq!(Index::open(&d, 1).err().map(|e| e.kind()), Some(std::io::ErrorKind::WouldBlock));
    for k in 0..10 {
        w.upsert(&doc(k)).unwrap();
    }
    let before = Index::on_disk(&d);
    // A reader sees the log before any checkpoint.
    let mut r = Index::open_read_only(&d, 1).unwrap();
    assert!(r.read_only());
    assert_eq!(r.versions().len(), 10);
    assert_eq!(r.upsert(&doc(99)).err().map(|e| e.kind()), Some(std::io::ErrorKind::PermissionDenied));
    assert_eq!(r.checkpoint().err().map(|e| e.kind()), Some(std::io::ErrorKind::PermissionDenied));
    w.checkpoint().unwrap();
    assert_ne!(Index::on_disk(&d), before);
    assert_eq!(Index::open_read_only(&d, 1).unwrap().versions().len(), 10);
    // Dropping a reader changes nothing; dropping the writer frees the lock.
    drop(r);
    drop(w);
    let w = Index::open(&d, 1).unwrap();
    assert_eq!(w.versions().len(), 10);
    drop(w);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn snapshots_during_checkpoints_and_merges_are_whole() {
    let d = dir("race");
    let mut w = Index::open(&d, 1).unwrap();
    w.flush_passages = 300;
    w.merge_factor = 2;
    let done = Arc::new(AtomicBool::new(false));
    let reader = {
        let (d, done) = (d.clone(), done.clone());
        std::thread::spawn(move || {
            let (mut snapshots, mut last) = (0, 0);
            while !done.load(Ordering::Relaxed) {
                let r = Index::open_read_only(&d, 1).expect("a snapshot opens while the writer works");
                let n = r.versions().len();
                // Keys are written in order and never deleted: a snapshot can't have fewer than
                // an earlier one, and has every key below its count.
                assert!(n >= last, "{n} after {last}");
                let v = r.versions();
                assert!((0..n as u64).all(|k| v.contains_key(&k)), "a hole in a snapshot of {n}");
                last = n;
                snapshots += 1;
            }
            snapshots
        })
    };
    for k in 0..6000 {
        w.upsert(&doc(k)).unwrap();
        if k % 997 == 0 {
            w.checkpoint().unwrap();
        }
    }
    w.checkpoint().unwrap();
    w.wait_merge().unwrap();
    done.store(true, Ordering::Relaxed);
    let snapshots = reader.join().unwrap();
    assert!(snapshots > 5, "only {snapshots} snapshots");
    assert_eq!(Index::open_read_only(&d, 1).unwrap().versions().len(), 6000);
    drop(w);
    std::fs::remove_dir_all(&d).unwrap();
}
