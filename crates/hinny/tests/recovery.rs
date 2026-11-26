//! Recovery must fail closed, and a failed persistence attempt must remain retryable.

use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use hinny::index::Index;
use hinny::{Database, Document, Schema};

fn dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("hinny-recovery-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn put(ix: &mut Index, key: u64) {
    ix.put(key, 1, &[vec![format!("document {key}")]]).unwrap();
}

fn files(path: &Path) -> BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(path).unwrap().map(|e| {
        let e = e.unwrap();
        (e.file_name().to_string_lossy().into_owned(), std::fs::read(e.path()).unwrap_or_default())
    }).collect()
}

fn saved(name: &str) -> PathBuf {
    let path = dir(name);
    let mut ix = Index::open(&path, 1).unwrap();
    ix.auto_merge = false;
    put(&mut ix, 1);
    ix.checkpoint().unwrap();
    drop(ix);
    path
}

#[test]
fn unreadable_manifest_preserves_every_file() {
    let path = saved("manifest-utf8");
    std::fs::write(path.join("manifest"), [0xff]).unwrap();
    let before = files(&path);
    assert_eq!(Index::open(&path, 1).err().map(|e| e.kind()), Some(ErrorKind::InvalidData));
    assert_eq!(files(&path), before);
    assert_eq!(Index::open_read_only(&path, 1).err().map(|e| e.kind()), Some(ErrorKind::InvalidData));
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn missing_manifest_with_segments_is_not_an_empty_index() {
    let path = saved("manifest-missing");
    std::fs::remove_file(path.join("manifest")).unwrap();
    let before = files(&path);
    assert!(Index::open(&path, 1).is_err());
    assert_eq!(files(&path), before);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn invalid_manifest_metadata_preserves_every_file() {
    for (label, text) in [
        ("next", "fugudb-index 3\nnext broken\nsegment 1\n"),
        ("collision", "fugudb-index 3\nnext 1\nsegment 1\n"),
        ("unknown", "fugudb-index 3\nnext 2\nsegmen 1\n"),
        ("truncated", "fugudb-index 3\n"),
        ("duplicate", "fugudb-index 3\nnext 2\nsegment 1\nsegment 1\n"),
    ] {
        let path = saved(label);
        std::fs::write(path.join("manifest"), text).unwrap();
        let before = files(&path);
        assert_eq!(Index::open(&path, 1).err().map(|e| e.kind()), Some(ErrorKind::InvalidData), "{label}");
        assert_eq!(files(&path), before, "{label}");
        std::fs::remove_dir_all(path).unwrap();
    }
}

#[test]
fn unreadable_or_truncated_deletions_are_not_empty_deletions() {
    for directory in [false, true] {
        let path = saved(if directory { "del-directory" } else { "del-truncated" });
        let del = path.join("seg-00000001.del");
        if directory {
            std::fs::create_dir(&del).unwrap();
        } else {
            std::fs::write(&del, [1]).unwrap();
        }
        let before = files(&path);
        assert!(Index::open(&path, 1).is_err());
        assert_eq!(files(&path), before);
        assert!(Index::open_read_only(&path, 1).is_err());
        std::fs::remove_dir_all(path).unwrap();
    }
}

#[test]
fn deletion_persistence_failure_is_retryable() {
    let path = saved("del-retry");
    let mut ix = Index::open(&path, 1).unwrap();
    ix.auto_merge = false;
    ix.delete(1).unwrap();
    let obstruction = path.join("seg-00000001.1.del.tmp");
    std::fs::create_dir(&obstruction).unwrap();
    assert!(ix.checkpoint().is_err());
    std::fs::remove_dir(&obstruction).unwrap();
    put(&mut ix, 2);
    ix.checkpoint().unwrap();
    ix.simulate_crash();
    let ix = Index::open(&path, 1).unwrap();
    assert!(ix.get(1).is_none(), "a failed bitmap write must not resurrect a deletion on retry");
    assert!(ix.get(2).is_some());
    drop(ix);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn failed_flush_keeps_memory_and_log_until_publication() {
    let path = dir("flush-retry");
    let mut ix = Index::open(&path, 1).unwrap();
    ix.auto_merge = false;
    put(&mut ix, 1);
    let obstruction = path.join("manifest.tmp");
    std::fs::create_dir(&obstruction).unwrap();
    assert!(ix.checkpoint().is_err());
    put(&mut ix, 2); // Publishing another write must still include the failed flush's data.
    assert!(ix.get(1).is_some());
    assert!(ix.get(2).is_some());
    std::fs::remove_dir(&obstruction).unwrap();
    ix.checkpoint().unwrap();
    ix.simulate_crash();
    let ix = Index::open(&path, 1).unwrap();
    assert_eq!(ix.versions().len(), 2);
    assert!(ix.get(1).is_some());
    assert!(ix.get(2).is_some());
    drop(ix);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn future_wal_is_rejected_before_open_can_checkpoint_or_cleanup() {
    let path = saved("future-wal");
    let mut header = b"fugu-log".to_vec();
    header.extend_from_slice(&u32::MAX.to_le_bytes());
    std::fs::write(path.join("wal-100"), header).unwrap();
    std::fs::write(path.join("seg-99999999.swt"), b"unreferenced").unwrap();
    let before = files(&path);
    assert!(Index::open(&path, 1).is_err());
    assert_eq!(files(&path), before);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn sync_keeps_data_in_memory_and_covers_replayed_wals() {
    let path = dir("sync-replay");
    let mut ix = Index::open(&path, 1).unwrap();
    put(&mut ix, 1);
    ix.sync().unwrap();
    assert_eq!(ix.stats().segments, 0);
    assert!(ix.stats().memory > 0);
    ix.simulate_crash();
    let mut ix = Index::open(&path, 1).unwrap();
    put(&mut ix, 2);
    // Reopening puts new writes in wal-2, but wal-1 still holds the first document.
    std::fs::rename(path.join("wal-1"), path.join("saved-wal")).unwrap();
    assert_eq!(ix.sync().err().map(|e| e.kind()), Some(ErrorKind::NotFound));
    std::fs::rename(path.join("saved-wal"), path.join("wal-1")).unwrap();
    ix.sync().unwrap();
    assert_eq!(ix.stats().segments, 0);
    ix.simulate_crash();
    let ix = Index::open(&path, 1).unwrap();
    assert_eq!(ix.versions().len(), 2);
    let mut reader = Index::open_read_only(&path, 1).unwrap();
    assert_eq!(reader.sync().err().map(|e| e.kind()), Some(ErrorKind::PermissionDenied));
    drop((reader, ix));
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn wal_barrier_precedes_persisting_replacement_deletions() {
    let path = saved("wal-before-delete");
    let mut ix = Index::open(&path, 1).unwrap();
    ix.auto_merge = false;
    ix.sync().unwrap();
    ix.put(1, 2, &[vec!["replacement".into()]]).unwrap();
    // Force the WAL barrier to fail. No replacement deletion may be persisted before it.
    let wal = path.join("wal-3");
    assert!(wal.exists());
    std::fs::rename(&wal, path.join("saved-wal")).unwrap();
    let manifest = std::fs::read(path.join("manifest")).unwrap();
    assert!(ix.checkpoint().is_err());
    assert_eq!(std::fs::read(path.join("manifest")).unwrap(), manifest);
    assert!(!path.join("seg-00000001.del").exists() && !path.join("seg-00000001.1.del").exists());
    std::fs::rename(path.join("saved-wal"), wal).unwrap();
    ix.checkpoint().unwrap();
    ix.simulate_crash();
    let ix = Index::open(&path, 1).unwrap();
    assert_eq!(ix.versions().get(&1), Some(&2));
    drop(ix);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn failed_merge_publication_can_be_retried() {
    let path = dir("merge-retry");
    let mut ix = Index::open(&path, 1).unwrap();
    ix.auto_merge = false;
    for key in 1..=2 {
        put(&mut ix, key);
        ix.checkpoint().unwrap();
    }
    assert!(ix.start_merge(true));
    let obstruction = path.join("manifest.tmp");
    std::fs::create_dir(&obstruction).unwrap();
    assert!(ix.wait_merge().is_err());
    ix.delete(1).unwrap();
    put(&mut ix, 3);
    std::fs::remove_dir(&obstruction).unwrap();
    ix.checkpoint().unwrap();
    assert!(ix.start_merge(true));
    ix.wait_merge().unwrap();
    ix.simulate_crash();
    let ix = Index::open(&path, 1).unwrap();
    assert!(ix.get(1).is_none());
    assert!(ix.get(2).is_some());
    assert!(ix.get(3).is_some());
    drop(ix);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn schema_round_trip_and_failed_write_preserve_metadata() {
    let path = dir("schema").join("nested/database");
    let db = Database::open(&path).unwrap();
    let schema = Schema::new().text("body text", 1.0).string("author name");
    let ns = db.namespace("docs", schema.clone()).unwrap();
    ns.upsert(Document::new(1).text("body text", "hello").string("author name", "Ada")).unwrap();
    ns.sync().unwrap();
    drop(ns);
    let ns = db.namespace("docs", schema.clone()).unwrap();
    assert_eq!(ns.text(1).unwrap()[0].1, "hello");
    drop(ns);
    let file = path.join("docs/schema");
    let original = std::fs::read(&file).unwrap();
    let obstruction = path.join("docs/schema.tmp");
    std::fs::create_dir(&obstruction).unwrap();
    assert!(db.namespace("docs", schema.clone().int("year")).is_err());
    assert_eq!(std::fs::read(&file).unwrap(), original);
    std::fs::remove_dir(&obstruction).unwrap();
    assert!(db.namespace("docs", schema.clone()).is_ok());
    std::fs::write(&file, [0xff]).unwrap();
    assert_eq!(db.namespace("docs", schema).err().map(|e| e.kind()), Some(ErrorKind::InvalidData));
    assert_eq!(std::fs::read(&file).unwrap(), [0xff]);
    for name in ["", "bad\nname", "bad\rname", "bad\tname"] {
        assert_eq!(db.namespace("bad", Schema::new().string(name)).err().map(|e| e.kind()), Some(ErrorKind::InvalidInput));
        assert!(!path.join("bad").exists());
    }
    std::fs::remove_dir_all(path.ancestors().nth(2).unwrap()).unwrap();
}

#[test]
fn manifest_damage_that_still_parses_fails_its_checksum() {
    let path = saved("manifest-checksum");
    let manifest = std::fs::read_to_string(path.join("manifest")).unwrap();
    assert!(manifest.lines().last().unwrap().starts_with("crc32 "), "{manifest}");
    // A plausible manifest naming no segment: before checksums, the index would open empty.
    let damaged = manifest.lines().filter(|l| !l.starts_with("segment ")).collect::<Vec<_>>().join("\n") + "\n";
    std::fs::write(path.join("manifest"), damaged).unwrap();
    let before = files(&path);
    let e = Index::open(&path, 1).err().unwrap();
    assert_eq!(e.kind(), ErrorKind::InvalidData);
    assert!(e.to_string().contains("checksum"), "{e}");
    assert_eq!(files(&path), before, "a refused manifest must leave every file in place");
    assert_eq!(Index::open_read_only(&path, 1).err().map(|e| e.kind()), Some(ErrorKind::InvalidData));
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn deletion_damage_fails_its_checksum() {
    let path = saved("del-checksum");
    let mut ix = Index::open(&path, 1).unwrap();
    ix.auto_merge = false;
    ix.delete(1).unwrap();
    ix.checkpoint().unwrap();
    drop(ix);
    let del = path.join("seg-00000001.1.del");
    let mut bytes = std::fs::read(&del).unwrap();
    assert_eq!(&bytes[..8], b"fugu-del");
    let bit = bytes.len() - 5; // the bitmap's last byte
    bytes[bit] ^= 0x01;
    std::fs::write(&del, bytes).unwrap();
    let before = files(&path);
    assert_eq!(Index::open(&path, 1).err().map(|e| e.kind()), Some(ErrorKind::InvalidData));
    assert_eq!(files(&path), before);
    std::fs::remove_dir_all(path).unwrap();
}

/// Each save of a segment's deletions is a new file that the manifest names, so a manifest and
/// its files are one state: the file before goes, one the manifest names can't go missing
/// unnoticed, and one no manifest names is cleared away.
#[test]
fn deletion_files_are_named_by_the_manifest() {
    let path = saved("del-named");
    let mut ix = Index::open(&path, 1).unwrap();
    ix.auto_merge = false;
    let (first, second) = (path.join("seg-00000001.1.del"), path.join("seg-00000001.2.del"));
    ix.delete(1).unwrap();
    ix.checkpoint().unwrap();
    assert!(first.exists() && !second.exists());
    assert!(std::fs::read_to_string(path.join("manifest")).unwrap().contains("segment 1 1\n"));
    put(&mut ix, 2);
    ix.delete(2).unwrap();
    put(&mut ix, 3);
    ix.checkpoint().unwrap();
    let manifest = std::fs::read_to_string(path.join("manifest")).unwrap();
    // (Key 2 was written and deleted in memory: only a segment's own deletions make a file.)
    assert!(first.exists() && manifest.contains("segment 1 1\n"), "{manifest}");
    ix.delete(3).unwrap();
    ix.checkpoint().unwrap();
    drop(ix);
    // A file no manifest names goes when the index is next opened for writing.
    let stray = path.join("seg-00000001.9.del");
    std::fs::write(&stray, b"left by a save that never finished").unwrap();
    let ix = Index::open(&path, 1).unwrap();
    assert!(!stray.exists() && first.exists());
    assert!(ix.get(1).is_none() && ix.get(3).is_none());
    drop(ix);
    // One the manifest names, gone: an error for a writer and a reader, not "no deletions".
    let saved_bytes = std::fs::read(&first).unwrap();
    std::fs::remove_file(&first).unwrap();
    assert_eq!(Index::open(&path, 1).err().map(|e| e.kind()), Some(ErrorKind::NotFound));
    assert!(Index::open_read_only(&path, 1).is_err());
    std::fs::write(&first, saved_bytes).unwrap();
    let mut ix = Index::open(&path, 1).unwrap();
    ix.auto_merge = false;
    assert!(ix.get(1).is_none());
    // A segment's second save is the next number, and the first goes.
    put(&mut ix, 10);
    put(&mut ix, 11);
    ix.checkpoint().unwrap();
    let saves: Vec<PathBuf> = (1..=2).map(|n| path.join(format!("seg-00000003.{n}.del"))).collect();
    ix.delete(10).unwrap();
    ix.checkpoint().unwrap();
    assert!(saves[0].exists() && !saves[1].exists());
    ix.delete(11).unwrap();
    ix.checkpoint().unwrap();
    assert!(!saves[0].exists() && saves[1].exists());
    assert!(std::fs::read_to_string(path.join("manifest")).unwrap().contains("segment 3 2\n"));
    drop(ix);
    let ix = Index::open(&path, 1).unwrap();
    assert!(ix.get(10).is_none() && ix.get(11).is_none());
    drop(ix);
    std::fs::remove_dir_all(path).unwrap();
}
