//! Damage is found: every segment carries a checksum of its header (checked when it's opened) and
//! of every mebibyte after it, and stored text a checksum per block (`verify` reads them all).

use std::path::{Path, PathBuf};

use hinny::{Database, Document, Schema};

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("hinny-integrity-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn schema() -> Schema {
    Schema::new().text("body", 1.0).int("n").vector(8)
}

/// 600 documents with text, an attribute and a vector, in two segments.
fn write(d: &Path) {
    let ns = Database::open(d).unwrap().namespace("docs", schema()).unwrap();
    let doc = |k: u64| Document::new(k).text("body", format!("document {k} about topic{} at some length, so that its text fills blocks", k % 7)).int("n", k as i64).vector((0..8).map(|i| ((k * 3 + i * 5) % 11) as f32 - 5.0).collect());
    ns.upsert_many((0..400).map(doc)).unwrap();
    ns.flush().unwrap();
    ns.upsert_many((400..600).map(doc)).unwrap();
    ns.flush().unwrap();
}

fn files(d: &Path, ext: &str) -> Vec<PathBuf> {
    let mut f: Vec<PathBuf> = std::fs::read_dir(d.join("docs")).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == ext)).collect();
    f.sort();
    f
}

/// Flips one bit of `path` at `at` (from the end if negative).
fn flip(path: &Path, at: i64) {
    let mut bytes = std::fs::read(path).unwrap();
    let i = if at < 0 { bytes.len() - (-at) as usize } else { at as usize };
    bytes[i] ^= 0x10;
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn whole_files_verify() {
    let d = dir("whole");
    write(&d);
    let ns = Database::open(&d).unwrap().namespace("docs", schema()).unwrap();
    let v = ns.verify().unwrap();
    assert_eq!((v.segments, v.unchecked_segments, v.blobs), (2, 0, 2));
    // And after a merge, and with writes still in memory.
    ns.optimize().unwrap();
    ns.upsert(Document::new(1000).text("body", "in memory")).unwrap();
    let v = ns.verify().unwrap();
    assert_eq!((v.segments, v.unchecked_segments), (1, 0));
    drop(ns);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_flipped_bit_in_a_segment_is_found() {
    let d = dir("segment");
    write(&d);
    let segment = files(&d, "swt").remove(0);
    let len = std::fs::metadata(&segment).unwrap().len() as i64;
    // In the postings, in the middle, and in the last bytes the checksums cover.
    for at in [300, len / 2, len * 9 / 10] {
        flip(&segment, at);
        let ns = Database::open(&d).unwrap().namespace("docs", schema()).unwrap();
        let e = ns.verify().unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
        assert!(e.to_string().contains("segment 1 is damaged"), "{e}");
        drop(ns);
        flip(&segment, at);
    }
    // Put back as it was, it's whole again.
    let ns = Database::open(&d).unwrap().namespace("docs", schema()).unwrap();
    assert_eq!(ns.verify().unwrap().segments, 2);
    drop(ns);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_damaged_header_or_checksum_table_refuses_to_open() {
    let d = dir("header");
    write(&d);
    let segment = files(&d, "swt").remove(0);
    // The header (the document count, an offset), and the checksums themselves at the end.
    for at in [8, 100, 170, -1, -6] {
        flip(&segment, at);
        let e = Database::open(&d).and_then(|db| db.namespace("docs", schema())).err().unwrap_or_else(|| panic!("opened with byte {at} flipped"));
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{at}: {e}");
        assert!(e.to_string().contains("is damaged"), "{at}: {e}");
        flip(&segment, at);
    }
    // A file cut short.
    let bytes = std::fs::read(&segment).unwrap();
    std::fs::write(&segment, &bytes[..bytes.len() - 40]).unwrap();
    assert!(Database::open(&d).and_then(|db| db.namespace("docs", schema())).is_err());
    std::fs::write(&segment, &bytes).unwrap();
    let ns = Database::open(&d).unwrap().namespace("docs", schema()).unwrap();
    assert_eq!(ns.versions().len(), 600);
    drop(ns);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_flipped_bit_in_stored_text_is_found() {
    let d = dir("text");
    write(&d);
    let blob = files(&d, "fzt").remove(0);
    let len = std::fs::metadata(&blob).unwrap().len() as i64;
    for at in [len / 3, len - 3] {
        flip(&blob, at);
        let ns = Database::open(&d).unwrap().namespace("docs", schema()).unwrap();
        let e = ns.verify().unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
        assert!(e.to_string().contains("stored text file"), "{e}");
        drop(ns);
        flip(&blob, at);
    }
    let ns = Database::open(&d).unwrap().namespace("docs", schema()).unwrap();
    assert_eq!(ns.verify().unwrap().blobs, 2);
    drop(ns);
    std::fs::remove_dir_all(&d).unwrap();
}
