//! Forks: a namespace of its own with what another holds, made of links to its files. The two
//! hold the same at the fork and are apart from then on, through writes, deletes, merges,
//! reopening, and the other's deletion; and a fork that fails leaves nothing behind.

use std::path::PathBuf;

use hinny::{Database, Document, Namespace, Schema, Value, attr};

fn db(name: &str) -> (Database, PathBuf) {
    let dir = std::env::temp_dir().join(format!("hinny-fork-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    (Database::open(&dir).unwrap(), dir)
}

/// One test at a time: injected failures (the `failpoints` feature) are the whole process's.
static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Why a fork was refused.
fn refused(fork: std::io::Result<Namespace>) -> std::io::ErrorKind {
    fork.err().expect("a fork that should have been refused").kind()
}

fn schema() -> Schema {
    Schema::new().text("body", 1.0).int("n").tags("labels").vector(2)
}

fn doc(key: u64, word: &str) -> Document {
    let angle = key as f32 * 0.1;
    Document::new(key).text("body", format!("{word} number {key}")).int("n", key as i64).tags("labels", [word, if key % 2 == 0 { "even" } else { "odd" }]).vector(vec![angle.cos(), angle.sin()])
}

/// Everything a namespace holds, as its answers: every document's attributes and text by key,
/// a count per tag, and the keys a text, a vector and a filtered search find.
fn state(ns: &Namespace) -> String {
    let mut all = ns.query().limit(100_000).run().unwrap();
    all.sort_by_key(|h| h.key);
    let docs: Vec<String> = all.iter().map(|h| format!("{} {:?} {:?}", h.key, ns.get(h.key).unwrap(), h.text)).collect();
    let keys = |q: hinny::Query| q.limit(20).run().unwrap().iter().map(|h| h.key).collect::<Vec<_>>();
    format!(
        "{}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}",
        docs.join("\n"),
        ns.count_by("labels", None, 0),
        keys(ns.query().text("first number")),
        keys(ns.query().vector(vec![1.0, 0.0])),
        keys(ns.query().text("second").filter(attr("labels").has("even") & attr("n").gte(10))),
        ns.count(None),
    )
}

#[test]
fn a_fork_holds_the_same_and_is_apart_from_then_on() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let (db, dir) = db("apart");
    let ns = db.namespace("main", schema()).unwrap();
    // Two segments, deletions in the first, and documents still in memory.
    ns.upsert_many((0..300).map(|k| doc(k, "first"))).unwrap();
    ns.flush().unwrap();
    ns.upsert_many((300..500).map(|k| doc(k, "second"))).unwrap();
    for k in (0..300).step_by(7) {
        ns.delete(k).unwrap();
    }
    ns.flush().unwrap();
    ns.upsert_many((500..560).map(|k| doc(k, "third"))).unwrap();
    ns.delete(301).unwrap();
    let before = state(&ns);

    let fork = ns.fork("sandbox").unwrap();
    assert_eq!(fork.name(), "sandbox");
    // Each is its own: an identity that stays through writes and reopening, and isn't the other's.
    let (mine, theirs) = (ns.id(), fork.id());
    assert!(mine.len() == 32 && theirs.len() == 32 && mine != theirs, "{mine} {theirs}");
    assert_eq!(state(&fork), before);
    assert_eq!(state(&ns), before);
    assert_eq!(db.namespaces().unwrap(), ["main", "sandbox"]);
    // Linked, not copied: a segment's file has two names.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let segments: Vec<PathBuf> = std::fs::read_dir(dir.join("sandbox")).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "swt")).collect();
        assert!(!segments.is_empty());
        assert!(segments.iter().all(|p| p.metadata().unwrap().nlink() == 2), "{segments:?}");
    }

    // Changes in the fork: the original holds what it held.
    fork.upsert_many((0..40).map(|k| doc(k, "changed"))).unwrap();
    for k in 400..450 {
        fork.delete(k).unwrap();
    }
    assert_eq!(state(&ns), before);
    fork.flush().unwrap();
    fork.optimize().unwrap();
    assert_eq!(state(&ns), before);
    let forked = state(&fork);
    assert_ne!(forked, before);
    assert_eq!(fork.get(3).unwrap().iter().find(|(n, _)| n == "labels").map(|(_, v)| v.clone()), Some(Value::Tags(vec!["changed".into(), "odd".into()])));

    // And changes in the original, with a merge that removes the files the fork began with.
    ns.upsert_many((560..600).map(|k| doc(k, "later"))).unwrap();
    ns.delete(0).unwrap();
    ns.flush().unwrap();
    ns.optimize().unwrap();
    let after = state(&ns);
    assert_ne!(after, before);
    assert_eq!(state(&fork), forked);

    // A namespace can be deleted once nothing has it open; the others are untouched.
    let scratch = ns.fork("scratch").unwrap();
    assert_eq!(db.delete_namespace("scratch").unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
    assert_eq!(state(&scratch), after);
    drop(scratch);
    db.delete_namespace("scratch").unwrap();
    assert_eq!(db.namespaces().unwrap(), ["main", "sandbox"]);
    assert_eq!(db.delete_namespace("scratch").unwrap_err().kind(), std::io::ErrorKind::NotFound);
    assert_eq!(db.delete_namespace("../main").unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
    assert_eq!((state(&ns), state(&fork)), (after.clone(), forked.clone()));

    // A fork of the fork, and names that can't be.
    let second = fork.fork("sandbox-2").unwrap();
    assert_eq!(state(&second), forked);
    assert_eq!(refused(ns.fork("sandbox")), std::io::ErrorKind::AlreadyExists);
    assert_eq!(refused(ns.fork("main")), std::io::ErrorKind::AlreadyExists);
    assert_eq!(refused(ns.fork("not a name")), std::io::ErrorKind::InvalidInput);
    assert_eq!(refused(ns.fork("")), std::io::ErrorKind::InvalidInput);

    // Closed and opened again, each is what it was; and the fork outlives the original.
    assert_eq!((ns.id(), fork.id()), (mine.clone(), theirs.clone()));
    assert!(second.id() != theirs && second.id() != mine);
    drop((ns, fork, second));
    let again = db.open_namespace("main").unwrap();
    assert_eq!(again.id(), mine);
    assert_eq!(state(&again), after);
    drop(again);
    std::fs::remove_dir_all(dir.join("main")).unwrap();
    std::fs::remove_dir_all(dir.join("sandbox")).unwrap();
    let second = db.open_namespace("sandbox-2").unwrap();
    assert_eq!(state(&second), forked);
    second.upsert(doc(1000, "new")).unwrap();
    assert_eq!(second.count(None), second.query().limit(100_000).run().unwrap().len() as u64);
    assert_eq!(db.namespaces().unwrap(), ["sandbox-2"]);
    drop(second);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_fork_into_another_database_and_of_an_empty_namespace() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let (db, dir) = db("into");
    let (other, other_dir) = self::db("into-other");
    let ns = db.namespace("main", schema()).unwrap();
    let empty = ns.fork("empty").unwrap();
    assert_eq!(empty.count(None), 0);
    ns.upsert_many((0..50).map(|k| doc(k, "first"))).unwrap();
    assert_eq!(empty.count(None), 0);
    let there = ns.fork_into(&other, "copy").unwrap();
    assert_eq!(state(&there), state(&ns));
    assert_eq!(other.namespaces().unwrap(), ["copy"]);
    // What a process that died while forking left is no namespace, and is in no one's way.
    std::fs::create_dir_all(dir.join(".fork-again")).unwrap();
    std::fs::write(dir.join(".fork-again").join("schema"), "left behind").unwrap();
    assert_eq!(db.namespaces().unwrap(), ["empty", "main"]);
    let again = ns.fork("again").unwrap();
    assert_eq!(state(&again), state(&ns));
    assert!(!dir.join(".fork-again").exists());
    drop((ns, empty, there, again));
    std::fs::remove_dir_all(dir).unwrap();
    std::fs::remove_dir_all(other_dir).unwrap();
}

/// Every I/O failure a fork can meet, one at a time: the fork fails, leaves no namespace and
/// nothing half made, and the original holds what it held and takes writes.
#[cfg(feature = "failpoints")]
#[test]
fn a_fork_that_fails_leaves_nothing_and_harms_nothing() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    use hinny::faults;
    let (db, dir) = db("faults");
    let ns = db.namespace("main", schema()).unwrap();
    ns.upsert_many((0..200).map(|k| doc(k, "first"))).unwrap();
    ns.flush().unwrap();
    ns.upsert_many((200..260).map(|k| doc(k, "second"))).unwrap();
    ns.delete(5).unwrap();
    ns.flush().unwrap();
    let before = state(&ns);
    // The points a fork reaches (one that succeeds says which, and how often).
    faults::reset();
    drop(ns.fork("probe").unwrap());
    let reached = faults::reset();
    std::fs::remove_dir_all(dir.join("probe")).unwrap();
    assert!(reached.contains_key("fork link") && reached.contains_key("write manifest"), "{reached:?}");
    let mut failed = 0;
    for (point, times) in &reached {
        for nth in 1..=*times {
            faults::fail_at(point, nth);
            let result = ns.fork("sandbox");
            faults::reset();
            match result {
                Err(_) => {
                    failed += 1;
                    assert_eq!(db.namespaces().unwrap(), ["main"], "{point} {nth}");
                    assert!(!dir.join("sandbox").exists() && !dir.join(".fork-sandbox").exists(), "{point} {nth}");
                }
                // (A failure after the fork was whole: opening it, say.)
                Ok(fork) => {
                    assert_eq!(state(&fork), before, "{point} {nth}");
                    drop(fork);
                    std::fs::remove_dir_all(dir.join("sandbox")).unwrap();
                }
            }
            // A failed log write stops an index until it's reopened; anything else doesn't.
            if ns.upsert(doc(9_999, "probe")).is_ok() {
                ns.delete(9_999).unwrap();
                assert_eq!(state(&ns), before, "{point} {nth}");
            }
        }
    }
    assert!(failed > 5, "{failed} of {reached:?}");
    drop(ns);
    assert_eq!(state(&db.open_namespace("main").unwrap()), before);
    std::fs::remove_dir_all(dir).unwrap();
}

/// A copy made file by file from what `files` lists is the namespace as it was, read-only, and
/// follows the original when the files that changed are copied again and it is refreshed.
#[test]
fn a_copy_of_the_listed_files_is_a_replica() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let (db, dir) = db("replica");
    let (there, there_dir) = self::db("replica-copy");
    let ns = db.namespace("main", schema()).unwrap();
    ns.upsert_many((0..200).map(|k| doc(k, "first"))).unwrap();
    ns.flush().unwrap();
    ns.upsert_many((200..260).map(|k| doc(k, "second"))).unwrap();
    for k in (0..200).step_by(9) {
        ns.delete(k).unwrap();
    }
    // The copy: the named files first, then the schema, the manifest last.
    let copy = |ns: &Namespace| {
        let listed = ns.files().unwrap();
        let to = there_dir.join("main");
        std::fs::create_dir_all(&to).unwrap();
        for (name, size) in &listed.files {
            if !to.join(name).exists() {
                std::fs::copy(ns.dir().join(name), to.join(name)).unwrap();
            }
            assert_eq!(std::fs::metadata(to.join(name)).unwrap().len(), *size, "{name}");
        }
        std::fs::write(to.join("schema"), &listed.schema).unwrap();
        std::fs::write(to.join("manifest.tmp"), &listed.manifest).unwrap();
        std::fs::rename(to.join("manifest.tmp"), to.join("manifest")).unwrap();
    };
    copy(&ns);
    let replica = there.open_read_only("main").unwrap();
    assert!(replica.read_only() && !ns.read_only());
    // A copy is the same namespace; one made anew under the name is not.
    assert_eq!((replica.id(), ns.files().unwrap().id), (ns.id(), ns.id()));
    assert_eq!(state(&replica), state(&ns));
    assert_eq!(replica.upsert(doc(999, "no")).unwrap_err().kind(), std::io::ErrorKind::PermissionDenied);
    assert!(!replica.refresh().unwrap());
    // The original moves on (writes, deletes in old segments, a merge); the replica doesn't
    // until the changed files are copied and it is refreshed.
    let before = state(&replica);
    ns.upsert_many((260..300).map(|k| doc(k, "third"))).unwrap();
    ns.delete(1).unwrap();
    ns.delete(201).unwrap();
    ns.flush().unwrap();
    assert_eq!(state(&replica), before);
    copy(&ns);
    assert_eq!(state(&replica), before);
    assert!(replica.refresh().unwrap());
    assert_eq!(state(&replica), state(&ns));
    ns.optimize().unwrap();
    ns.upsert(doc(300, "fourth")).unwrap();
    copy(&ns);
    assert!(replica.refresh().unwrap() && !replica.refresh().unwrap());
    assert_eq!(state(&replica), state(&ns));
    // Opened again later, it is what was copied last.
    drop(replica);
    assert_eq!(state(&there.open_read_only("main").unwrap()), state(&ns));
    assert_eq!(refused(there.open_read_only("nothing")), std::io::ErrorKind::NotFound);
    drop(ns);
    std::fs::remove_dir_all(dir).unwrap();
    std::fs::remove_dir_all(there_dir).unwrap();
}
