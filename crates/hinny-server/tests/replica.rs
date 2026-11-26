//! Replication end to end: a primary on an ephemeral port, replicas in other directories of the
//! same process, syncing over HTTP.

use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use axum::Router;
use axum::extract::Request;
use axum::middleware::{self, Next};
use axum::response::IntoResponse;
use hinny::{Document, Metric, Namespace, Patch, Schema, attr};
use hinny_replica::Replica;
use hinny_server::{Options, Server};
use serde_json::{Value, json};

const TOKEN: &str = "secret";

/// A directory of its own, removed when the test ends.
struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("hinny-replica-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        Dir(d)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// What the primary is asked for files, and how a test makes those requests go wrong.
#[derive(Default)]
struct Taps {
    /// File requests so far.
    files: AtomicUsize,
    /// File requests answered 404 (gone).
    gone: AtomicUsize,
    /// File requests from this many on fail with a 500 (0: none do).
    fail_from: AtomicUsize,
    /// A namespace to optimize just before the next file request: its listed files are merged
    /// away under the replica copying them.
    merge_first: Mutex<Option<String>>,
}

/// A router served on 127.0.0.1 on an ephemeral port, on a runtime of its own, until dropped.
struct Running {
    address: SocketAddr,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Running {
    fn new(app: Router) -> Running {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let thread = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
            runtime.block_on(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                axum::serve(listener, app)
                    .with_graceful_shutdown(async {
                        let _ = stopped.await;
                    })
                    .await
                    .unwrap();
            });
        });
        Running { address, stop: Some(stop), thread: Some(thread) }
    }

    fn url(&self) -> String {
        format!("http://{}", self.address)
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.stop.take().unwrap().send(());
        let _ = self.thread.take().unwrap().join();
    }
}

struct Primary {
    server: Arc<Server>,
    taps: Arc<Taps>,
    running: Running,
    _dir: Dir,
}

impl Primary {
    fn new(name: &str) -> Primary {
        let dir = Dir::new(&format!("{name}-primary"));
        let server = Server::open(&dir.0).unwrap();
        // Every listing made anew, so a sync sees every write before it; the test of the
        // interval sets its own.
        server.set_replica_interval(Duration::ZERO);
        let taps = Arc::new(Taps::default());
        let (tapped, merging) = (taps.clone(), server.clone());
        let app = hinny_server::router(server.clone(), &Options { token: Some(TOKEN.into()), ..Options::default() }).layer(middleware::from_fn(move |request: Request, next: Next| {
            let (taps, server) = (tapped.clone(), merging.clone());
            async move {
                if !request.uri().path().contains("/replica/files/") {
                    return next.run(request).await;
                }
                let n = taps.files.fetch_add(1, Ordering::SeqCst) + 1;
                let from = taps.fail_from.load(Ordering::SeqCst);
                if from > 0 && n >= from {
                    return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "made to fail").into_response();
                }
                let merge = taps.merge_first.lock().unwrap().take();
                if let Some(name) = merge {
                    tokio::task::spawn_blocking(move || server.namespace(&name).unwrap().optimize().unwrap()).await.unwrap();
                }
                let response = next.run(request).await;
                if response.status() == axum::http::StatusCode::NOT_FOUND {
                    taps.gone.fetch_add(1, Ordering::SeqCst);
                }
                response
            }
        }));
        Primary { server, taps, running: Running::new(app), _dir: dir }
    }

    fn url(&self) -> String {
        self.running.url()
    }

    fn files_asked(&self) -> usize {
        self.taps.files.load(Ordering::SeqCst)
    }
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

/// The files a namespace is copied as, in a directory.
fn copied_files(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| hinny_replica::is_file_name(n)).collect()
}

/// The files the primary's manifest names now.
fn listed_files(ns: &Namespace) -> BTreeSet<String> {
    ns.files().unwrap().files.into_iter().map(|(n, _)| n).collect()
}

/// One HTTP request over a socket of its own: the status and the body.
fn send(address: SocketAddr, method: &str, path: &str, token: Option<&str>, body: Option<&Value>) -> (u16, Vec<u8>) {
    let mut stream = TcpStream::connect(address).unwrap();
    let body = body.map(Value::to_string).unwrap_or_default();
    let authorization = token.map(|t| format!("Authorization: Bearer {t}\r\n")).unwrap_or_default();
    let request = format!("{method} {path} HTTP/1.1\r\nHost: test\r\n{authorization}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    let split = response.windows(4).position(|w| w == b"\r\n\r\n").expect("a whole response");
    let head = String::from_utf8_lossy(&response[..split]).to_lowercase();
    assert!(!head.contains("transfer-encoding: chunked"), "{head}");
    let payload = response[split + 4..].to_vec();
    if let Some(length) = head.lines().find_map(|l| l.strip_prefix("content-length: ")) {
        assert_eq!(length.trim().parse::<usize>().unwrap(), payload.len(), "{head}");
    }
    (head[9..12].parse().unwrap(), payload)
}

fn send_json(address: SocketAddr, method: &str, path: &str, token: Option<&str>, body: Option<&Value>) -> (u16, Value) {
    let (status, payload) = send(address, method, path, token, body);
    (status, serde_json::from_slice(&payload).unwrap_or_else(|_| panic!("{method} {path}: {status}, not JSON: {}", String::from_utf8_lossy(&payload))))
}

#[test]
fn a_replica_follows_the_primary_at_each_sync() {
    let primary = Primary::new("follows");
    let local = Dir::new("follows-replica");
    let replica = Replica::new(&primary.url(), Some(TOKEN), &local.0).unwrap();
    let ns = primary.server.create("docs", schema()).unwrap();
    // Two segments, deletions in the first, and documents still in memory.
    ns.upsert_many((0..200).map(|k| doc(k, "first"))).unwrap();
    ns.flush().unwrap();
    ns.upsert_many((200..260).map(|k| doc(k, "second"))).unwrap();
    for k in (0..200).step_by(9) {
        ns.delete(k).unwrap();
    }
    assert_eq!(replica.namespace("docs").err().unwrap().kind(), std::io::ErrorKind::NotFound);

    assert!(replica.sync("docs").unwrap());
    let copy = replica.namespace("docs").unwrap();
    assert!(copy.read_only());
    assert_eq!(state(&copy), state(&ns));
    assert_eq!(replica.namespaces().unwrap(), ["docs"]);
    assert_eq!(copied_files(&local.0.join("docs")), listed_files(&ns));

    // Writes, deletes and patches on the primary: not on the replica until it syncs.
    let before = state(&copy);
    ns.upsert_many((260..300).map(|k| doc(k, "third"))).unwrap();
    ns.delete(1).unwrap();
    ns.delete(201).unwrap();
    assert!(ns.patch(2, &Patch::new().int("n", -2).tags("labels", ["patched"])).unwrap());
    assert_ne!(state(&ns), before);
    assert_eq!(state(&copy), before);
    assert!(replica.sync("docs").unwrap());
    // The handle given out before sees it: it was refreshed.
    assert_eq!(state(&copy), state(&ns));
    assert_eq!(copy.count(Some(&attr("labels").has("patched"))), 1);
    assert!(copy.get(1).is_none());

    // Nothing changed: nothing downloaded.
    let asked = primary.files_asked();
    assert!(!replica.sync("docs").unwrap());
    assert!(!replica.sync("docs").unwrap());
    assert_eq!(primary.files_asked(), asked);

    // A second sync of a few new documents downloads only what is new, not the segments the
    // replica has.
    ns.upsert_many((300..310).map(|k| doc(k, "fourth"))).unwrap();
    let have = copied_files(&local.0.join("docs"));
    assert!(replica.sync("docs").unwrap());
    let now = copied_files(&local.0.join("docs"));
    assert_eq!(primary.files_asked() - asked, now.difference(&have).count());
    assert_eq!(state(&copy), state(&ns));
}

/// Vectors compared by dot product and by Euclidean distance, of lengths from 1 to 5: the replica
/// compares them as the primary does, from its segments and from what was still in memory.
#[test]
fn a_replica_compares_vectors_by_the_primarys_metric() {
    let primary = Primary::new("metric");
    let local = Dir::new("metric-replica");
    let replica = Replica::new(&primary.url(), Some(TOKEN), &local.0).unwrap();
    for metric in [Metric::Dot, Metric::Euclidean] {
        let name = metric.name();
        let ns = primary.server.create(name, Schema::new().text("body", 1.0).int("n").tags("labels").vector_with(2, metric)).unwrap();
        let long = |key: u64| {
            let d = doc(key, "first");
            let angle = key as f32 * 0.1;
            let by = 1.0 + (key % 5) as f32;
            d.vector(vec![by * angle.cos(), by * angle.sin()])
        };
        ns.upsert_many((0..200).map(long)).unwrap();
        ns.flush().unwrap();
        ns.upsert_many((200..240).map(long)).unwrap();
        assert!(replica.sync(name).unwrap());
        let copy = replica.namespace(name).unwrap();
        assert_eq!(copy.schema().metric, metric);
        assert_eq!(state(&copy), state(&ns));
        for q in [vec![1.0, 0.0], vec![0.3, 2.0], vec![-4.0, 1.0]] {
            let scores = |ns: &Namespace| ns.query().vector(q.clone()).limit(20).run().unwrap().iter().map(|h| (h.key, h.score)).collect::<Vec<_>>();
            assert_eq!(scores(&copy), scores(&ns), "{name}, {q:?}");
        }
        // Under dot product the longest vectors near the query come first, which cosine wouldn't
        // put there.
        if metric == Metric::Dot {
            let first = ns.query().vector(vec![1.0, 0.0]).limit(1).run().unwrap()[0].key;
            assert_eq!(first % 5, 4, "{first}");
        }
    }
}

#[test]
fn an_optimize_on_the_primary_removes_the_old_files_here() {
    let primary = Primary::new("optimize");
    let local = Dir::new("optimize-replica");
    let replica = Replica::new(&primary.url(), Some(TOKEN), &local.0).unwrap();
    let ns = primary.server.create("docs", schema()).unwrap();
    // Segments until the copy holds two or more: the primary's own merges may fold the first
    // few together in the background before a sync sees them.
    let mut round = 0u64;
    let old = loop {
        ns.upsert_many((round * 100..round * 100 + 100).map(|k| doc(k, "first"))).unwrap();
        ns.delete(round * 100 + 3).unwrap();
        ns.flush().unwrap();
        assert!(replica.sync("docs").unwrap());
        let copied = copied_files(&local.0.join("docs"));
        if copied.iter().filter(|n| n.ends_with(".swt")).count() >= 2 {
            break copied;
        }
        round += 1;
        assert!(round < 20, "the primary never had two segments at once");
    };

    ns.optimize().unwrap();
    assert!(replica.sync("docs").unwrap());
    let copy = replica.namespace("docs").unwrap();
    assert_eq!(state(&copy), state(&ns));
    let now = copied_files(&local.0.join("docs"));
    assert_eq!(now, listed_files(&ns));
    assert!(old.iter().any(|n| !now.contains(n)), "the optimize merged nothing away: {old:?}");
    // Nothing else is left behind: no part files, no temporary files.
    let leftovers: Vec<String> = std::fs::read_dir(local.0.join("docs")).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| !hinny_replica::is_file_name(n) && !["schema", "manifest", "replica-of"].contains(&n.as_str())).collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
    // And it opens afresh as that.
    let db = hinny::Database::open(&local.0).unwrap();
    assert_eq!(state(&db.open_read_only("docs").unwrap()), state(&ns));
}

#[test]
fn namespaces_come_and_go_with_the_primary() {
    let primary = Primary::new("come-and-go");
    let local = Dir::new("come-and-go-replica");
    let replica = Replica::new(&primary.url(), Some(TOKEN), &local.0).unwrap();
    let a = primary.server.create("a", schema()).unwrap();
    a.upsert_many((0..50).map(|k| doc(k, "first"))).unwrap();
    assert_eq!(replica.sync_all().unwrap(), ["a"]);
    assert!(replica.sync_all().unwrap().is_empty());

    // One made later appears.
    let b = primary.server.create("b", schema()).unwrap();
    b.upsert_many((0..20).map(|k| doc(k, "second"))).unwrap();
    assert_eq!(replica.sync_all().unwrap(), ["b"]);
    assert_eq!(replica.namespaces().unwrap(), ["a", "b"]);
    assert_eq!(state(&replica.namespace("b").unwrap()), state(&b));

    // One deleted on the primary disappears, though a handle to it is still held here (it
    // keeps answering from the files it has open).
    let held = replica.namespace("a").unwrap();
    drop(a);
    primary.server.delete("a").unwrap();
    assert_eq!(replica.sync_all().unwrap(), ["a"]);
    assert_eq!(replica.namespaces().unwrap(), ["b"]);
    assert!(!local.0.join("a").exists());
    assert_eq!(replica.namespace("a").err().unwrap().kind(), std::io::ErrorKind::NotFound);
    assert_eq!(held.count(None), 50);
    // So does one deleted and synced by name.
    drop(b);
    primary.server.delete("b").unwrap();
    assert!(replica.sync("b").unwrap());
    assert!(!replica.sync("b").unwrap());
    assert!(replica.namespaces().unwrap().is_empty());

    // Deleted and made again under the same name between two syncs: its files have the old
    // one's names, and none of the old ones may stay.
    let c = primary.server.create("c", schema()).unwrap();
    c.upsert_many((0..30).map(|k| doc(k, "first"))).unwrap();
    assert!(replica.sync("c").unwrap());
    drop(c);
    primary.server.delete("c").unwrap();
    let c = primary.server.create("c", schema()).unwrap();
    c.upsert_many((100..130).map(|k| doc(k, "second"))).unwrap();
    assert!(replica.sync("c").unwrap());
    assert_eq!(state(&replica.namespace("c").unwrap()), state(&c));
}

#[test]
fn a_listing_is_kept_for_the_interval_unless_the_manifest_moves_on() {
    let primary = Primary::new("interval");
    primary.server.set_replica_interval(Duration::from_secs(3600));
    let local = Dir::new("interval-replica");
    let replica = Replica::new(&primary.url(), Some(TOKEN), &local.0).unwrap();
    let ns = primary.server.create("docs", schema()).unwrap();
    ns.upsert_many((0..40).map(|k| doc(k, "first"))).unwrap();
    assert!(replica.sync("docs").unwrap());
    // Written to memory only: the listing kept from the last sync still holds, so the replica
    // doesn't see it (and the primary wrote no segment for it).
    ns.upsert_many((40..60).map(|k| doc(k, "second"))).unwrap();
    assert!(!replica.sync("docs").unwrap());
    assert_eq!(replica.namespace("docs").unwrap().count(None), 40);
    // A flush moves the manifest on, which a kept listing can't hide.
    ns.flush().unwrap();
    assert!(replica.sync("docs").unwrap());
    assert_eq!(state(&replica.namespace("docs").unwrap()), state(&ns));
}

#[test]
fn a_file_merged_away_while_copying_starts_the_sync_over() {
    let primary = Primary::new("merged-away");
    let local = Dir::new("merged-away-replica");
    let replica = Replica::new(&primary.url(), Some(TOKEN), &local.0).unwrap();
    let ns = primary.server.create("docs", schema()).unwrap();
    // Two segments or more, so the optimize has something to merge away (the primary's own
    // merges may fold the first few together in the background).
    let mut round = 0u64;
    while listed_files(&ns).iter().filter(|n| n.ends_with(".swt")).count() < 2 {
        ns.upsert_many((round * 100..round * 100 + 100).map(|k| doc(k, "first"))).unwrap();
        ns.flush().unwrap();
        round += 1;
        assert!(round < 20, "the primary never had two segments at once");
    }
    *primary.taps.merge_first.lock().unwrap() = Some("docs".into());
    assert!(replica.sync("docs").unwrap());
    assert!(primary.taps.gone.load(Ordering::SeqCst) > 0, "no file was gone: the test merged nothing away");
    assert_eq!(state(&replica.namespace("docs").unwrap()), state(&ns));
    assert_eq!(copied_files(&local.0.join("docs")), listed_files(&ns));
}

#[test]
fn an_interrupted_sync_leaves_the_state_before_it() {
    let primary = Primary::new("interrupted");
    let local = Dir::new("interrupted-replica");
    let replica = Replica::new(&primary.url(), Some(TOKEN), &local.0).unwrap();
    let ns = primary.server.create("docs", schema()).unwrap();
    ns.upsert_many((0..100).map(|k| doc(k, "first"))).unwrap();
    assert!(replica.sync("docs").unwrap());
    let before = state(&ns);

    // New segments and a merge: the next sync has several files to fetch, and the second
    // fails, as a dropped connection or a full disk would.
    for round in 1..4u64 {
        ns.upsert_many((round * 100..round * 100 + 50).map(|k| doc(k, "second"))).unwrap();
        ns.delete(round).unwrap();
        ns.flush().unwrap();
    }
    ns.optimize().unwrap();
    ns.upsert_many((900..910).map(|k| doc(k, "third"))).unwrap();
    primary.taps.fail_from.store(primary.files_asked() + 2, Ordering::SeqCst);
    let e = replica.sync("docs").unwrap_err();
    assert!(e.to_string().contains("made to fail"), "{e}");
    assert!(primary.files_asked() >= 2);
    // What was there still is, to the handle given out and to one opened afresh.
    assert_eq!(state(&replica.namespace("docs").unwrap()), before);
    let db = hinny::Database::open(&local.0).unwrap();
    assert_eq!(state(&db.open_read_only("docs").unwrap()), before);

    // A first sync that fails leaves nothing that opens.
    let fresh = primary.server.create("fresh", schema()).unwrap();
    fresh.upsert_many((0..20).map(|k| doc(k, "first"))).unwrap();
    assert!(replica.sync_all().is_err());
    assert!(!replica.namespaces().unwrap().contains(&"fresh".to_string()));
    assert_eq!(db.open_read_only("fresh").err().unwrap().kind(), std::io::ErrorKind::NotFound);

    // Once the primary answers again, both catch up, fetching only what they lack.
    primary.taps.fail_from.store(0, Ordering::SeqCst);
    let have = copied_files(&local.0.join("docs"));
    let asked = primary.files_asked();
    let changed = replica.sync_all().unwrap();
    assert_eq!(changed, ["docs", "fresh"]);
    let missing = listed_files(&ns).difference(&have).count() + listed_files(&fresh).len();
    assert_eq!(primary.files_asked() - asked, missing);
    assert_eq!(state(&replica.namespace("docs").unwrap()), state(&ns));
    assert_eq!(state(&replica.namespace("fresh").unwrap()), state(&fresh));
}

#[test]
fn the_file_route_serves_only_a_namespaces_files() {
    let primary = Primary::new("file-route");
    let ns = primary.server.create("docs", schema()).unwrap();
    ns.upsert_many((0..30).map(|k| doc(k, "first"))).unwrap();
    let files = ns.files().unwrap();
    let address = primary.running.address;
    let base = "/v1/namespaces/docs/replica";

    let (status, listing) = send_json(address, "GET", base, Some(TOKEN), None);
    assert_eq!(status, 200, "{listing}");
    assert_eq!(listing["schema"], json!(files.schema));
    assert_eq!(listing["manifest"].as_str().unwrap().as_bytes(), files.manifest);
    assert_eq!(listing["state"], json!(hinny_replica::state(&files.schema, &files.manifest)));
    let listed: Vec<(String, u64)> = listing["files"].as_array().unwrap().iter().map(|f| (f["name"].as_str().unwrap().to_string(), f["size"].as_u64().unwrap())).collect();
    assert_eq!(listed, files.files);

    // Each listed file, whole, with its length.
    for (name, size) in &listed {
        let (status, bytes) = send(address, "GET", &format!("{base}/files/{name}"), Some(TOKEN), None);
        assert_eq!((status, bytes.len() as u64), (200, *size), "{name}");
        assert_eq!(bytes, std::fs::read(ns.dir().join(name)).unwrap(), "{name}");
    }
    // Never a path, nor the schema or manifest, nor anything not of a namespace's kinds.
    for bad in ["../schema", "manifest", "schema", "seg-1.swt/../x", "..%2Fschema", "%2e%2e/schema", "seg-1.swt%2F..%2Fx", "lock", "wal-1", "seg-1.swt.part", "x/seg-1.swt"] {
        let (status, answer) = send_json(address, "GET", &format!("{base}/files/{bad}"), Some(TOKEN), None);
        assert_eq!((status, answer["error"]["code"].as_str()), (400, Some("invalid")), "{bad}: {answer}");
    }
    // A name of the right kind that isn't there is gone: a 404.
    let (status, answer) = send_json(address, "GET", &format!("{base}/files/seg-99999999.swt"), Some(TOKEN), None);
    assert_eq!((status, answer["error"]["code"].as_str()), (404, Some("not_found")), "{answer}");
    let (status, _) = send_json(address, "GET", "/v1/namespaces/nothing/replica", Some(TOKEN), None);
    assert_eq!(status, 404);

    // Both behind the token.
    for path in [base.to_string(), format!("{base}/files/{}", listed[0].0)] {
        assert_eq!(send_json(address, "GET", &path, None, None).0, 401, "{path}");
        assert_eq!(send_json(address, "GET", &path, Some("wrong"), None).0, 401, "{path}");
    }
}

#[test]
fn a_wrong_token_syncs_nothing() {
    let primary = Primary::new("token");
    let ns = primary.server.create("docs", schema()).unwrap();
    ns.upsert(doc(1, "first")).unwrap();
    for token in [Some("wrong"), None] {
        let local = Dir::new(&format!("token-replica-{}", token.is_some()));
        let replica = Replica::new(&primary.url(), token, &local.0).unwrap();
        let e = replica.sync_all().unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied, "{e}");
        let e = replica.sync("docs").unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied, "{e}");
        assert!(replica.namespaces().unwrap().is_empty());
    }
}

#[test]
fn a_replica_server_answers_reads_and_refuses_writes() {
    let primary = Primary::new("server");
    let ns = primary.server.create("docs", schema()).unwrap();
    ns.upsert_many((0..80).map(|k| doc(k, "first"))).unwrap();
    ns.delete(5).unwrap();

    let local = Dir::new("server-replica");
    let server = Server::replica_of(&local.0, &primary.url(), Some(TOKEN)).unwrap();
    assert_eq!(server.replica().unwrap().sync_all().unwrap(), ["docs"]);
    let running = Running::new(hinny_server::router(server.clone(), &Options::default()));
    let (here, there) = (running.address, primary.running.address);
    let both = |method: &str, path: &str, body: Option<Value>| {
        let mine = send_json(here, method, path, None, body.as_ref());
        let theirs = send_json(there, method, path, Some(TOKEN), body.as_ref());
        assert_eq!(mine.0, 200, "{method} {path}: {}", mine.1);
        assert_eq!(mine, theirs, "{method} {path}");
        mine.1
    };

    // The reads, answered as the primary answers them.
    assert_eq!(both("GET", "/v1/namespaces", None), json!({"namespaces": ["docs"]}));
    both("GET", "/v1/namespaces/docs/documents/7", None);
    both("POST", "/v1/namespaces/docs/query", Some(json!({"text": "first number", "limit": 10})));
    both("POST", "/v1/namespaces/docs/query", Some(json!({"vector": [1.0, 0.0], "filter": {"eq": ["labels", "even"]}, "limit": 5})));
    assert_eq!(both("POST", "/v1/namespaces/docs/count", None), json!({"count": 79}));
    both("POST", "/v1/namespaces/docs/count", Some(json!({"filter": {"range": ["n", 10, 20]}})));
    both("POST", "/v1/namespaces/docs/count_by", Some(json!({"attribute": "labels"})));
    let (status, described) = send_json(here, "GET", "/v1/namespaces/docs", None, None);
    assert_eq!(status, 200);
    let (_, theirs) = send_json(there, "GET", "/v1/namespaces/docs", Some(TOKEN), None);
    assert_eq!((&described["name"], &described["schema"]), (&theirs["name"], &theirs["schema"]));
    assert_eq!(send_json(here, "GET", "/v1/namespaces/docs/documents/5", None, None).0, 404);
    assert_eq!(send_json(here, "GET", "/v1/namespaces/nothing", None, None).0, 404);

    // Every write: refused, pointing at the primary.
    let writes: [(&str, &str, Option<Value>); 14] = [
        ("PUT", "/v1/namespaces/docs", Some(json!({"text": ["body"]}))),
        ("PUT", "/v1/namespaces/new", Some(json!({"text": ["body"]}))),
        ("DELETE", "/v1/namespaces/docs", None),
        ("POST", "/v1/namespaces/docs/apply", Some(json!([{"key": 1000, "text": {"body": "no"}}]))),
        ("DELETE", "/v1/namespaces/docs/documents/7", None),
        ("POST", "/v1/namespaces/docs/delete_where", Some(json!({"filter": {"eq": ["labels", "even"]}}))),
        ("POST", "/v1/namespaces/docs/patch", Some(json!({"key": 7, "patch": {"ints": {"n": 1}}}))),
        ("POST", "/v1/namespaces/docs/patch_where", Some(json!({"filter": {"eq": ["labels", "even"]}, "patch": {"ints": {"n": 1}}}))),
        ("POST", "/v1/namespaces/docs/upsert_if", Some(json!({"document": {"key": 7}, "if": {"eq": ["labels", "even"]}}))),
        ("POST", "/v1/namespaces/docs/delete_if", Some(json!({"key": 7, "if": {"eq": ["labels", "even"]}}))),
        ("POST", "/v1/namespaces/docs/fork", Some(json!({"name": "copy"}))),
        ("POST", "/v1/namespaces/docs/flush", None),
        ("POST", "/v1/namespaces/docs/optimize", None),
        ("POST", "/v1/namespaces/docs/sync", None),
    ];
    for (method, path, body) in writes {
        let (status, answer) = send_json(here, method, path, None, body.as_ref());
        assert_eq!((status, answer["error"]["code"].as_str()), (405, Some("read_only")), "{method} {path}: {answer}");
        assert_eq!(answer["error"]["message"], json!(format!("this is a replica of {}; write to the primary", primary.url())));
    }
    // Nothing changed here or there.
    assert_eq!(both("POST", "/v1/namespaces/docs/count", None), json!({"count": 79}));
    assert_eq!(both("GET", "/v1/namespaces", None), json!({"namespaces": ["docs"]}));
    // A replica isn't replicated from.
    assert_eq!(send_json(here, "GET", "/v1/namespaces/docs/replica", None, None).0, 405);

    // A write on the primary shows here after the next sync, through the same server.
    ns.upsert(doc(500, "second")).unwrap();
    assert_eq!(send_json(here, "POST", "/v1/namespaces/docs/count", None, None).1, json!({"count": 79}));
    assert_eq!(server.replica().unwrap().sync_all().unwrap(), ["docs"]);
    assert_eq!(both("POST", "/v1/namespaces/docs/count", None), json!({"count": 80}));
    both("GET", "/v1/namespaces/docs/documents/500", None);

    // And on a timer: the server's syncing thread.
    let _syncing = server.replica().unwrap().sync_every(Duration::from_millis(20), |e| panic!("sync failed: {e}"));
    ns.upsert(doc(501, "second")).unwrap();
    let mut waited = Duration::ZERO;
    while send_json(here, "POST", "/v1/namespaces/docs/count", None, None).1 != json!({"count": 81}) {
        assert!(waited < Duration::from_secs(10), "the timer never synced the write");
        std::thread::sleep(Duration::from_millis(20));
        waited += Duration::from_millis(20);
    }
}
