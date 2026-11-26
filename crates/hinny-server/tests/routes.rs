//! The server's routes, driven in-process through the router (no sockets), and once over a
//! real socket.

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use hinny_server::{Options, Server};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

/// A database directory of its own, removed when the test ends.
struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("hinny-server-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        Dir(d)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct App {
    router: Router,
    server: Arc<Server>,
    token: Option<String>,
}

impl App {
    fn new(dir: &Dir, options: Options) -> App {
        let server = Server::open(&dir.0).unwrap();
        App { router: hinny_server::router(server.clone(), &options), server, token: options.token }
    }

    /// Sends a request with the app's token, if it has one.
    async fn call(&self, method: Method, path: &str, body: Option<Value>) -> (StatusCode, Value) {
        let token = self.token.clone();
        self.raw(method, path, body.map(|b| b.to_string()), token.as_deref()).await
    }

    async fn raw(&self, method: Method, path: &str, body: Option<String>, token: Option<&str>) -> (StatusCode, Value) {
        let mut request = Request::builder().method(method).uri(path);
        if let Some(t) = token {
            request = request.header("authorization", format!("Bearer {t}"));
        }
        let request = request.header("content-type", "application/json").body(body.map(Body::from).unwrap_or_else(Body::empty)).unwrap();
        let response = self.router.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let value = if bytes.is_empty() { Value::Null } else { serde_json::from_slice(&bytes).unwrap_or_else(|_| panic!("{status}: not JSON: {}", String::from_utf8_lossy(&bytes))) };
        (status, value)
    }

    async fn ok(&self, method: Method, path: &str, body: Option<Value>) -> Value {
        let (status, value) = self.call(method, path, body).await;
        assert_eq!(status, StatusCode::OK, "{path}: {value}");
        value
    }

    /// Expects an error with this status and code; returns its message.
    async fn fails(&self, method: Method, path: &str, body: Option<Value>, status: StatusCode, code: &str) -> String {
        let (got, value) = self.call(method, path, body).await;
        assert_eq!((got, value["error"]["code"].as_str()), (status, Some(code)), "{path}: {value}");
        value["error"]["message"].as_str().unwrap().to_string()
    }
}

fn b64(floats: &[f32]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes: Vec<u8> = floats.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, b)| n | (*b as u32) << (16 - 8 * i));
        for i in 0..=chunk.len() {
            out.push(ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char);
        }
    }
    while out.len() % 4 != 0 {
        out.push('=');
    }
    out
}

fn schema() -> Value {
    json!({"text": [{"name": "title", "boost": 2.0}, "body"], "ints": ["year"], "strings": ["kind"], "tags": ["labels"], "vector": 3})
}

fn documents() -> Value {
    json!([
        {"key": 1, "text": {"title": "Lease", "body": "The apartment lease for 2025"}, "ints": {"year": 2025}, "strings": {"kind": "pdf"}, "tags": {"labels": ["a", "b"]}, "vector": [1.0, 0.0, 0.0]},
        {"key": "2", "text": {"title": "Tax return", "body": "Filed in March"}, "ints": {"year": 2024}, "strings": {"kind": "pdf"}, "tags": {"labels": ["b"]}, "vector_b64": b64(&[0.0, 1.0, 0.0])},
        {"key": 3, "text": {"title": "Notes", "body": "Leases and other contracts"}, "ints": {"year": 2023}, "strings": {"kind": "note"}, "tags": {"labels": ["a"]}, "vector": [0.0, 0.0, 1.0]},
    ])
}

fn keys(hits: &Value) -> Vec<u64> {
    hits["hits"].as_array().unwrap().iter().map(|h| h["key"].as_u64().unwrap()).collect()
}

#[tokio::test]
async fn every_route() {
    let dir = Dir::new("routes");
    let app = App::new(&dir, Options::default());
    assert_eq!(app.ok(Method::GET, "/healthz", None).await, json!({"ok": true}));
    assert_eq!(app.ok(Method::GET, "/v1/namespaces", None).await, json!({"namespaces": []}));

    let created = app.ok(Method::PUT, "/v1/namespaces/docs", Some(schema())).await;
    assert_eq!(created["name"], "docs");
    assert_eq!(created["schema"]["tags"], json!(["labels"]));
    assert_eq!(created["schema"]["vector"], 3);
    // The same schema again: the namespace as it is.
    assert_eq!(app.ok(Method::PUT, "/v1/namespaces/docs", Some(schema())).await, created);

    assert_eq!(app.ok(Method::POST, "/v1/namespaces/docs/apply", Some(documents())).await, json!({"changes": 3}));
    let described = app.ok(Method::GET, "/v1/namespaces/docs", None).await;
    assert_eq!(described["schema"], created["schema"]);
    assert_eq!(described["stats"]["documents"], 3);

    let doc = app.ok(Method::GET, "/v1/namespaces/docs/documents/1", None).await;
    assert_eq!(doc["attributes"]["year"], 2025);
    assert_eq!(doc["attributes"]["labels"], json!(["a", "b"]));
    assert_eq!(doc["text"]["title"], "Lease");

    // Text; a vector as numbers and as base64; a filter on tags.
    let hits = app.ok(Method::POST, "/v1/namespaces/docs/query", Some(json!({"text": "lease", "limit": 5}))).await;
    assert_eq!(keys(&hits)[0], 1);
    assert!(hits["hits"][0]["snippet"]["text"].is_string());
    let hits = app.ok(Method::POST, "/v1/namespaces/docs/query", Some(json!({"vector": [0.0, 0.9, 0.1]}))).await;
    assert_eq!(keys(&hits)[0], 2);
    let hits = app.ok(Method::POST, "/v1/namespaces/docs/query", Some(json!({"vector_b64": b64(&[0.1, 0.0, 0.9])}))).await;
    assert_eq!(keys(&hits)[0], 3);
    assert!(hits["hits"][0]["vector_score"].as_f64().unwrap() > 0.9);
    let hits = app.ok(Method::POST, "/v1/namespaces/docs/query", Some(json!({"filter": {"eq": ["labels", "a"]}, "order_by": {"attribute": "year"}}))).await;
    assert_eq!(keys(&hits), [3, 1]);

    assert_eq!(app.ok(Method::POST, "/v1/namespaces/docs/count", None).await, json!({"count": 3}));
    assert_eq!(app.ok(Method::POST, "/v1/namespaces/docs/count", Some(json!({"filter": {"eq": ["labels", "b"]}}))).await, json!({"count": 2}));
    let counts = app.ok(Method::POST, "/v1/namespaces/docs/count_by", Some(json!({"attribute": "kind"}))).await;
    assert_eq!(counts, json!({"counts": [["pdf", 2], ["note", 1]]}));
    let counts = app.ok(Method::POST, "/v1/namespaces/docs/count_by", Some(json!({"attribute": "labels", "filter": {"range": ["year", 2024, null]}, "limit": 1}))).await;
    assert_eq!(counts["counts"].as_array().unwrap().len(), 1);

    for route in ["flush", "optimize", "sync"] {
        assert_eq!(app.ok(Method::POST, &format!("/v1/namespaces/docs/{route}"), None).await, json!({"ok": true}));
    }

    assert_eq!(app.ok(Method::DELETE, "/v1/namespaces/docs/documents/2", None).await, json!({"ok": true}));
    app.fails(Method::GET, "/v1/namespaces/docs/documents/2", None, StatusCode::NOT_FOUND, "not_found").await;
    assert_eq!(app.ok(Method::POST, "/v1/namespaces/docs/delete_where", Some(json!({"filter": {"eq": ["kind", "note"]}}))).await, json!({"deleted": 1}));
    assert_eq!(app.ok(Method::POST, "/v1/namespaces/docs/count", Some(json!({}))).await, json!({"count": 1}));

    // A batch puts and deletes as one.
    let batch = json!([{"delete": 1}, {"key": 4, "text": {"body": "a new one"}, "tags": {"labels": ["solo"]}}]);
    assert_eq!(app.ok(Method::POST, "/v1/namespaces/docs/apply", Some(batch)).await, json!({"changes": 2}));
    assert_eq!(app.ok(Method::POST, "/v1/namespaces/docs/count", None).await, json!({"count": 1}));
    assert_eq!(app.ok(Method::GET, "/v1/namespaces/docs/documents/4", None).await["attributes"]["labels"], json!(["solo"]));

    assert_eq!(app.ok(Method::GET, "/v1/namespaces", None).await, json!({"namespaces": ["docs"]}));
    app.server.close_all();
}

#[tokio::test]
async fn a_fork_stays_apart_from_its_original() {
    let dir = Dir::new("fork");
    let app = App::new(&dir, Options::default());
    app.ok(Method::PUT, "/v1/namespaces/docs", Some(schema())).await;
    app.ok(Method::POST, "/v1/namespaces/docs/apply", Some(documents())).await;

    let fork = app.ok(Method::POST, "/v1/namespaces/docs/fork", Some(json!({"name": "trial"}))).await;
    assert_eq!(fork["name"], "trial");
    assert_eq!(fork["schema"], app.ok(Method::GET, "/v1/namespaces/docs", None).await["schema"]);

    // Changes on either side stay on that side.
    app.ok(Method::POST, "/v1/namespaces/trial/delete_where", Some(json!({"filter": {"eq": ["kind", "pdf"]}}))).await;
    app.ok(Method::POST, "/v1/namespaces/docs/apply", Some(json!([{"key": 9, "text": {"body": "only in the original"}}]))).await;
    assert_eq!(app.ok(Method::POST, "/v1/namespaces/trial/count", None).await, json!({"count": 1}));
    assert_eq!(app.ok(Method::POST, "/v1/namespaces/docs/count", None).await, json!({"count": 4}));
    app.fails(Method::GET, "/v1/namespaces/trial/documents/9", None, StatusCode::NOT_FOUND, "not_found").await;

    app.fails(Method::POST, "/v1/namespaces/docs/fork", Some(json!({"name": "trial"})), StatusCode::CONFLICT, "exists").await;
    assert_eq!(app.ok(Method::GET, "/v1/namespaces", None).await, json!({"namespaces": ["docs", "trial"]}));

    // Deleting the fork leaves the original whole.
    assert_eq!(app.ok(Method::DELETE, "/v1/namespaces/trial", None).await, json!({"deleted": "trial"}));
    assert_eq!(app.ok(Method::POST, "/v1/namespaces/docs/count", None).await, json!({"count": 4}));
    app.server.close_all();
}

#[tokio::test]
async fn deleting_a_namespace() {
    let dir = Dir::new("delete");
    let app = App::new(&dir, Options::default());
    app.ok(Method::PUT, "/v1/namespaces/docs", Some(schema())).await;
    app.ok(Method::POST, "/v1/namespaces/docs/apply", Some(documents())).await;
    // Another handle on it (as another process would have) keeps it from being deleted.
    app.server.close_all();
    let db = hinny::Database::open(&dir.0).unwrap();
    let held = db.open_namespace("docs").unwrap();
    app.fails(Method::DELETE, "/v1/namespaces/docs", None, StatusCode::LOCKED, "locked").await;
    drop(held);
    assert!(dir.0.join("docs").exists());
    assert_eq!(app.ok(Method::POST, "/v1/namespaces/docs/count", None).await, json!({"count": 3}));
    app.ok(Method::DELETE, "/v1/namespaces/docs", None).await;
    assert!(!dir.0.join("docs").exists());
    // Not found in the server's words, not the engine's (which names the server's directory).
    for (method, path) in [(Method::GET, "/v1/namespaces/docs"), (Method::POST, "/v1/namespaces/docs/count"), (Method::DELETE, "/v1/namespaces/docs"), (Method::POST, "/v1/namespaces/docs/patch")] {
        let body = path.ends_with("/patch").then(|| json!({"key": 1, "patch": {}}));
        assert_eq!(app.fails(method, path, body, StatusCode::NOT_FOUND, "not_found").await, "no namespace \"docs\"", "{path}");
    }
    assert_eq!(app.ok(Method::GET, "/v1/namespaces", None).await, json!({"namespaces": []}));
    // Made again, it starts empty.
    app.ok(Method::PUT, "/v1/namespaces/docs", Some(schema())).await;
    assert_eq!(app.ok(Method::POST, "/v1/namespaces/docs/count", None).await, json!({"count": 0}));
    app.server.close_all();
}

#[tokio::test]
async fn patches_and_writes_with_a_condition() {
    let dir = Dir::new("patch");
    let app = App::new(&dir, Options::default());
    app.ok(Method::PUT, "/v1/namespaces/docs", Some(schema())).await;
    app.ok(Method::POST, "/v1/namespaces/docs/apply", Some(documents())).await;
    let route = |r: &str| format!("/v1/namespaces/docs/{r}");

    // A patch that changes one attribute keeps the text, the other attributes and the vector.
    assert_eq!(app.ok(Method::POST, &route("patch"), Some(json!({"key": 1, "patch": {"strings": {"kind": "scan"}}}))).await, json!({"patched": true}));
    let doc = app.ok(Method::GET, &route("documents/1"), None).await;
    assert_eq!((doc["attributes"]["kind"].as_str(), doc["attributes"]["year"].as_i64()), (Some("scan"), Some(2025)));
    assert_eq!(doc["attributes"]["labels"], json!(["a", "b"]));
    assert_eq!((doc["text"]["title"].as_str(), doc["text"]["body"].as_str()), (Some("Lease"), Some("The apartment lease for 2025")));
    let hits = app.ok(Method::POST, &route("query"), Some(json!({"vector": [1.0, 0.0, 0.0], "limit": 1, "filter": {"eq": ["kind", "scan"]}}))).await;
    assert_eq!(keys(&hits), [1]);
    // Text, tags, a version and a vector in base64; "remove" takes parts away; a key as a string.
    let patch = json!({"key": "1", "patch": {"text": {"body": "Signed in May"}, "tags": {"labels": ["c"]}, "version": 4, "vector_b64": b64(&[0.0, 0.0, 1.0]), "remove": ["year", "title"]}});
    assert_eq!(app.ok(Method::POST, &route("patch"), Some(patch)).await, json!({"patched": true}));
    let doc = app.ok(Method::GET, &route("documents/1"), None).await;
    assert!(doc["attributes"].get("year").is_none() && doc["text"].get("title").is_none());
    assert_eq!((doc["attributes"]["labels"].clone(), doc["text"]["body"].as_str()), (json!(["c"]), Some("Signed in May")));
    assert_eq!(keys(&app.ok(Method::POST, &route("query"), Some(json!({"text": "signed"}))).await), [1]);
    // The patched vector is the document's now: against it the document scores 1, where its
    // first vector scored 0. (Document 3 has the same vector, so the two tie.)
    let hits = app.ok(Method::POST, &route("query"), Some(json!({"vector": [0.0, 0.0, 1.0], "limit": 3}))).await;
    let patched = hits["hits"].as_array().unwrap().iter().find(|h| h["key"] == 1).expect("document 1 among the hits");
    assert!(patched["vector_score"].as_f64().unwrap() > 0.99, "{patched}");
    // A key that isn't there: false, and so is a condition that doesn't hold; one that does patches.
    assert_eq!(app.ok(Method::POST, &route("patch"), Some(json!({"key": 99, "patch": {"ints": {"year": 1}}}))).await, json!({"patched": false}));
    assert_eq!(app.ok(Method::POST, &route("patch"), Some(json!({"key": 2, "patch": {"ints": {"year": 1}}, "if": {"eq": ["kind", "note"]}}))).await, json!({"patched": false}));
    assert_eq!(app.ok(Method::POST, &route("patch"), Some(json!({"key": 2, "patch": {"ints": {"year": 1}}, "if": {"eq": ["kind", "pdf"]}}))).await, json!({"patched": true}));
    assert_eq!(app.ok(Method::GET, &route("documents/2"), None).await["attributes"]["year"], 1);

    // Every document that passes a filter, and how many.
    assert_eq!(app.ok(Method::POST, &route("patch_where"), Some(json!({"filter": {"eq": ["kind", "pdf"]}, "patch": {"strings": {"kind": "doc"}}}))).await, json!({"patched": 1}));
    assert_eq!(app.ok(Method::POST, &route("patch_where"), Some(json!({"filter": {"eq": ["kind", "pdf"]}, "patch": {"strings": {"kind": "doc"}}}))).await, json!({"patched": 0}));
    assert_eq!(app.ok(Method::POST, &route("count"), Some(json!({"filter": {"eq": ["kind", "doc"]}}))).await, json!({"count": 1}));

    // upsert_if: written where there is none, refused when the condition fails, written when it holds.
    let write = |rev: i64, condition: Value| json!({"document": {"key": 10, "text": {"title": format!("revision {rev}")}, "ints": {"year": rev}, "vector": [1.0, 1.0, 0.0]}, "if": condition});
    assert_eq!(app.ok(Method::POST, &route("upsert_if"), Some(write(5, json!({"range": ["year", null, 4]})))).await, json!({"written": true}));
    assert_eq!(app.ok(Method::POST, &route("upsert_if"), Some(write(4, json!({"range": ["year", null, 3]})))).await, json!({"written": false}));
    assert_eq!(app.ok(Method::GET, &route("documents/10"), None).await["attributes"]["year"], 5);
    assert_eq!(app.ok(Method::POST, &route("upsert_if"), Some(write(6, json!({"range": ["year", null, 5]})))).await, json!({"written": true}));
    assert_eq!(app.ok(Method::GET, &route("documents/10"), None).await["text"]["title"], "revision 6");

    // delete_if, both ways.
    assert_eq!(app.ok(Method::POST, &route("delete_if"), Some(json!({"key": 10, "if": {"eq": ["year", 5]}}))).await, json!({"deleted": false}));
    app.ok(Method::GET, &route("documents/10"), None).await;
    assert_eq!(app.ok(Method::POST, &route("delete_if"), Some(json!({"key": 10, "if": {"eq": ["year", 6]}}))).await, json!({"deleted": true}));
    app.fails(Method::GET, &route("documents/10"), None, StatusCode::NOT_FOUND, "not_found").await;
    assert_eq!(app.ok(Method::POST, &route("delete_if"), Some(json!({"key": 10, "if": {"eq": ["year", 6]}}))).await, json!({"deleted": false}));

    // Mistakes: a part there isn't, a part missing, a patch that isn't one, what the engine refuses.
    let m = app.fails(Method::POST, &route("patch"), Some(json!({"key": 1, "patch": {"key": 1}})), StatusCode::BAD_REQUEST, "invalid").await;
    assert!(m.contains("no part \"key\"") && m.contains("remove"), "{m}");
    app.fails(Method::POST, &route("patch"), Some(json!({"key": 1})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, &route("patch"), Some(json!({"patch": {}})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, &route("patch"), Some(json!({"key": 1, "patch": {}, "when": {}})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, &route("patch"), Some(json!({"key": 1, "patch": [], "if": {}})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, &route("patch"), Some(json!({"key": 1, "patch": {"ints": {"month": 3}}})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, &route("patch_where"), Some(json!({"patch": {}})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, &route("patch_where"), Some(json!({"filter": {"exists": "year"}})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, &route("upsert_if"), Some(json!({"document": {"key": 1}})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, &route("upsert_if"), Some(json!({"if": {"exists": "year"}})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, &route("delete_if"), Some(json!({"key": 1})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, &route("delete_if"), Some(json!({"if": {"exists": "year"}})), StatusCode::BAD_REQUEST, "invalid").await;
    app.server.close_all();
}

#[tokio::test]
async fn a_word_being_typed_matches_the_words_that_start_with_it() {
    let dir = Dir::new("typing");
    let app = App::new(&dir, Options::default());
    app.ok(Method::PUT, "/v1/namespaces/docs", Some(schema())).await;
    app.ok(Method::POST, "/v1/namespaces/docs/apply", Some(documents())).await;
    let query = |body: Value| async { keys(&app.ok(Method::POST, "/v1/namespaces/docs/query", Some(body)).await) };
    assert_eq!(query(json!({"text": "ret"})).await, Vec::<u64>::new());
    assert_eq!(query(json!({"text": "ret", "typing": 32})).await, [2]);
    assert_eq!(query(json!({"text": "tax ret", "typing": 32, "all_words": true})).await, [2]);
    assert_eq!(query(json!({"text": "ret", "typing": 0})).await, Vec::<u64>::new());
    app.fails(Method::POST, "/v1/namespaces/docs/query", Some(json!({"text": "ret", "typing": "yes"})), StatusCode::BAD_REQUEST, "invalid").await;
    app.server.close_all();
}

#[tokio::test]
async fn a_pattern_filter_matches_the_whole_string() {
    let dir = Dir::new("glob");
    let app = App::new(&dir, Options::default());
    app.ok(Method::PUT, "/v1/namespaces/files", Some(json!({"text": ["name"], "strings": ["path"], "tags": ["labels"]}))).await;
    let files = json!([
        {"key": 1, "text": {"name": "taxes"}, "strings": {"path": "/home/mia/taxes/2025.pdf"}, "tags": {"labels": ["Tax", "inbox"]}},
        {"key": 2, "text": {"name": "taxes"}, "strings": {"path": "/home/leo/taxes/2024.PDF"}, "tags": {"labels": ["family"]}},
        {"key": 3, "text": {"name": "taxes"}, "strings": {"path": "/home/mia/photos/2025.pdf.jpg"}},
    ]);
    app.ok(Method::POST, "/v1/namespaces/files/apply", Some(files)).await;
    let app = &app;
    let query = |filter: Value| async move {
        let hits = app.ok(Method::POST, "/v1/namespaces/files/query", Some(json!({"filter": filter}))).await;
        let mut found = keys(&hits);
        found.sort_unstable();
        found
    };
    assert_eq!(query(json!({"glob": ["path", "/home/*/taxes/*.pdf"]})).await, [1]);
    assert_eq!(query(json!({"iglob": ["path", "/home/*/taxes/*.pdf"]})).await, [1, 2]);
    assert_eq!(query(json!({"glob": ["path", "/home/???/taxes/*"]})).await, [1, 2]);
    assert_eq!(query(json!({"glob": ["labels", "T?x"]})).await, [1]);
    assert_eq!(query(json!({"iglob": ["labels", "t?x"]})).await, [1]);
    assert_eq!(query(json!({"glob": ["labels", "t?x"]})).await, Vec::<u64>::new());
    let m = app.fails(Method::POST, "/v1/namespaces/files/query", Some(json!({"filter": {"glob": ["path"]}})), StatusCode::BAD_REQUEST, "invalid").await;
    assert!(m.contains("glob takes"), "{m}");
    app.server.close_all();
}

#[tokio::test]
async fn a_schema_grows_by_putting_it_again() {
    let dir = Dir::new("grow");
    let app = App::new(&dir, Options::default());
    app.ok(Method::PUT, "/v1/namespaces/docs", Some(json!({"text": ["body"]}))).await;
    app.ok(Method::POST, "/v1/namespaces/docs/apply", Some(json!([{"key": 1, "text": {"body": "kept"}}]))).await;
    let grown = app.ok(Method::PUT, "/v1/namespaces/docs", Some(json!({"text": ["body"], "ints": ["year"]}))).await;
    assert_eq!(grown["schema"]["ints"], json!(["year"]));
    app.ok(Method::POST, "/v1/namespaces/docs/apply", Some(json!([{"key": 2, "text": {"body": "new"}, "ints": {"year": 2026}}]))).await;
    assert_eq!(app.ok(Method::POST, "/v1/namespaces/docs/count", None).await, json!({"count": 2}));
    // A change the engine refuses (a text field taken away) is its message, as invalid.
    let message = app.fails(Method::PUT, "/v1/namespaces/docs", Some(json!({"text": ["title"]})), StatusCode::BAD_REQUEST, "invalid").await;
    assert!(message.contains("another schema"), "{message}");
    // And the namespace is still there, as it was.
    assert_eq!(app.ok(Method::POST, "/v1/namespaces/docs/count", None).await, json!({"count": 2}));
    app.server.close_all();
}

/// Three vectors against the query [1, 0]: by cosine the short [0.5, 0] in the query's direction
/// is first; by dot product the long [3, 1] is, then [1, 1], and the short one last, each scoring
/// its product with the query.
#[tokio::test]
async fn a_namespace_compares_vectors_by_its_metric() {
    let dir = Dir::new("metric");
    let app = App::new(&dir, Options::default());
    let dot = json!({"vector": {"dimensions": 2, "metric": "dot"}});
    let created = app.ok(Method::PUT, "/v1/namespaces/docs", Some(dot.clone())).await;
    assert_eq!(created["schema"]["vector"], json!({"dimensions": 2, "metric": "dot"}));
    // The same schema again: the namespace as it is.
    assert_eq!(app.ok(Method::PUT, "/v1/namespaces/docs", Some(dot)).await, created);
    let vectors = json!([{"key": 1, "vector": [0.5, 0.0]}, {"key": 2, "vector": [3.0, 1.0]}, {"key": 3, "vector": [1.0, 1.0]}]);
    app.ok(Method::POST, "/v1/namespaces/docs/apply", Some(vectors.clone())).await;
    assert_eq!(app.ok(Method::GET, "/v1/namespaces/docs", None).await["schema"]["vector"], json!({"dimensions": 2, "metric": "dot"}));
    let hits = app.ok(Method::POST, "/v1/namespaces/docs/query", Some(json!({"vector": [1.0, 0.0]}))).await;
    let scores: Vec<(u64, f64)> = hits["hits"].as_array().unwrap().iter().map(|h| (h["key"].as_u64().unwrap(), h["score"].as_f64().unwrap())).collect();
    assert_eq!(scores.iter().map(|s| s.0).collect::<Vec<_>>(), [2, 3, 1], "{hits}");
    for ((_, got), want) in scores.iter().zip([3.0, 1.0, 0.5]) {
        assert!((got - want).abs() < 1e-4, "{hits}");
    }
    // Another metric for it: refused, and the namespace is still there, as it was.
    let m = app.fails(Method::PUT, "/v1/namespaces/docs", Some(json!({"vector": {"dimensions": 2, "metric": "euclidean"}})), StatusCode::BAD_REQUEST, "invalid").await;
    assert!(m.contains("another schema"), "{m}");
    let m = app.fails(Method::PUT, "/v1/namespaces/docs", Some(json!({"vector": 2})), StatusCode::BAD_REQUEST, "invalid").await;
    assert!(m.contains("another schema"), "{m}");
    assert_eq!(keys(&app.ok(Method::POST, "/v1/namespaces/docs/query", Some(json!({"vector": [1.0, 0.0]}))).await), [2, 3, 1]);
    // Cosine, written as the number it always was, ranks the short one first.
    let created = app.ok(Method::PUT, "/v1/namespaces/plain", Some(json!({"vector": {"dimensions": 2, "metric": "cosine"}}))).await;
    assert_eq!(created["schema"]["vector"], 2);
    app.ok(Method::POST, "/v1/namespaces/plain/apply", Some(vectors)).await;
    assert_eq!(keys(&app.ok(Method::POST, "/v1/namespaces/plain/query", Some(json!({"vector": [1.0, 0.0]}))).await), [1, 2, 3]);
    let m = app.fails(Method::PUT, "/v1/namespaces/other", Some(json!({"vector": {"dimensions": 2, "metric": "manhattan"}})), StatusCode::BAD_REQUEST, "invalid").await;
    assert!(m.contains("cosine, euclidean, dot"), "{m}");
    app.server.close_all();
}

#[tokio::test]
async fn a_patch_gives_sparse_terms_again() {
    let dir = Dir::new("sparse-patch");
    let app = App::new(&dir, Options::default());
    app.ok(Method::PUT, "/v1/namespaces/docs", Some(json!({"text": ["body"], "sparse": ["terms"], "ints": ["year"]}))).await;
    app.ok(Method::POST, "/v1/namespaces/docs/apply", Some(json!([{"key": 1, "text": {"body": "The lease"}, "sparse": {"terms": {"lease": 1.4}}, "ints": {"year": 2025}}]))).await;
    let route = |r: &str| format!("/v1/namespaces/docs/{r}");
    let found = || async { keys(&app.ok(Method::POST, &route("query"), Some(json!({"sparse": {"terms": {"lease": 1.0}}}))).await) };
    assert_eq!(found().await, [1]);
    let m = app.fails(Method::POST, &route("patch"), Some(json!({"key": 1, "patch": {"ints": {"year": 2026}}})), StatusCode::BAD_REQUEST, "invalid").await;
    assert!(m.contains("sparse"), "{m}");
    assert_eq!(app.ok(Method::POST, &route("patch"), Some(json!({"key": 1, "patch": {"ints": {"year": 2026}, "sparse": {"terms": {"lease": 1.4}}}}))).await, json!({"patched": true}));
    assert_eq!(app.ok(Method::GET, "/v1/namespaces/docs/documents/1", None).await["attributes"]["year"], 2026);
    assert_eq!(found().await, [1]);
    let body = json!({"filter": {"exists": "year"}, "patch": {"ints": {"year": 2027}, "sparse": {"terms": {"lease": 1.4}}}});
    assert_eq!(app.ok(Method::POST, &route("patch_where"), Some(body)).await, json!({"patched": 1}));
    assert_eq!(found().await, [1]);
    assert_eq!(app.ok(Method::POST, &route("patch"), Some(json!({"key": 1, "patch": {"ints": {"year": 2028}, "remove": ["terms"]}}))).await, json!({"patched": true}));
    assert_eq!(found().await, Vec::<u64>::new());
    app.server.close_all();
}

#[tokio::test]
async fn errors() {
    let dir = Dir::new("errors");
    let app = App::new(&dir, Options { token: None, body_limit: 4096 });
    app.ok(Method::PUT, "/v1/namespaces/docs", Some(schema())).await;

    let bad_json = app.raw(Method::POST, "/v1/namespaces/docs/apply", Some("[{\"key\": 1,".into()), None).await;
    assert_eq!((bad_json.0, bad_json.1["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("invalid")));
    assert!(bad_json.1["error"]["message"].as_str().unwrap().contains("isn't JSON"));

    let m = app.fails(Method::PUT, "/v1/namespaces/other", Some(json!({"txt": []})), StatusCode::BAD_REQUEST, "invalid").await;
    assert!(m.contains("no part \"txt\""), "{m}");
    app.fails(Method::PUT, "/v1/namespaces/other", None, StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::GET, "/v1/namespaces/nothing", None, StatusCode::NOT_FOUND, "not_found").await;
    app.fails(Method::POST, "/v1/namespaces/nothing/query", Some(json!({"text": "x"})), StatusCode::NOT_FOUND, "not_found").await;
    app.fails(Method::GET, "/v1/namespaces/.deleted-x", None, StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::DELETE, "/v1/namespaces/%2E%2E", None, StatusCode::BAD_REQUEST, "invalid").await;
    // The engine's InvalidInput: an attribute the schema doesn't have, a vector of the wrong length.
    let m = app.fails(Method::POST, "/v1/namespaces/docs/apply", Some(json!([{"key": 1, "ints": {"month": 3}}])), StatusCode::BAD_REQUEST, "invalid").await;
    assert!(m.contains("month"), "{m}");
    app.fails(Method::POST, "/v1/namespaces/docs/query", Some(json!({"vector": [1.0]})), StatusCode::BAD_REQUEST, "invalid").await;
    let m = app.fails(Method::POST, "/v1/namespaces/docs/query", Some(json!({"vector": [1.0, 0.0, 0.0], "vector_b64": b64(&[1.0, 0.0, 0.0])})), StatusCode::BAD_REQUEST, "invalid").await;
    assert!(m.contains("both"), "{m}");
    app.fails(Method::POST, "/v1/namespaces/docs/query", Some(json!({"vector_b64": "not base64!"})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, "/v1/namespaces/docs/query", Some(json!(["text"])), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, "/v1/namespaces/docs/query", Some(json!({"filter": {"near": ["year", 1]}})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, "/v1/namespaces/docs/delete_where", None, StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, "/v1/namespaces/docs/count", Some(json!({"filtre": {}})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, "/v1/namespaces/docs/count_by", Some(json!({})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::POST, "/v1/namespaces/docs/fork", Some(json!({})), StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::GET, "/v1/namespaces/docs/documents/abc", None, StatusCode::BAD_REQUEST, "invalid").await;
    app.fails(Method::GET, "/v1/nothing", None, StatusCode::NOT_FOUND, "not_found").await;

    // Larger than the limit.
    let big = json!([{"key": 1, "text": {"body": "word ".repeat(2000)}}]);
    app.fails(Method::POST, "/v1/namespaces/docs/apply", Some(big), StatusCode::PAYLOAD_TOO_LARGE, "too_large").await;

    // Another handle on the namespace's directory (as another process would have): locked.
    app.server.close_all();
    let db = hinny::Database::open(&dir.0).unwrap();
    let held = db.open_namespace("docs").unwrap();
    app.fails(Method::POST, "/v1/namespaces/docs/count", None, StatusCode::LOCKED, "locked").await;
    drop(held);
    assert_eq!(app.ok(Method::POST, "/v1/namespaces/docs/count", None).await, json!({"count": 0}));

    // A query past its deadline. (A deadline of zero is past before the search starts.)
    app.ok(Method::POST, "/v1/namespaces/docs/apply", Some(documents())).await;
    let (status, value) = app.call(Method::POST, "/v1/namespaces/docs/query", Some(json!({"text": "lease", "timeout_ms": 0}))).await;
    if status != StatusCode::OK {
        assert_eq!((status, value["error"]["code"].as_str()), (StatusCode::GATEWAY_TIMEOUT, Some("timeout")), "{value}");
    }
    app.server.close_all();
}

#[tokio::test]
async fn a_token_guards_everything_but_healthz() {
    let dir = Dir::new("auth");
    let app = App::new(&dir, Options { token: Some("s3cret".into()), body_limit: 1 << 20 });
    assert_eq!(app.raw(Method::GET, "/healthz", None, None).await.0, StatusCode::OK);
    let (status, value) = app.raw(Method::GET, "/v1/namespaces", None, None).await;
    assert_eq!((status, value["error"]["code"].as_str()), (StatusCode::UNAUTHORIZED, Some("unauthorized")));
    let (status, _) = app.raw(Method::GET, "/v1/namespaces", None, Some("s3cre")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, _) = app.raw(Method::PUT, "/v1/namespaces/docs", Some(schema().to_string()), Some("wrong!")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(!dir.0.join("docs").exists(), "a refused request did nothing");
    assert_eq!(app.raw(Method::GET, "/v1/namespaces", None, Some("s3cret")).await, (StatusCode::OK, json!({"namespaces": []})));
    app.ok(Method::PUT, "/v1/namespaces/docs", Some(schema())).await;
}

/// Over a real socket on an ephemeral port: a write, a query, and a shutdown that closes the
/// namespace so another handle can open it with the write in it.
#[tokio::test]
async fn over_a_socket() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let dir = Dir::new("socket");
    let server = Server::open(&dir.0).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let options = Options { token: Some("t".into()), ..Options::default() };
    let serving = tokio::spawn(async move {
        hinny_server::serve(listener, server, &options, async {
            let _ = stopped.await;
        })
        .await
    });

    async fn send(address: std::net::SocketAddr, method: &str, path: &str, body: &str) -> (u16, Value) {
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        let request = format!("{method} {path} HTTP/1.1\r\nHost: test\r\nAuthorization: Bearer t\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).await.unwrap();
        let status = response[9..12].parse().unwrap();
        let (_, payload) = response.split_once("\r\n\r\n").unwrap();
        (status, serde_json::from_str(payload).unwrap())
    }

    assert_eq!(send(address, "PUT", "/v1/namespaces/docs", &json!({"text": ["body"]}).to_string()).await.0, 200);
    assert_eq!(send(address, "POST", "/v1/namespaces/docs/apply", &json!([{"key": 7, "text": {"body": "over the wire"}}]).to_string()).await, (200, json!({"changes": 1})));
    let (status, hits) = send(address, "POST", "/v1/namespaces/docs/query", &json!({"text": "wire"}).to_string()).await;
    assert_eq!((status, keys(&hits)), (200, vec![7]));

    stop.send(()).unwrap();
    serving.await.unwrap().unwrap();
    // The server let go of the namespace, and the write is there.
    let db = hinny::Database::open(&dir.0).unwrap();
    assert_eq!(db.open_namespace("docs").unwrap().count(None), 1);
}
