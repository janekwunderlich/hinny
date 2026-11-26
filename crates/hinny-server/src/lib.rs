//! A hinny database directory, served over HTTP: the routes `router` builds, in the JSON the C
//! interface speaks (`hinny_json`). The first step toward a hosted hinny; one process owns the
//! directory and takes every write.
//!
//! Replicas copy it: the primary lists each namespace's files (`/replica`) and serves them one
//! by one (`/replica/files/{name}`); a server made with `Server::replica_of` keeps copies of
//! them with `hinny_replica` and answers the reading routes from those, refusing the writing ones.
//!
//! Every namespace is opened on first use and kept open: the engine allows one writer per
//! namespace directory, so the server holds one handle each and every request shares it.
//! Queries never wait for writes (the engine's views), so one handle serves both. Engine calls
//! block (disk, a mutex for writes), so they run on tokio's blocking threads, never on the
//! threads that drive connections.

use std::collections::HashMap;
use std::io::{self, ErrorKind};
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};
use std::time::{Duration, Instant, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, Path as Params, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{MethodRouter, any, get, post};
use axum::{Json, Router};
use hinny::{Database, Namespace};
use hinny_replica::Replica;
use serde_json::{Value, json};
use tokio::io::{AsyncRead, ReadBuf};

/// The default request body limit: a batch of a hundred thousand documents of 384 dimensions
/// in base64 fits, with room for their text.
pub const DEFAULT_BODY_LIMIT: usize = 256 << 20;

/// How long deleting or reopening a namespace waits for the requests using it to finish.
const DRAIN: Duration = Duration::from_secs(10);

/// How long a namespace's listing for replicas is handed out again before it is made anew
/// (`Server::set_replica_interval`).
pub const DEFAULT_REPLICA_INTERVAL: Duration = Duration::from_secs(1);

// MARK: Errors

/// An error as the client sees it: {"error": {"code", "message"}} with an HTTP status.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> ApiError {
        ApiError { status, code, message: message.into() }
    }

    /// What the client sent was wrong: JSON that isn't, a part a document can't have.
    pub fn invalid(message: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::BAD_REQUEST, "invalid", message)
    }

    pub fn not_found(message: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::NOT_FOUND, "not_found", message)
    }
}

/// The engine's errors by kind; the message is the engine's own.
impl From<io::Error> for ApiError {
    fn from(e: io::Error) -> ApiError {
        let (status, code) = match e.kind() {
            ErrorKind::InvalidInput => (StatusCode::BAD_REQUEST, "invalid"),
            ErrorKind::NotFound => (StatusCode::NOT_FOUND, "not_found"),
            ErrorKind::AlreadyExists => (StatusCode::CONFLICT, "exists"),
            ErrorKind::WouldBlock => (StatusCode::LOCKED, "locked"),
            ErrorKind::TimedOut => (StatusCode::GATEWAY_TIMEOUT, "timeout"),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "io"),
        };
        ApiError::new(status, code, e.to_string())
    }
}

/// `hinny_json`'s errors are about what the client sent.
impl From<String> for ApiError {
    fn from(message: String) -> ApiError {
        ApiError::invalid(message)
    }
}

impl From<&str> for ApiError {
    fn from(message: &str) -> ApiError {
        ApiError::invalid(message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({"error": {"code": self.code, "message": self.message}}))).into_response()
    }
}

type Answer = Result<Json<Value>, ApiError>;

// MARK: The open namespaces

/// One namespace's place in the server: its handle once opened. Requests lock it only to clone
/// the handle; deleting and reopening hold it throughout, so no request opens a second handle
/// (which the engine would refuse) or finds one half closed.
type Slot = Arc<Mutex<Option<Arc<Namespace>>>>;

/// A namespace's listing for replicas, kept a while: making one writes what the namespace holds
/// in memory out as a segment, which a replica asking every second must not cause every second.
struct Listing {
    made: Instant,
    /// The manifest it lists: one that has moved on since (a merge) makes it stale at once.
    manifest: Vec<u8>,
    answer: Arc<Value>,
}

/// The database directory and the namespaces open in it.
pub struct Server {
    db: Database,
    slots: Mutex<HashMap<String, Slot>>,
    /// The last listing of each namespace, for replicas; each behind its own lock, so replicas
    /// asking at once make it once.
    listings: Mutex<HashMap<String, Arc<Mutex<Option<Listing>>>>>,
    /// How long a listing is handed out again, in milliseconds.
    listing_ms: AtomicU64,
    /// Set when this server is a replica: it answers reads from these copies, and no writes.
    replica: Option<Arc<Replica>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic while holding one leaves a map or an Option, both still whole.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Server {
    /// Serves the database in `dir` (created if there is none).
    pub fn open(dir: impl AsRef<Path>) -> io::Result<Arc<Server>> {
        Server::new(Database::open(dir)?, None)
    }

    /// A replica of the server at `primary_url`, kept in `dir`: it serves the reading routes
    /// from its copies and refuses the writing ones. Nothing is copied until its replica syncs
    /// (`Server::replica`, `Replica::sync_all`, `Replica::sync_every`).
    pub fn replica_of(dir: impl AsRef<Path>, primary_url: &str, token: Option<&str>) -> io::Result<Arc<Server>> {
        let replica = Replica::new(primary_url, token, &dir)?;
        Server::new(Database::open(dir)?, Some(Arc::new(replica)))
    }

    fn new(db: Database, replica: Option<Arc<Replica>>) -> io::Result<Arc<Server>> {
        Ok(Arc::new(Server { db, slots: Mutex::default(), listings: Mutex::default(), listing_ms: AtomicU64::new(DEFAULT_REPLICA_INTERVAL.as_millis() as u64), replica }))
    }

    /// The replica this server answers from, if it is one.
    pub fn replica(&self) -> Option<&Arc<Replica>> {
        self.replica.as_ref()
    }

    /// How long a namespace's listing is handed out to replicas again before it is made anew
    /// (and what the namespace holds in memory written out as a segment with it). Replicas see
    /// writes up to this much later than they would otherwise.
    pub fn set_replica_interval(&self, interval: Duration) {
        self.listing_ms.store(interval.as_millis() as u64, Ordering::Relaxed);
    }

    fn slot(&self, name: &str) -> Result<Slot, ApiError> {
        if !hinny::valid_name(name) {
            return Err(ApiError::invalid(format!("{name:?} isn't a namespace name: names are letters, digits, - and _")));
        }
        Ok(lock(&self.slots).entry(name.to_string()).or_default().clone())
    }

    /// Drops the slot of a name that turned out to have no namespace, so names asked for and
    /// not found don't pile up. Only when no request holds the slot: one that does may be
    /// about to open the namespace into it.
    fn forget_if_empty(&self, name: &str) {
        let mut slots = lock(&self.slots);
        if let Some(slot) = slots.get(name) {
            if Arc::strong_count(slot) == 1 && lock(slot).is_none() {
                slots.remove(name);
            }
        }
    }

    /// The namespace, opened as it is if it isn't open yet (`not_found` if there is none).
    pub fn namespace(&self, name: &str) -> Result<Arc<Namespace>, ApiError> {
        if let Some(replica) = &self.replica {
            if !hinny::valid_name(name) {
                return Err(ApiError::invalid(format!("{name:?} isn't a namespace name: names are letters, digits, - and _")));
            }
            return replica.namespace(name).map_err(|e| self.about(name, e));
        }
        let slot = self.slot(name)?;
        let opened = {
            let mut held = lock(&slot);
            match &*held {
                Some(ns) => Ok(ns.clone()),
                None => self.db.open_namespace(name).map(|ns| held.insert(Arc::new(ns)).clone()),
            }
        };
        if opened.is_err() {
            drop(slot);
            self.forget_if_empty(name);
        }
        opened.map_err(|e| self.about(name, e))
    }

    /// Opens or creates a namespace with a schema, as `Database::namespace` does. An open one
    /// whose schema is the same is left as it is; one asked for with another schema is closed
    /// and opened again with it, which is how the engine grows a schema.
    pub fn create(&self, name: &str, schema: hinny::Schema) -> Result<Arc<Namespace>, ApiError> {
        let slot = self.slot(name)?;
        let mut held = lock(&slot);
        if let Some(ns) = &*held {
            // Compared as JSON: what a client can say, not how the schema was built.
            if hinny_json::schema_json(ns.schema()) == hinny_json::schema_json(&schema) {
                return Ok(ns.clone());
            }
            close(&mut held, name)?;
            self.forget_listing(name);
        }
        let ns = Arc::new(self.db.namespace(name, schema)?);
        *held = Some(ns.clone());
        Ok(ns)
    }

    /// Forks `from` into a new namespace `name`, open from then on.
    pub fn fork(&self, from: &str, name: &str) -> Result<Arc<Namespace>, ApiError> {
        let source = self.namespace(from)?;
        let slot = self.slot(name)?;
        let mut held = lock(&slot);
        if held.is_some() {
            return Err(ApiError::new(StatusCode::CONFLICT, "exists", format!("there is a namespace {name:?} already")));
        }
        let fork = Arc::new(source.fork(name)?);
        *held = Some(fork.clone());
        Ok(fork)
    }

    /// Closes a namespace and deletes it (`Database::delete_namespace`).
    pub fn delete(&self, name: &str) -> Result<(), ApiError> {
        let slot = self.slot(name)?;
        let mut held = lock(&slot);
        close(&mut held, name)?;
        self.forget_listing(name);
        let deleted = self.db.delete_namespace(name);
        // Its place among the open namespaces goes with it, or a name deleted is a name kept.
        drop(held);
        drop(slot);
        self.forget_if_empty(name);
        deleted.map_err(|e| self.about(name, e))
    }

    /// An engine error for the namespace `name`. That there is no such namespace is said in
    /// the server's words, since the engine's names the server's directory. A namespace that
    /// is there and lacks something it needs (a file its manifest names) is damaged, not
    /// missing: that stays an error of the server's, with the engine's message.
    fn about(&self, name: &str, e: io::Error) -> ApiError {
        match e.kind() {
            ErrorKind::NotFound if !self.names().is_ok_and(|names| names.iter().any(|n| n == name)) => ApiError::not_found(format!("no namespace {name:?}")),
            ErrorKind::NotFound => ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "io", e.to_string()),
            _ => e.into(),
        }
    }

    /// The database's namespaces, open or not.
    pub fn names(&self) -> io::Result<Vec<String>> {
        match &self.replica {
            Some(replica) => replica.namespaces(),
            None => self.db.namespaces(),
        }
    }

    /// What a replica copies of a namespace: `{"state", "id", "schema", "manifest", "files":
    /// [{"name", "size"}]}`. The same answer as last time if that is younger than the replica
    /// interval and the manifest hasn't moved on since.
    pub fn listing(&self, name: &str) -> Result<Arc<Value>, ApiError> {
        if let Some(replica) = &self.replica {
            return Err(ApiError::new(StatusCode::METHOD_NOT_ALLOWED, "read_only", format!("this is a replica of {}; replicate from the primary", replica.primary())));
        }
        let ns = self.namespace(name)?;
        let cell = lock(&self.listings).entry(name.to_string()).or_default().clone();
        let mut held = lock(&cell);
        let fresh_for = Duration::from_millis(self.listing_ms.load(Ordering::Relaxed));
        if let Some(last) = &*held
            && last.made.elapsed() < fresh_for
            && std::fs::read(ns.dir().join("manifest")).is_ok_and(|m| m == last.manifest)
        {
            return Ok(last.answer.clone());
        }
        let files = ns.files()?;
        let manifest = String::from_utf8(files.manifest.clone()).map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "io", format!("namespace {name:?}'s manifest isn't text; this server can't list it for replicas")))?;
        let answer = Arc::new(json!({
            "state": hinny_replica::state(&files.schema, &files.manifest),
            "id": namespace_id(&ns),
            "schema": files.schema,
            "manifest": manifest,
            "files": files.files.iter().map(|(n, size)| json!({"name": n, "size": size})).collect::<Vec<_>>(),
        }));
        *held = Some(Listing { made: Instant::now(), manifest: files.manifest, answer: answer.clone() });
        Ok(answer)
    }

    fn forget_listing(&self, name: &str) {
        lock(&self.listings).remove(name);
    }

    /// Closes every namespace (their `Drop` checkpoints what's in memory): at shutdown.
    pub fn close_all(&self) {
        let slots: Vec<Slot> = lock(&self.slots).drain().map(|(_, s)| s).collect();
        for slot in slots {
            drop(lock(&slot).take());
        }
    }
}

/// Takes the handle out of a held slot and drops it once the requests using it are done, so
/// that it really closes (its `Drop` checkpoints and lets go of the directory). `locked` if they
/// aren't done in time; the handle stays.
fn close(held: &mut Option<Arc<Namespace>>, name: &str) -> Result<(), ApiError> {
    let Some(ns) = held.take() else { return Ok(()) };
    let start = Instant::now();
    // No new request can clone it while the slot is held; the ones that have finish.
    while Arc::strong_count(&ns) > 1 {
        if start.elapsed() > DRAIN {
            let users = Arc::strong_count(&ns) - 1;
            *held = Some(ns);
            return Err(ApiError::new(StatusCode::LOCKED, "locked", format!("namespace {name:?} is in use by {users} requests that didn't finish within {}s", DRAIN.as_secs())));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    drop(ns);
    Ok(())
}

/// Which namespace of this name a namespace is. One deleted and made again is another, so a
/// replica holding the old one's files (which have the new one's names) knows not to keep
/// them.
fn namespace_id(ns: &Namespace) -> String {
    // The engine's own (kept in the manifest since index format 7). One written before has
    // none until it is next saved: the directory's creation time stands in, where the
    // filesystem keeps one.
    let id = ns.id();
    if !id.is_empty() {
        return id;
    }
    std::fs::metadata(ns.dir()).and_then(|m| m.created()).ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| format!("{:x}", d.as_nanos())).unwrap_or_default()
}

/// Runs an engine call on a blocking thread.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, ApiError> + Send + 'static) -> Result<T, ApiError> {
    match tokio::task::spawn_blocking(f).await {
        Ok(result) => result,
        Err(e) => Err(ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", format!("the request failed inside the server: {e}"))),
    }
}

// MARK: Requests

/// A request's body as JSON. An empty body is `null`, for the routes whose body is optional.
fn body(bytes: Result<Bytes, BytesRejection>) -> Result<Value, ApiError> {
    let bytes = bytes.map_err(|r| match r.status() {
        StatusCode::PAYLOAD_TOO_LARGE => ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "too_large", format!("the request body is larger than the server's limit ({})", r.body_text())),
        status => ApiError::new(status, "invalid", r.body_text()),
    })?;
    if bytes.iter().all(u8::is_ascii_whitespace) {
        return Ok(Value::Null);
    }
    serde_json::from_slice(&bytes).map_err(|e| ApiError::invalid(format!("the request body isn't JSON: {e}")))
}

/// A body that must be a JSON object; an empty one is `{}`.
fn fields(bytes: Result<Bytes, BytesRejection>, what: &str) -> Result<serde_json::Map<String, Value>, ApiError> {
    match body(bytes)? {
        Value::Null => Ok(serde_json::Map::new()),
        Value::Object(o) => Ok(o),
        other => Err(ApiError::invalid(format!("{what} must be a JSON object, not {}", kind(&other)))),
    }
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

fn optional_filter(parts: &serde_json::Map<String, Value>) -> Result<Option<hinny::Filter>, ApiError> {
    Ok(parts.get("filter").filter(|f| !f.is_null()).map(hinny_json::filter).transpose()?)
}

fn described(ns: &Namespace) -> Value {
    json!({"name": ns.name(), "schema": hinny_json::schema_json(ns.schema())})
}

async fn healthz() -> Json<Value> {
    Json(json!({"ok": true}))
}

async fn list(State(server): State<Arc<Server>>) -> Answer {
    let names = blocking(move || Ok(server.names()?)).await?;
    Ok(Json(json!({"namespaces": names})))
}

async fn open(State(server): State<Arc<Server>>, Params(name): Params<String>, bytes: Result<Bytes, BytesRejection>) -> Answer {
    let schema = match body(bytes)? {
        Value::Null => return Err(ApiError::invalid("PUT a namespace with its schema as the body, {} for none")),
        value => hinny_json::schema_from(&value)?,
    };
    blocking(move || Ok(Json(described(&*server.create(&name, schema)?)))).await
}

async fn describe(State(server): State<Arc<Server>>, Params(name): Params<String>) -> Answer {
    blocking(move || {
        let ns = server.namespace(&name)?;
        let mut out = described(&ns);
        out["stats"] = hinny_json::stats(&ns.stats());
        Ok(Json(out))
    })
    .await
}

async fn remove(State(server): State<Arc<Server>>, Params(name): Params<String>) -> Answer {
    blocking(move || {
        server.delete(&name)?;
        Ok(Json(json!({"deleted": name})))
    })
    .await
}

async fn apply(State(server): State<Arc<Server>>, Params(name): Params<String>, bytes: Result<Bytes, BytesRejection>) -> Answer {
    let value = body(bytes)?;
    blocking(move || {
        // Parsed on the blocking thread too: a large batch takes a while to read.
        let changes = hinny_json::changes(&value)?;
        let n = changes.len();
        server.namespace(&name)?.apply(changes)?;
        Ok(Json(json!({"changes": n})))
    })
    .await
}

fn document_key(key: &str) -> Result<u64, ApiError> {
    Ok(hinny_json::key(&Value::String(key.to_string()))?)
}

async fn get_document(State(server): State<Arc<Server>>, Params((name, key)): Params<(String, String)>) -> Answer {
    let key = document_key(&key)?;
    blocking(move || {
        // One snapshot for both, so they're of the same version of the document.
        let snapshot = server.namespace(&name)?.snapshot();
        let attrs = snapshot.get(key).ok_or_else(|| ApiError::not_found(format!("no document {key} in namespace {name:?}")))?;
        let text = snapshot.text(key).unwrap_or_default();
        Ok(Json(hinny_json::stored(&attrs, &text)))
    })
    .await
}

async fn delete_document(State(server): State<Arc<Server>>, Params((name, key)): Params<(String, String)>) -> Answer {
    let key = document_key(&key)?;
    blocking(move || {
        server.namespace(&name)?.delete(key)?;
        Ok(Json(json!({"ok": true})))
    })
    .await
}

async fn query(State(server): State<Arc<Server>>, Params(name): Params<String>, bytes: Result<Bytes, BytesRejection>) -> Answer {
    let mut parts = fields(bytes, "the query")?;
    blocking(move || {
        let vector = hinny_json::take_vector(&mut parts, "the query")?;
        let ns = server.namespace(&name)?;
        let hits = hinny_json::query(&ns, &Value::Object(parts), vector)?.run()?;
        Ok(Json(json!({"hits": hinny_json::hits(&hits)})))
    })
    .await
}

/// Checks a body's parts against the ones a route takes.
fn only(parts: &serde_json::Map<String, Value>, allowed: &[&str], what: &str) -> Result<(), ApiError> {
    match parts.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(k) => Err(ApiError::invalid(format!("{what} has no part {k:?} ({})", allowed.join(", ")))),
        None => Ok(()),
    }
}

async fn count(State(server): State<Arc<Server>>, Params(name): Params<String>, bytes: Result<Bytes, BytesRejection>) -> Answer {
    let parts = fields(bytes, "a count")?;
    only(&parts, &["filter"], "a count")?;
    let filter = optional_filter(&parts)?;
    blocking(move || Ok(Json(json!({"count": server.namespace(&name)?.count(filter.as_ref())})))).await
}

async fn count_by(State(server): State<Arc<Server>>, Params(name): Params<String>, bytes: Result<Bytes, BytesRejection>) -> Answer {
    let parts = fields(bytes, "a count_by")?;
    only(&parts, &["attribute", "filter", "limit"], "a count_by")?;
    let attribute = parts.get("attribute").and_then(Value::as_str).ok_or("a count_by needs an attribute: {\"attribute\": name}")?.to_string();
    let limit = match parts.get("limit").filter(|l| !l.is_null()) {
        Some(l) => l.as_u64().ok_or("a count_by's limit must be a non-negative integer (0 for every value)")? as usize,
        None => 0,
    };
    let filter = optional_filter(&parts)?;
    blocking(move || {
        let counts = server.namespace(&name)?.count_by(&attribute, filter.as_ref(), limit);
        Ok(Json(json!({"counts": hinny_json::counts(&counts)})))
    })
    .await
}

async fn delete_where(State(server): State<Arc<Server>>, Params(name): Params<String>, bytes: Result<Bytes, BytesRejection>) -> Answer {
    let parts = fields(bytes, "a delete_where")?;
    only(&parts, &["filter"], "a delete_where")?;
    // Required: a missing filter deleting everything would be one typo from losing the data.
    let filter = optional_filter(&parts)?.ok_or("a delete_where needs a filter: {\"filter\": {..}}")?;
    blocking(move || Ok(Json(json!({"deleted": server.namespace(&name)?.delete_where(&filter)?})))).await
}

/// A part of a body that the route can't do without.
fn required<'a>(parts: &'a serde_json::Map<String, Value>, name: &str, what: &str, shape: &str) -> Result<&'a Value, ApiError> {
    parts.get(name).filter(|v| !v.is_null()).ok_or_else(|| ApiError::invalid(format!("{what} needs a {name}: {shape}")))
}

async fn patch(State(server): State<Arc<Server>>, Params(name): Params<String>, bytes: Result<Bytes, BytesRejection>) -> Answer {
    let parts = fields(bytes, "a patch")?;
    only(&parts, &["key", "patch", "if"], "a patch")?;
    let key = hinny_json::key(required(&parts, "key", "a patch", "{\"key\": k, \"patch\": {..}}")?)?;
    let patch = hinny_json::patch(required(&parts, "patch", "a patch", "{\"key\": k, \"patch\": {..}}")?, None)?;
    let condition = parts.get("if").filter(|c| !c.is_null()).map(hinny_json::filter).transpose()?;
    blocking(move || {
        let ns = server.namespace(&name)?;
        let patched = match &condition {
            Some(condition) => ns.patch_if(key, &patch, condition)?,
            None => ns.patch(key, &patch)?,
        };
        Ok(Json(json!({"patched": patched})))
    })
    .await
}

async fn patch_where(State(server): State<Arc<Server>>, Params(name): Params<String>, bytes: Result<Bytes, BytesRejection>) -> Answer {
    let parts = fields(bytes, "a patch_where")?;
    only(&parts, &["filter", "patch"], "a patch_where")?;
    // Required, as for delete_where.
    let filter = hinny_json::filter(required(&parts, "filter", "a patch_where", "{\"filter\": {..}, \"patch\": {..}}")?)?;
    let patch = hinny_json::patch(required(&parts, "patch", "a patch_where", "{\"filter\": {..}, \"patch\": {..}}")?, None)?;
    blocking(move || Ok(Json(json!({"patched": server.namespace(&name)?.patch_where(&filter, &patch)?})))).await
}

async fn upsert_if(State(server): State<Arc<Server>>, Params(name): Params<String>, bytes: Result<Bytes, BytesRejection>) -> Answer {
    let parts = fields(bytes, "an upsert_if")?;
    only(&parts, &["document", "if"], "an upsert_if")?;
    let document = hinny_json::document(required(&parts, "document", "an upsert_if", "{\"document\": {..}, \"if\": {..}}")?, None)?;
    let condition = hinny_json::filter(required(&parts, "if", "an upsert_if", "{\"document\": {..}, \"if\": {..}}")?)?;
    blocking(move || Ok(Json(json!({"written": server.namespace(&name)?.upsert_if(document, &condition)?})))).await
}

async fn delete_if(State(server): State<Arc<Server>>, Params(name): Params<String>, bytes: Result<Bytes, BytesRejection>) -> Answer {
    let parts = fields(bytes, "a delete_if")?;
    only(&parts, &["key", "if"], "a delete_if")?;
    let key = hinny_json::key(required(&parts, "key", "a delete_if", "{\"key\": k, \"if\": {..}}")?)?;
    let condition = hinny_json::filter(required(&parts, "if", "a delete_if", "{\"key\": k, \"if\": {..}}")?)?;
    blocking(move || Ok(Json(json!({"deleted": server.namespace(&name)?.delete_if(key, &condition)?})))).await
}

async fn fork(State(server): State<Arc<Server>>, Params(name): Params<String>, bytes: Result<Bytes, BytesRejection>) -> Answer {
    let parts = fields(bytes, "a fork")?;
    only(&parts, &["name"], "a fork")?;
    let to = parts.get("name").and_then(Value::as_str).ok_or("a fork needs the new namespace's name: {\"name\": ..}")?.to_string();
    blocking(move || Ok(Json(described(&*server.fork(&name, &to)?)))).await
}

/// One of the namespace's maintenance calls.
async fn maintain(server: Arc<Server>, name: String, f: fn(&Namespace) -> io::Result<()>) -> Answer {
    blocking(move || {
        f(&*server.namespace(&name)?)?;
        Ok(Json(json!({"ok": true})))
    })
    .await
}

async fn flush(State(server): State<Arc<Server>>, Params(name): Params<String>) -> Answer {
    maintain(server, name, Namespace::flush).await
}

async fn optimize(State(server): State<Arc<Server>>, Params(name): Params<String>) -> Answer {
    maintain(server, name, Namespace::optimize).await
}

async fn sync(State(server): State<Arc<Server>>, Params(name): Params<String>) -> Answer {
    maintain(server, name, Namespace::sync).await
}

// MARK: Replication

async fn replica_listing(State(server): State<Arc<Server>>, Params(name): Params<String>) -> Answer {
    blocking(move || Ok(Json(Value::clone(&*server.listing(&name)?)))).await
}

/// One file of a namespace, streamed. Only the names of the files a namespace is copied as:
/// never a path, never its schema or manifest (those come with the listing, of one moment).
async fn replica_file(State(server): State<Arc<Server>>, Params((name, file)): Params<(String, String)>) -> Result<Response, ApiError> {
    if !hinny_replica::is_file_name(&file) {
        return Err(ApiError::invalid(format!("{file:?} isn't a file a replica copies: seg-<n>.swt, seg-<n>.del, seg-<n>.<m>.del or blob-<n>.fzt")));
    }
    let path = blocking(move || {
        let ns = server.servable(&name)?;
        Ok(ns.dir().join(&file))
    })
    .await?;
    // Opened before its size is read: the open file stays whole even if a merge removes its
    // name meanwhile.
    let opened = tokio::fs::File::open(&path).await;
    let gone = || ApiError::not_found(format!("no file {:?} in this namespace: merged away since it was listed; ask for the listing again", path.file_name().unwrap_or_default()));
    let file = match opened {
        Ok(f) => f,
        Err(e) if e.kind() == ErrorKind::NotFound => return Err(gone()),
        Err(e) => return Err(e.into()),
    };
    let size = file.metadata().await?.len();
    let body = FileBody { file, left: size, buffer: vec![0; 256 << 10].into_boxed_slice() };
    Ok(([(header::CONTENT_TYPE, "application/octet-stream")], Body::new(body)).into_response())
}

impl Server {
    /// The namespace, for serving its files to a replica (a replica's own copies aren't served).
    fn servable(&self, name: &str) -> Result<Arc<Namespace>, ApiError> {
        if let Some(replica) = &self.replica {
            return Err(ApiError::new(StatusCode::METHOD_NOT_ALLOWED, "read_only", format!("this is a replica of {}; replicate from the primary", replica.primary())));
        }
        self.namespace(name)
    }
}

/// A file's bytes as a response body, read as the connection takes them: a segment can be
/// larger than is worth holding in memory. Its exact size is the response's Content-Length.
struct FileBody {
    file: tokio::fs::File,
    left: u64,
    buffer: Box<[u8]>,
}

impl http_body::Body for FileBody {
    type Data = Bytes;
    type Error = io::Error;

    fn poll_frame(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<http_body::Frame<Bytes>, io::Error>>> {
        let this = self.get_mut();
        if this.left == 0 {
            return Poll::Ready(None);
        }
        let want = this.buffer.len().min(this.left.min(usize::MAX as u64) as usize);
        let mut read = ReadBuf::new(&mut this.buffer[..want]);
        match Pin::new(&mut this.file).poll_read(cx, &mut read) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
            Poll::Ready(Ok(())) if read.filled().is_empty() => Poll::Ready(Some(Err(io::Error::new(ErrorKind::UnexpectedEof, format!("the file ended {} bytes before its size", this.left))))),
            Poll::Ready(Ok(())) => {
                this.left -= read.filled().len() as u64;
                Poll::Ready(Some(Ok(http_body::Frame::data(Bytes::copy_from_slice(read.filled())))))
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        self.left == 0
    }

    fn size_hint(&self) -> http_body::SizeHint {
        http_body::SizeHint::with_exact(self.left)
    }
}

/// Every writing route, on a replica.
async fn read_only(State(server): State<Arc<Server>>) -> ApiError {
    let primary = server.replica.as_ref().map_or("its primary", |r| r.primary());
    ApiError::new(StatusCode::METHOD_NOT_ALLOWED, "read_only", format!("this is a replica of {primary}; write to the primary"))
}

async fn unknown() -> ApiError {
    ApiError::not_found("no such route (see the server's README for the routes)")
}

// MARK: Authorization

/// Compares in time that depends on the lengths only, not on where they first differ.
fn same(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        diff |= (a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0)) as usize;
    }
    std::hint::black_box(diff) == 0
}

async fn authorize(State(token): State<Arc<str>>, request: Request, next: Next) -> Response {
    let given = request.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer "));
    match given {
        Some(given) if same(given.as_bytes(), token.as_bytes()) => next.run(request).await,
        Some(_) => ApiError::new(StatusCode::UNAUTHORIZED, "unauthorized", "the token is wrong").into_response(),
        None => ApiError::new(StatusCode::UNAUTHORIZED, "unauthorized", "this server needs a token: Authorization: Bearer <token>").into_response(),
    }
}

// MARK: The router

/// How the server answers.
#[derive(Clone, Debug)]
pub struct Options {
    /// Every request but /healthz must carry it as `Authorization: Bearer <token>`.
    pub token: Option<String>,
    /// The largest request body accepted, in bytes.
    pub body_limit: usize,
}

impl Default for Options {
    fn default() -> Options {
        Options { token: None, body_limit: DEFAULT_BODY_LIMIT }
    }
}

/// Every route, for `server`.
pub fn router(server: Arc<Server>, options: &Options) -> Router {
    let ns = "/v1/namespaces/{ns}";
    // On a replica every write is refused, whatever the route: the same answer for all.
    let replica = server.replica.is_some();
    let write = |route: MethodRouter<Arc<Server>>| if replica { any(read_only) } else { route };
    let mut v1 = Router::new()
        .route("/v1/namespaces", get(list))
        .route(ns, if replica { get(describe).put(read_only).delete(read_only) } else { get(describe).put(open).delete(remove) })
        .route(&format!("{ns}/documents/{{key}}"), if replica { get(get_document).delete(read_only) } else { get(get_document).delete(delete_document) })
        .route(&format!("{ns}/query"), post(query))
        .route(&format!("{ns}/count"), post(count))
        .route(&format!("{ns}/count_by"), post(count_by))
        .route(&format!("{ns}/replica"), get(replica_listing))
        .route(&format!("{ns}/replica/files/{{*file}}"), get(replica_file));
    let writes: [(&str, MethodRouter<Arc<Server>>); 10] = [
        ("apply", post(apply)),
        ("delete_where", post(delete_where)),
        ("patch", post(patch)),
        ("patch_where", post(patch_where)),
        ("upsert_if", post(upsert_if)),
        ("delete_if", post(delete_if)),
        ("fork", post(fork)),
        ("flush", post(flush)),
        ("optimize", post(optimize)),
        ("sync", post(sync)),
    ];
    for (path, route) in writes {
        v1 = v1.route(&format!("{ns}/{path}"), write(route));
    }
    let mut v1 = v1.with_state(server);
    if let Some(token) = &options.token {
        v1 = v1.route_layer(middleware::from_fn_with_state(Arc::<str>::from(token.as_str()), authorize));
    }
    Router::new().route("/healthz", get(healthz)).merge(v1).fallback(unknown).layer(DefaultBodyLimit::max(options.body_limit))
}

/// Serves until `shutdown` resolves, then lets the requests in flight finish and closes every
/// namespace.
pub async fn serve(listener: tokio::net::TcpListener, server: Arc<Server>, options: &Options, shutdown: impl Future<Output = ()> + Send + 'static) -> io::Result<()> {
    let app = router(server.clone(), options);
    let served = axum::serve(listener, app).with_graceful_shutdown(shutdown).await;
    // Closed on a blocking thread: each checkpoints what it holds in memory.
    let closing = tokio::task::spawn_blocking(move || server.close_all()).await;
    served?;
    closing.map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_compare_whole() {
        assert!(same(b"secret", b"secret"));
        assert!(!same(b"secret", b"secreT"));
        assert!(!same(b"secret", b"secret-and-more"));
        assert!(!same(b"", b"x"));
        assert!(same(b"", b""));
    }

    #[test]
    fn names_that_could_leave_the_directory_are_refused() {
        for bad in ["", "..", ".", "a/b", "a b", ".deleted-x"] {
            assert!(!hinny::valid_name(bad), "{bad:?}");
        }
        assert!(hinny::valid_name("docs_2025-a"));
    }
}
