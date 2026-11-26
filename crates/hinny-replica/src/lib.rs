//! A replica of a `hinny-server`: a directory holding copies of the primary's namespaces, kept
//! current over HTTP, and read locally through `hinny::Database::open_read_only`. Reads cost what
//! they cost on the primary's own disk; writes go to the primary, and reach here at the next sync.
//!
//! A namespace is files the engine never writes again once a manifest names them (segments,
//! their deletion files, files of stored text), a `schema` and a `manifest`. The primary lists
//! them (`GET /v1/namespaces/{ns}/replica`); a sync downloads the ones that aren't here yet, then
//! writes the schema, then the manifest, last and atomically. A directory is what its manifest
//! says, so a sync that stops anywhere (a crash, a failed download) leaves the state before it,
//! or nothing if it was the first.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use hinny::{Database, Namespace};
use serde_json::Value;

/// How many listings a sync tries before giving up: each one that names a file the primary has
/// merged away by the time it is asked for is replaced by a new one.
const ATTEMPTS: usize = 5;

/// The file in a copied namespace's directory that says which namespace of the primary it is a
/// copy of (`id` in the listing). Not a name the engine uses.
const ORIGIN: &str = "replica-of";

// MARK: What the primary and the replica agree on

/// Whether `name` is one of the files a namespace is copied as: `seg-<digits>.swt`,
/// `seg-<digits>.del`, `seg-<digits>.<digits>.del` or `blob-<digits>.fzt`. Nothing else is
/// served or written, so no name can reach outside a namespace's directory or replace its
/// schema or manifest.
pub fn is_file_name(name: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if let Some(id) = name.strip_prefix("blob-").and_then(|r| r.strip_suffix(".fzt")) {
        return digits(id);
    }
    let Some(rest) = name.strip_prefix("seg-") else { return false };
    if let Some(id) = rest.strip_suffix(".swt") {
        return digits(id);
    }
    match rest.strip_suffix(".del").map(|r| r.split_once('.')) {
        Some(Some((id, number))) => digits(id) && digits(number),
        Some(None) => digits(&rest[..rest.len() - 4]),
        None => false,
    }
}

/// A namespace's state as the primary lists it: a hash of its schema and manifest, in hex. The
/// manifest changes with every change the namespace's files hold, so two copies of one state
/// have the same hash.
pub fn state(schema: &str, manifest: &[u8]) -> String {
    // FNV-1a, 128 bits: enough that two states differing by chance never hash alike, and no
    // dependency for it. The schema's length first, so no split of the bytes is another's.
    const PRIME: u128 = 0x0000000001000000000000000000013B;
    let mut h: u128 = 0x6c62272e07bb014262b821756295c58d;
    for b in (schema.len() as u64).to_le_bytes().iter().chain(schema.as_bytes()).chain(manifest) {
        h ^= *b as u128;
        h = h.wrapping_mul(PRIME);
    }
    format!("{h:032x}")
}

/// What the primary lists for a namespace.
struct Listing {
    state: String,
    /// Which namespace of the primary this is: a namespace deleted and made again under the
    /// same name is another, whose files have the same names as the old one's.
    id: String,
    schema: String,
    manifest: String,
    files: Vec<(String, u64)>,
}

impl Listing {
    fn from_json(name: &str, v: &Value) -> io::Result<Listing> {
        let bad = |what: &str| io::Error::new(ErrorKind::InvalidData, format!("the primary's listing of {name:?} has no {what}"));
        let text = |key: &str| v.get(key).and_then(Value::as_str).map(str::to_string).ok_or_else(|| bad(&format!("{key:?} string")));
        let files = v.get("files").and_then(Value::as_array).ok_or_else(|| bad("\"files\" array"))?;
        let files = files
            .iter()
            .map(|f| match (f.get("name").and_then(Value::as_str), f.get("size").and_then(Value::as_u64)) {
                (Some(n), Some(s)) => Ok((n.to_string(), s)),
                _ => Err(bad("files of {\"name\", \"size\"}")),
            })
            .collect::<io::Result<Vec<_>>>()?;
        let listing = Listing { state: text("state")?, id: v.get("id").and_then(Value::as_str).unwrap_or_default().to_string(), schema: text("schema")?, manifest: text("manifest")?, files };
        if state(&listing.schema, listing.manifest.as_bytes()) != listing.state {
            return Err(io::Error::new(ErrorKind::InvalidData, format!("the primary's listing of {name:?} has a state that isn't its schema's and manifest's: garbled on the way, or another server")));
        }
        // Checked here, before anything is written: a primary (or something posing as one)
        // naming "../x" or "manifest" must not get to write it.
        if let Some((bad, _)) = listing.files.iter().find(|(n, _)| !is_file_name(n)) {
            return Err(io::Error::new(ErrorKind::InvalidData, format!("the primary's listing of {name:?} names {bad:?}, which isn't a file a namespace is made of")));
        }
        Ok(listing)
    }
}

// MARK: The replica

/// A directory of copies of a primary's namespaces.
pub struct Replica {
    primary: String,
    authorization: Option<String>,
    agent: ureq::Agent,
    db: Database,
    dir: PathBuf,
    /// The read-only handles given out, refreshed by the syncs that change their namespace.
    open: Mutex<HashMap<String, Arc<Namespace>>>,
    /// One sync at a time: two writing one directory would remove each other's files.
    syncing: Mutex<()>,
}

/// What a sync of one listing did.
enum Applied {
    Unchanged,
    Changed { schema_changed: bool },
    /// A file it names was gone on the primary: merged away since. Ask again.
    Moved(String),
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic while holding one leaves a map or nothing, both still whole.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Replica {
    /// A replica of the `hinny-server` at `primary_url` (`http://host:port`), kept in `dir`
    /// (created if there is none). `token` is the primary's, if it has one. Nothing is fetched
    /// until a sync.
    pub fn new(primary_url: &str, token: Option<&str>, dir: impl AsRef<Path>) -> io::Result<Replica> {
        let primary = primary_url.trim_end_matches('/').to_string();
        if !(primary.starts_with("http://") || primary.starts_with("https://")) {
            return Err(io::Error::new(ErrorKind::InvalidInput, format!("the primary's URL must start with http:// or https://, not {primary_url:?}")));
        }
        let config = ureq::Agent::config_builder()
            // Its error answers are read for their message rather than turned into errors.
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(10)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            // Per read, not in all: a large file may take long, a stalled primary shouldn't hang
            // the sync for ever.
            .timeout_per_read(Some(Duration::from_secs(60)))
            .build();
        Ok(Replica {
            primary,
            authorization: token.map(|t| format!("Bearer {t}")),
            agent: config.into(),
            db: Database::open(&dir)?,
            dir: dir.as_ref().to_path_buf(),
            open: Mutex::default(),
            syncing: Mutex::default(),
        })
    }

    /// The primary's URL, as given (without a trailing `/`).
    pub fn primary(&self) -> &str {
        &self.primary
    }

    /// The directory the copies are in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The namespaces copied here (as of their last sync).
    pub fn namespaces(&self) -> io::Result<Vec<String>> {
        self.db.namespaces()
    }

    /// The local, read-only handle of a namespace copied here: `NotFound` if it hasn't been
    /// synced yet (or is gone on the primary). The same handle until the schema changes; each
    /// sync that changes the namespace refreshes it, so queries from then on see the new state.
    pub fn namespace(&self, name: &str) -> io::Result<Arc<Namespace>> {
        let mut open = lock(&self.open);
        if let Some(ns) = open.get(name) {
            return Ok(ns.clone());
        }
        let ns = Arc::new(self.db.open_read_only(name)?);
        open.insert(name.to_string(), ns.clone());
        Ok(ns)
    }

    /// Brings the local copy of `name` to the primary's state of it now: whether anything
    /// changed. A namespace that is gone on the primary is removed here.
    pub fn sync(&self, name: &str) -> io::Result<bool> {
        if !hinny::valid_name(name) {
            return Err(io::Error::new(ErrorKind::InvalidInput, format!("{name:?} isn't a namespace name: names are letters, digits, - and _")));
        }
        let _one = lock(&self.syncing);
        self.sync_held(name)
    }

    /// Syncs every namespace the primary has, and removes the local copies of those it has no
    /// more: the names of the ones that changed. One namespace failing doesn't stop the others;
    /// the error names every one that failed.
    pub fn sync_all(&self) -> io::Result<Vec<String>> {
        let _one = lock(&self.syncing);
        let names = self.primary_names()?;
        let mut changed = Vec::new();
        let mut failed = Vec::new();
        for name in &names {
            match self.sync_held(name) {
                Ok(true) => changed.push(name.clone()),
                Ok(false) => {}
                Err(e) => failed.push(format!("{name}: {e}")),
            }
        }
        // Copies of namespaces the primary has no more, finished or begun.
        for entry in fs::read_dir(&self.dir)?.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if hinny::valid_name(&name) && !names.contains(&name) && entry.path().join(ORIGIN).exists() {
                match self.remove(&name) {
                    Ok(true) => changed.push(name),
                    Ok(false) => {}
                    Err(e) => failed.push(format!("{name}: {e}")),
                }
            }
        }
        if !failed.is_empty() {
            return Err(io::Error::other(format!("syncing from {} failed for {} of {} namespaces: {}", self.primary, failed.len(), names.len(), failed.join("; "))));
        }
        Ok(changed)
    }

    /// Syncs every `every` on a thread of its own, until the returned `Syncing` is dropped.
    /// `report` hears every failed sync (the next one tries again).
    pub fn sync_every(self: &Arc<Self>, every: Duration, mut report: impl FnMut(&io::Error) + Send + 'static) -> Syncing {
        let (stop, stopped) = mpsc::channel::<()>();
        let replica = self.clone();
        let thread = std::thread::Builder::new()
            .name("hinny-replica-sync".into())
            .spawn(move || {
                while let Err(RecvTimeoutError::Timeout) = stopped.recv_timeout(every) {
                    if let Err(e) = replica.sync_all() {
                        report(&e);
                    }
                }
            })
            .expect("a thread for syncing");
        Syncing { stop: Some(stop), thread: Some(thread) }
    }

    fn sync_held(&self, name: &str) -> io::Result<bool> {
        let mut moved = Vec::new();
        for attempt in 0..ATTEMPTS {
            if attempt > 0 {
                // The primary keeps a listing for a moment; a merge changes the manifest, which
                // makes it list again, but give it the moment.
                std::thread::sleep(Duration::from_millis(50 << attempt));
            }
            let Some(listing) = self.listing(name)? else { return self.gone(name) };
            match self.apply(name, &listing)? {
                Applied::Unchanged => return Ok(false),
                Applied::Changed { schema_changed } => {
                    self.reopen(name, schema_changed)?;
                    return Ok(true);
                }
                Applied::Moved(file) => moved.push(file),
            }
        }
        Err(io::Error::other(format!("namespace {name:?} on {} kept changing while being copied: {ATTEMPTS} listings named files that were gone when asked for ({})", self.primary, moved.join(", "))))
    }

    /// Writes what `listing` lists into the local copy, the manifest last.
    fn apply(&self, name: &str, listing: &Listing) -> io::Result<Applied> {
        let dir = self.dir.join(name);
        let origin = fs::read_to_string(dir.join(ORIGIN)).ok();
        if origin.is_none() && dir.join("schema").exists() {
            // Not a copy this replica began: a namespace of its own, which a sync would mix
            // with the primary's files.
            return Err(io::Error::new(ErrorKind::AlreadyExists, format!("{} holds a namespace that isn't a copy from a primary; a replica's directory holds only copies", dir.display())));
        }
        if origin.as_ref().is_some_and(|o| *o != listing.id) {
            // Another namespace of that name on the primary than the one copied here: its files
            // have the old one's names, so none of them can be kept.
            self.remove(name)?;
        }
        let local_schema = fs::read_to_string(dir.join("schema")).ok();
        if let (Some(schema), Ok(manifest)) = (&local_schema, fs::read(dir.join("manifest")))
            && state(schema, &manifest) == listing.state
            && origin.is_some()
        {
            return Ok(Applied::Unchanged);
        }
        create_dir_synced(&self.dir, &dir)?;
        if origin.as_deref() != Some(listing.id.as_str()) {
            write_atomic(&dir.join(ORIGIN), listing.id.as_bytes())?;
        }
        for (file, size) in &listing.files {
            let path = dir.join(file);
            // Already here whole: none of these files changes once written, and each comes in
            // under its name only once complete.
            if fs::metadata(&path).is_ok_and(|m| m.len() == *size) {
                continue;
            }
            if !self.download(name, file, *size, &path)? {
                return Ok(Applied::Moved(file.clone()));
            }
        }
        // The files' names, on disk, before a manifest names them.
        sync_dir(&dir)?;
        let schema_changed = local_schema.as_deref() != Some(listing.schema.as_str());
        if schema_changed {
            write_atomic(&dir.join("schema"), listing.schema.as_bytes())?;
        }
        write_atomic(&dir.join("manifest"), listing.manifest.as_bytes())?;
        // What the new manifest doesn't name: merged away, or deletions saved again since.
        // Handles still reading the old state keep their files open, so this doesn't pull
        // anything from under a query.
        let named: HashSet<&str> = listing.files.iter().map(|(n, _)| n.as_str()).collect();
        for entry in fs::read_dir(&dir)?.flatten() {
            let file = entry.file_name().to_string_lossy().into_owned();
            if (is_file_name(&file) && !named.contains(file.as_str())) || file.ends_with(".part") {
                fs::remove_file(entry.path())?;
            }
        }
        Ok(Applied::Changed { schema_changed })
    }

    /// Downloads one file to `path`, through `<path>.part`: false if the primary has it no
    /// more.
    fn download(&self, name: &str, file: &str, size: u64, path: &Path) -> io::Result<bool> {
        let url = format!("{}/v1/namespaces/{name}/replica/files/{file}", self.primary);
        let mut response = self.get(&url)?;
        match response.status().as_u16() {
            200 => {}
            404 => return Ok(false),
            _ => return Err(self.refused(&url, &mut response)),
        }
        let part = path.with_file_name(format!("{file}.part"));
        let mut out = File::create(&part)?;
        // At most a byte more than listed, so a primary sending too much is caught without
        // reading all of it.
        let mut body = response.body_mut().with_config().limit(u64::MAX).reader().take(size + 1);
        let got = io::copy(&mut body, &mut out).map_err(|e| io::Error::new(e.kind(), format!("downloading {url}: {e}")))?;
        if got != size {
            drop(out);
            let _ = fs::remove_file(&part);
            return Err(io::Error::new(ErrorKind::InvalidData, format!("{url} sent {}{got} bytes; the listing says {size}", if got > size { "more than " } else { "" })));
        }
        out.flush()?;
        out.sync_all()?;
        drop(out);
        fs::rename(&part, path)?;
        Ok(true)
    }

    /// The handle given out for `name`, brought to what was just synced.
    fn reopen(&self, name: &str, schema_changed: bool) -> io::Result<()> {
        let mut open = lock(&self.open);
        let Some(ns) = open.get(name) else { return Ok(()) };
        if schema_changed {
            // A handle keeps the schema it was opened with; a new one has the new schema.
            // Queries on the old one finish on it.
            let fresh = Arc::new(self.db.open_read_only(name)?);
            open.insert(name.to_string(), fresh);
        } else {
            ns.refresh()?;
        }
        Ok(())
    }

    /// The primary has no namespace `name` (its listing was a 404): removes the local copy,
    /// once the primary's list of namespaces agrees. Whether there was one.
    fn gone(&self, name: &str) -> io::Result<bool> {
        // A 404 from the listing alone could be a primary without the replica routes (an older
        // server), which mustn't make the replica delete what it has.
        if self.primary_names()?.iter().any(|n| n == name) {
            return Err(io::Error::other(format!("the primary at {} has namespace {name:?} but no listing of it: is it a hinny-server without replication?", self.primary)));
        }
        self.remove(name)
    }

    /// Removes the local copy of `name`, whole or begun: whether there was one.
    fn remove(&self, name: &str) -> io::Result<bool> {
        // Handles given out keep reading what they have (their files stay open); new ones
        // find nothing.
        lock(&self.open).remove(name);
        let dir = self.dir.join(name);
        if dir.join("schema").exists() {
            self.db.delete_namespace(name)?;
            return Ok(true);
        }
        // A first sync that didn't finish: not a namespace yet, so not the engine's to delete.
        match fs::remove_dir_all(&dir) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e),
        }
    }

    // MARK: Talking to the primary

    fn get(&self, url: &str) -> io::Result<ureq::http::Response<ureq::Body>> {
        let mut request = self.agent.get(url);
        if let Some(a) = &self.authorization {
            request = request.header("Authorization", a);
        }
        request.call().map_err(|e| match e {
            ureq::Error::Io(e) => io::Error::new(e.kind(), format!("can't reach the primary at {url}: {e}")),
            e => io::Error::other(format!("can't reach the primary at {url}: {e}")),
        })
    }

    /// An error answer as an error: the primary's own message, with the status.
    fn refused(&self, url: &str, response: &mut ureq::http::Response<ureq::Body>) -> io::Error {
        let status = response.status().as_u16();
        let text = response.body_mut().with_config().limit(64 << 10).read_to_string().unwrap_or_default();
        let message = serde_json::from_str::<Value>(&text).ok().and_then(|v| v["error"]["message"].as_str().map(str::to_string)).unwrap_or(text);
        let kind = match status {
            401 | 403 => ErrorKind::PermissionDenied,
            404 => ErrorKind::NotFound,
            _ => ErrorKind::Other,
        };
        io::Error::new(kind, format!("the primary answered {url} with {status}: {message}"))
    }

    fn get_json(&self, url: &str) -> io::Result<Option<Value>> {
        let mut response = self.get(url)?;
        match response.status().as_u16() {
            200 => {}
            404 => return Ok(None),
            _ => return Err(self.refused(url, &mut response)),
        }
        // A listing is the schema and manifest and a line per file: far below this.
        let text = response.body_mut().with_config().limit(256 << 20).read_to_string().map_err(|e| io::Error::other(format!("reading {url}: {e}")))?;
        serde_json::from_str(&text).map(Some).map_err(|e| io::Error::new(ErrorKind::InvalidData, format!("{url} answered with something that isn't JSON: {e}")))
    }

    /// The primary's listing of `name`, or None if it has no such namespace.
    fn listing(&self, name: &str) -> io::Result<Option<Listing>> {
        let url = format!("{}/v1/namespaces/{name}/replica", self.primary);
        self.get_json(&url)?.map(|v| Listing::from_json(name, &v)).transpose()
    }

    /// The primary's namespaces.
    fn primary_names(&self) -> io::Result<Vec<String>> {
        let url = format!("{}/v1/namespaces", self.primary);
        let v = self.get_json(&url)?.ok_or_else(|| io::Error::new(ErrorKind::NotFound, format!("{url} isn't there: is {} a hinny-server?", self.primary)))?;
        let names = v.get("namespaces").and_then(Value::as_array).ok_or_else(|| io::Error::new(ErrorKind::InvalidData, format!("{url} answered without a \"namespaces\" array")))?;
        // Only names a namespace can have: anything else would be a path.
        Ok(names.iter().filter_map(Value::as_str).filter(|n| hinny::valid_name(n)).map(str::to_string).collect())
    }
}

/// The thread `Replica::sync_every` started; dropping this stops it, after the sync it is in.
pub struct Syncing {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for Syncing {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

// MARK: Files

/// Writes `path` whole or not at all: through `<path>.tmp`, synced, renamed.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let mut f = File::create(&tmp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp, path)?;
    sync_dir(path.parent().unwrap_or(Path::new(".")))
}

fn create_dir_synced(parent: &Path, dir: &Path) -> io::Result<()> {
    if !dir.is_dir() {
        fs::create_dir_all(dir)?;
        sync_dir(parent)?;
    }
    Ok(())
}

fn sync_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir; // No portable directory synchronization in std elsewhere.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_files_a_namespace_is_made_of() {
        for good in ["seg-00000001.swt", "seg-1.del", "seg-00000012.3.del", "blob-00000002.fzt"] {
            assert!(is_file_name(good), "{good:?}");
        }
        for bad in ["schema", "manifest", "manifest.tmp", "lock", "wal-1", "../schema", "seg-1.swt/../x", "seg-.swt", "seg-1.swt.part", "seg-1..del", "seg-1.x.del", "seg-a.del", "blob-.fzt", "blob-1.fzt/..", "seg-1.2.3.del", ".del", "seg-.del", ""] {
            assert!(!is_file_name(bad), "{bad:?}");
        }
    }

    #[test]
    fn states_differ_with_either_part() {
        let a = state("s", b"m");
        assert_eq!(a, state("s", b"m"));
        assert_eq!(a.len(), 32);
        assert_ne!(a, state("s", b"n"));
        assert_ne!(a, state("t", b"m"));
        // Where the schema ends is part of it.
        assert_ne!(state("sm", b""), state("s", b"m"));
    }
}
