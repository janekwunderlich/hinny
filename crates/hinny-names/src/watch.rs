//! Keeping the index current with FSEvents (called directly, no crate).
//!
//! The stream starts from the event ID saved with the index, so what changed while nothing was
//! watching is replayed first ("history"), and the index never needs a full rescan at start.

use std::ffi::{CStr, c_char, c_void};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::store::Index;
use crate::walk::read_dir;

// FSEvents flags (CoreServices/FSEvents.h).
const CREATE_NO_DEFER: u32 = 0x02;
const CREATE_WATCH_ROOT: u32 = 0x04;
const CREATE_FILE_EVENTS: u32 = 0x10;
pub const MUST_SCAN_SUBDIRS: u32 = 0x01;
pub const HISTORY_DONE: u32 = 0x10;
pub const ROOT_CHANGED: u32 = 0x20;
pub const ITEM_IS_DIR: u32 = 0x20000;

#[repr(C)]
struct Context {
    version: isize,
    info: *mut c_void,
    retain: *const c_void,
    release: *const c_void,
    copy_description: *const c_void,
}

type Callback = extern "C" fn(*const c_void, *mut c_void, usize, *const *const c_char, *const u32, *const u64);

#[link(name = "CoreServices", kind = "framework")]
unsafe extern "C" {
    fn FSEventStreamCreate(alloc: *const c_void, cb: Callback, ctx: *mut Context, paths: *const c_void, since: u64, latency: f64, flags: u32) -> *mut c_void;
    fn FSEventStreamSetDispatchQueue(stream: *mut c_void, queue: *mut c_void);
    fn FSEventStreamStart(stream: *mut c_void) -> u8;
    fn FSEventStreamStop(stream: *mut c_void);
    fn FSEventStreamInvalidate(stream: *mut c_void);
    fn FSEventStreamRelease(stream: *mut c_void);
    pub fn FSEventsGetCurrentEventId() -> u64;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithBytes(alloc: *const c_void, bytes: *const u8, len: isize, encoding: u32, external: u8) -> *const c_void;
    fn CFArrayCreate(alloc: *const c_void, values: *const *const c_void, n: isize, callbacks: *const c_void) -> *const c_void;
    fn CFRelease(value: *const c_void);
    static kCFTypeArrayCallBacks: c_void;
}

unsafe extern "C" {
    fn dispatch_queue_create(label: *const c_char, attr: *const c_void) -> *mut c_void;
    fn dispatch_queue_attr_make_with_qos_class(attr: *const c_void, qos: u32, relative: i32) -> *const c_void;
}

pub struct Event {
    pub path: PathBuf,
    pub flags: u32,
    pub id: u64,
}

type Handler = Box<dyn Fn(Vec<Event>) + Send + Sync>;

/// A running FSEvents stream; stops when dropped.
pub struct Watcher {
    stream: *mut c_void,
    _handler: Box<Handler>,
}

unsafe impl Send for Watcher {}

extern "C" fn callback(_stream: *const c_void, info: *mut c_void, n: usize, paths: *const *const c_char, flags: *const u32, ids: *const u64) {
    let handler = unsafe { &*(info as *const Handler) };
    let events = (0..n)
        .map(|i| unsafe {
            Event {
                path: PathBuf::from(std::ffi::OsStr::from_bytes(CStr::from_ptr(*paths.add(i)).to_bytes())),
                flags: *flags.add(i),
                id: *ids.add(i),
            }
        })
        .collect();
    handler(events);
}

impl Watcher {
    /// Watches `root` from event `since` (0: from now), calling `handler` with batches of events.
    pub fn start(root: &Path, since: u64, latency: f64, handler: impl Fn(Vec<Event>) + Send + Sync + 'static) -> Option<Watcher> {
        let handler: Box<Handler> = Box::new(Box::new(handler));
        unsafe {
            let bytes = root.as_os_str().as_bytes();
            let path = CFStringCreateWithBytes(std::ptr::null(), bytes.as_ptr(), bytes.len() as isize, 0x0800_0100, 0);
            let paths = CFArrayCreate(std::ptr::null(), &path, 1, &kCFTypeArrayCallBacks as *const c_void);
            let mut ctx = Context { version: 0, info: (&*handler as *const Handler) as *mut c_void, retain: std::ptr::null(), release: std::ptr::null(), copy_description: std::ptr::null() };
            // kFSEventStreamEventIdSinceNow is all ones.
            let since = if since == 0 { u64::MAX } else { since };
            let stream = FSEventStreamCreate(std::ptr::null(), callback, &mut ctx, paths, since, latency, CREATE_NO_DEFER | CREATE_WATCH_ROOT | CREATE_FILE_EVENTS);
            CFRelease(paths);
            CFRelease(path);
            if stream.is_null() {
                return None;
            }
            // Utility priority: applying events is background work, not something a person waits on.
            let attr = dispatch_queue_attr_make_with_qos_class(std::ptr::null(), crate::power::Qos::Utility as u32, 0);
            let queue = dispatch_queue_create(c"dev.hinny.names.events".as_ptr(), attr);
            FSEventStreamSetDispatchQueue(stream, queue);
            if FSEventStreamStart(stream) == 0 {
                FSEventStreamInvalidate(stream);
                FSEventStreamRelease(stream);
                return None;
            }
            Some(Watcher { stream, _handler: handler })
        }
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        unsafe {
            FSEventStreamStop(self.stream);
            FSEventStreamInvalidate(self.stream);
            FSEventStreamRelease(self.stream);
        }
    }
}

/// What applying a batch did, for logs and benchmarks.
#[derive(Default, Debug, Clone, Copy)]
pub struct Applied {
    pub added: usize,
    pub removed: usize,
    pub updated: usize,
    pub history_done: bool,
}

/// What a changed path is now, checked without holding any lock.
enum Seen {
    Exists { is_dir: bool, is_symlink: bool, modified: u32 },
    Gone,
    /// Can't tell (no permission under macOS privacy settings): leave the entry as it is.
    Unknown,
}

/// Applies a batch of events: added and changed paths are (re)read, gone ones are removed, and a
/// new folder is read whole (moving a folder in is a single event).
///
/// In three phases, so searches aren't held up: the file system is checked with no lock, entries
/// are looked up under the read lock (searches run alongside), and only the changes themselves take
/// the write lock.
pub fn apply(index: &RwLock<Index>, events: &[Event]) -> Applied {
    let mut out = Applied::default();
    let root = index.read().unwrap().root.clone();
    let seen: Vec<Seen> = events
        .iter()
        .map(|event| {
            if event.flags & HISTORY_DONE != 0 || !event.path.starts_with(&root) {
                return Seen::Unknown;
            }
            match std::fs::symlink_metadata(&event.path) {
                Ok(meta) => Seen::Exists {
                    is_dir: meta.is_dir(),
                    is_symlink: meta.file_type().is_symlink(),
                    modified: meta.modified().ok().and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs() as u32),
                },
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Seen::Gone,
                Err(_) => Seen::Unknown,
            }
        })
        .collect();

    // Which entries the events are about, and whether anything needs writing at all.
    let (build_id, found, needs_write) = {
        let ix = index.read().unwrap();
        let found: Vec<Option<usize>> = events.iter().zip(&seen).map(|(e, s)| if matches!(s, Seen::Unknown) { None } else { ix.lookup(&e.path) }).collect();
        let needs_write = events.iter().zip(&seen).zip(&found).any(|((event, s), f)| match (s, f) {
            (Seen::Exists { modified, .. }, Some(e)) => ix.modified(*e) != *modified || event.flags & MUST_SCAN_SUBDIRS != 0,
            (Seen::Exists { .. }, None) => true,
            (Seen::Gone, Some(_)) => true,
            _ => false,
        });
        (ix.build_id, found, needs_write)
    };
    let last_id = events.iter().map(|e| e.id).max().unwrap_or(0);
    out.history_done = events.iter().any(|e| e.flags & HISTORY_DONE != 0);

    let mut ix = index.write().unwrap();
    ix.last_event_id = ix.last_event_id.max(last_id);
    if !needs_write {
        return out;
    }
    // A compaction between the phases renumbered the entries: look them up again.
    let found = if ix.build_id == build_id { found } else { events.iter().zip(&seen).map(|(e, s)| if matches!(s, Seen::Unknown) { None } else { ix.lookup(&e.path) }).collect() };
    for ((event, s), f) in events.iter().zip(&seen).zip(found) {
        match (s, f) {
            (Seen::Exists { is_dir, modified, .. }, Some(e)) => {
                if ix.modified(e) != *modified {
                    ix.set_modified(e, *modified);
                    out.updated += 1;
                }
                if event.flags & MUST_SCAN_SUBDIRS != 0 && *is_dir {
                    out.added += add_tree(&mut ix, e, &event.path);
                }
            }
            (Seen::Exists { is_dir, is_symlink, modified }, None) => {
                // (Another event in this batch may have added it already.)
                let existing = ix.lookup(&event.path);
                if existing.is_none() {
                    if let Some(e) = ensure(&mut ix, &event.path, *is_dir, *modified, &mut out) {
                        if *is_dir && !*is_symlink {
                            out.added += add_tree(&mut ix, e, &event.path);
                        }
                    }
                }
            }
            (Seen::Gone, Some(e)) => {
                if !ix.is_deleted(e) {
                    ix.remove(e);
                    out.removed += 1;
                }
            }
            _ => {}
        }
    }
    out
}

/// Adds `path` (and any missing folders above it); returns its index.
fn ensure(ix: &mut Index, path: &Path, is_dir: bool, modified: u32, out: &mut Applied) -> Option<usize> {
    if let Some(e) = ix.lookup(path) {
        return Some(e);
    }
    let parent_path = path.parent()?;
    let parent = match ix.lookup(parent_path) {
        Some(p) => p,
        None => {
            let meta = std::fs::symlink_metadata(parent_path).ok()?;
            let m = meta.modified().ok().and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs() as u32);
            ensure(ix, parent_path, true, m, out)?
        }
    };
    out.added += 1;
    Some(ix.add(parent, path.file_name()?.as_bytes(), is_dir, modified))
}

/// Reads a folder that just appeared (and everything in it) into the index.
fn add_tree(ix: &mut Index, dir: usize, path: &Path) -> usize {
    let mut added = 0;
    let mut stack = vec![(dir, path.to_path_buf())];
    let mut buffer = vec![0u8; 256 * 1024];
    let mut entries = Vec::new();
    while let Some((e, p)) = stack.pop() {
        entries.clear();
        read_dir(&p, &mut entries, &mut buffer);
        for entry in entries.drain(..) {
            if ix.child(e, &entry.name).is_some() {
                continue;
            }
            let child = ix.add(e, &entry.name, entry.is_dir, entry.modified);
            added += 1;
            if entry.is_dir && !entry.is_symlink && !entry.stop {
                stack.push((child, p.join(std::ffi::OsStr::from_bytes(&entry.name))));
            }
        }
    }
    added
}

/// Watches `index` live: applies events as they come and calls `on_batch` after each batch.
pub fn watch_index(index: Arc<RwLock<Index>>, on_batch: impl Fn(Applied, &[Event]) + Send + Sync + 'static) -> Option<Watcher> {
    let (root, since) = {
        let ix = index.read().unwrap();
        (ix.root.clone(), ix.last_event_id)
    };
    Watcher::start(&root, since, crate::power::event_latency(), move |events| {
        let applied = apply(&index, &events);
        on_batch(applied, &events);
    })
}
