//! C interface for the Swift app (header: `include/hinny_names.h`). One opaque handle per index;
//! every call is thread-safe.

use std::ffi::{CStr, c_char};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use crate::service::{self, Service};

#[repr(C)]
pub struct HinnyNameHit {
    pub entry: u32,
    pub score: f32,
    /// `store::Traits` bits (directory, document, application, hidden, …).
    pub traits: u8,
    /// Seconds since 1970.
    pub modified: u32,
}

fn path_arg<'a>(p: *const c_char) -> Option<&'a Path> {
    if p.is_null() {
        return None;
    }
    Some(Path::new(std::ffi::OsStr::from_bytes(unsafe { CStr::from_ptr(p) }.to_bytes())))
}

/// Opens the index at `path` (replaying its journal); null if it doesn't exist or can't be read.
#[unsafe(no_mangle)]
pub extern "C" fn hinny_names_open(path: *const c_char) -> *mut Service {
    let Some(path) = path_arg(path) else { return std::ptr::null_mut() };
    match Service::open(path) {
        Ok(s) => Box::into_raw(Box::new(s)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Builds an index of everything under `root` at `path`. Blocking; returns the number of entries,
/// or -1 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn hinny_names_build(root: *const c_char, path: *const c_char) -> i64 {
    let (Some(root), Some(path)) = (path_arg(root), path_arg(path)) else { return -1 };
    service::build(root, path).map_or(-1, |n| n as i64)
}

#[unsafe(no_mangle)]
pub extern "C" fn hinny_names_count(s: *const Service) -> u64 {
    let Some(s) = (unsafe { s.as_ref() }) else { return 0 };
    s.index.read().unwrap().total() as u64
}

/// The best `capacity` entries for `query`, written to `out`; returns how many.
#[unsafe(no_mangle)]
pub extern "C" fn hinny_names_search(s: *const Service, query: *const c_char, out: *mut HinnyNameHit, capacity: u32) -> u32 {
    let Some(s) = (unsafe { s.as_ref() }) else { return 0 };
    if query.is_null() || out.is_null() || capacity == 0 {
        return 0;
    }
    let query = String::from_utf8_lossy(unsafe { CStr::from_ptr(query) }.to_bytes()).into_owned();
    let hits = s.search(&query, capacity as usize);
    let index = s.index.read().unwrap();
    for (i, hit) in hits.iter().enumerate() {
        unsafe {
            *out.add(i) = HinnyNameHit { entry: hit.index as u32, score: hit.score, traits: index.traits(hit.index).0, modified: index.modified(hit.index) };
        }
    }
    hits.len() as u32
}

/// Writes the entry's full path (NUL-terminated) into `buffer`; returns its length, or 0 if it
/// doesn't fit or the entry is gone.
#[unsafe(no_mangle)]
pub extern "C" fn hinny_names_path(s: *const Service, entry: u32, buffer: *mut c_char, capacity: u32) -> u32 {
    let Some(s) = (unsafe { s.as_ref() }) else { return 0 };
    let index = s.index.read().unwrap();
    let e = entry as usize;
    if e >= index.total() || index.is_deleted(e) || buffer.is_null() {
        return 0;
    }
    let path = index.path(e);
    let bytes = path.as_os_str().as_bytes();
    if bytes.len() + 1 > capacity as usize {
        return 0;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer as *mut u8, bytes.len());
        *buffer.add(bytes.len()) = 0;
    }
    bytes.len() as u32
}

/// Follows file system changes from now on (and catches up on what changed since the index was
/// saved). Returns false if FSEvents couldn't start.
#[unsafe(no_mangle)]
pub extern "C" fn hinny_names_watch(s: *const Service) -> bool {
    let Some(s) = (unsafe { s.as_ref() }) else { return false };
    s.watch(|_line| {})
}

/// Saves pending changes to the journal and closes the index.
#[unsafe(no_mangle)]
pub extern "C" fn hinny_names_close(s: *mut Service) {
    if !s.is_null() {
        drop(unsafe { Box::from_raw(s) });
    }
}
