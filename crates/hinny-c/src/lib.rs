//! hinny's C interface (`include/hinny.h`): what Swift, Kotlin, Node and the rest call.
//!
//! Small on purpose: handles for a database and a namespace, and one function per operation.
//! Documents, queries and results cross as JSON, vectors as arrays of float, so a binding in
//! another language is a page of code, and adding a field to a query doesn't change the
//! interface. A JSON document costs a few microseconds to parse; writes that matter go in
//! batches (`hinny_apply`), and a query's time is the search.
//!
//! Every function is safe to call from any thread. Nothing here panics across the boundary: a
//! panic inside becomes an error (in builds that unwind; the release profile aborts instead).

use std::ffi::{CStr, CString, c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};

use hinny::{Change, Database, Namespace};
use hinny_json::{Fallible, document, filter, parse, patch, schema};
use serde_json::json;

/// A database: a directory of namespaces.
pub struct HinnyDb(Database);
/// A namespace: documents under one schema.
pub struct HinnyNamespace(Namespace);

// MARK: Crossing the boundary

/// Stores `message` in `*error` if there's somewhere to store it.
unsafe fn fail(error: *mut *mut c_char, message: String) {
    if !error.is_null() {
        // A message with a NUL in it is cut there: better than no message.
        let clean: String = message.chars().take_while(|c| *c != '\0').collect();
        // SAFETY: the caller passed a place for a pointer.
        unsafe { *error = CString::new(clean).unwrap_or_default().into_raw() };
    }
}

/// Runs `f`; an error or a panic goes to `*error` and `failed` comes back.
unsafe fn guard<T>(error: *mut *mut c_char, failed: T, f: impl FnOnce() -> Fallible<T>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        Ok(Err(message)) => {
            // SAFETY: as `fail`.
            unsafe { fail(error, message) };
            failed
        }
        Err(_) => {
            // SAFETY: as `fail`.
            unsafe { fail(error, "hinny: an internal error (a panic)".to_string()) };
            failed
        }
    }
}

/// A C string as text.
unsafe fn text<'a>(pointer: *const c_char, what: &str) -> Fallible<&'a str> {
    if pointer.is_null() {
        return Err(format!("{what} is NULL"));
    }
    // SAFETY: the caller passed a NUL-terminated string that outlives the call.
    unsafe { CStr::from_ptr(pointer) }.to_str().map_err(|_| format!("{what} isn't UTF-8"))
}

/// A handle as what it points to.
unsafe fn handle<'a, T>(pointer: *mut T, what: &str) -> Fallible<&'a T> {
    // SAFETY: the caller passed a handle this library returned and hasn't closed.
    unsafe { pointer.as_ref() }.ok_or_else(|| format!("{what} is NULL"))
}

/// A vector argument: `dimensions` floats, or none.
unsafe fn floats(vector: *const f32, dimensions: usize) -> Option<Vec<f32>> {
    if vector.is_null() || dimensions == 0 {
        return None;
    }
    // SAFETY: the caller passed `dimensions` floats.
    Some(unsafe { std::slice::from_raw_parts(vector, dimensions) }.to_vec())
}

/// A string for the caller to free with `hinny_free_string`.
fn give(s: String) -> *mut c_char {
    CString::new(s).map(CString::into_raw).unwrap_or(std::ptr::null_mut())
}

fn io(e: std::io::Error) -> String {
    e.to_string()
}

// MARK: The interface

/// The library's version. Not to be freed.
#[unsafe(no_mangle)]
pub extern "C" fn hinny_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Frees a string a function here returned.
///
/// # Safety
/// `string` is NULL or came from this library, and isn't used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_free_string(string: *mut c_char) {
    if !string.is_null() {
        // SAFETY: it came from `CString::into_raw` here.
        drop(unsafe { CString::from_raw(string) });
    }
}

/// A stable key for a string id (FNV-1a, 63 bits; the same as Python's `hinny.key`).
///
/// # Safety
/// `id` is a NUL-terminated string (NULL gives the key of the empty string).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_key(id: *const c_char) -> u64 {
    // SAFETY: as documented.
    let bytes = if id.is_null() { &[][..] } else { unsafe { CStr::from_ptr(id) }.to_bytes() };
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h & (u64::MAX >> 1)
}

/// Opens the database in a directory.
///
/// # Safety
/// `directory` is a NUL-terminated string; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_open(directory: *const c_char, error: *mut *mut c_char) -> *mut HinnyDb {
    // SAFETY: as documented, for this and the calls inside.
    unsafe {
        guard(error, std::ptr::null_mut(), || {
            let db = Database::open(text(directory, "the directory")?).map_err(io)?;
            Ok(Box::into_raw(Box::new(HinnyDb(db))))
        })
    }
}

/// Closes a database handle.
///
/// # Safety
/// `db` is NULL or a handle from `hinny_open` that isn't used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_close(db: *mut HinnyDb) {
    if !db.is_null() {
        // SAFETY: it came from `Box::into_raw` in `hinny_open`.
        drop(unsafe { Box::from_raw(db) });
    }
}

/// The database's namespaces, a JSON array of names.
///
/// # Safety
/// `db` is a handle from `hinny_open`; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_namespaces(db: *mut HinnyDb, error: *mut *mut c_char) -> *mut c_char {
    // SAFETY: as documented.
    unsafe { guard(error, std::ptr::null_mut(), || Ok(give(json!(handle(db, "the database")?.0.namespaces().map_err(io)?).to_string()))) }
}

/// Opens a namespace with a schema (see hinny.h).
///
/// # Safety
/// `db` is a handle from `hinny_open`; `name` is a NUL-terminated string and `schema_json` is
/// one or NULL; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_namespace_open(db: *mut HinnyDb, name: *const c_char, schema_json: *const c_char, error: *mut *mut c_char) -> *mut HinnyNamespace {
    // SAFETY: as documented.
    unsafe {
        guard(error, std::ptr::null_mut(), || {
            let (db, name) = (handle(db, "the database")?, text(name, "the namespace's name")?);
            // Without a schema: the namespace as it is, which must exist.
            let ns = match schema_json.is_null() {
                true => db.0.open_namespace(name),
                false => db.0.namespace(name, schema(text(schema_json, "the schema")?)?),
            };
            Ok(Box::into_raw(Box::new(HinnyNamespace(ns.map_err(io)?))))
        })
    }
}

/// Forks a namespace (see hinny.h).
///
/// # Safety
/// `ns` is a handle from `hinny_namespace_open`; `name` is a NUL-terminated string; `error` is
/// NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_namespace_fork(ns: *mut HinnyNamespace, name: *const c_char, error: *mut *mut c_char) -> *mut HinnyNamespace {
    // SAFETY: as documented.
    unsafe {
        guard(error, std::ptr::null_mut(), || {
            let (ns, name) = (handle(ns, "the namespace")?, text(name, "the fork's name")?);
            Ok(Box::into_raw(Box::new(HinnyNamespace(ns.0.fork(name).map_err(io)?))))
        })
    }
}

/// Deletes a namespace and everything in it (see hinny.h).
///
/// # Safety
/// `db` is a handle from `hinny_open`; `name` is a NUL-terminated string; `error` is NULL or a
/// place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_namespace_delete(db: *mut HinnyDb, name: *const c_char, error: *mut *mut c_char) -> c_int {
    // SAFETY: as documented.
    unsafe {
        guard(error, -1, || {
            let (db, name) = (handle(db, "the database")?, text(name, "the namespace's name")?);
            db.0.delete_namespace(name).map_err(io)?;
            Ok(0)
        })
    }
}

/// Closes a namespace handle.
///
/// # Safety
/// `ns` is NULL or a handle from `hinny_namespace_open` that isn't used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_namespace_close(ns: *mut HinnyNamespace) {
    if !ns.is_null() {
        // SAFETY: it came from `Box::into_raw` in `hinny_namespace_open`.
        drop(unsafe { Box::from_raw(ns) });
    }
}

/// Adds or replaces a document (see hinny.h).
///
/// # Safety
/// `ns` is a namespace handle; `document_json` is a NUL-terminated string; `vector` is NULL or
/// points to `dimensions` floats; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_upsert(ns: *mut HinnyNamespace, document_json: *const c_char, vector: *const f32, dimensions: usize, error: *mut *mut c_char) -> c_int {
    // SAFETY: as documented.
    unsafe {
        guard(error, -1, || {
            let doc = document(&parse(text(document_json, "the document")?, "the document")?, floats(vector, dimensions))?;
            handle(ns, "the namespace")?.0.upsert(doc).map_err(io)?;
            Ok(0)
        })
    }
}

/// Applies a batch of documents and deletes as one change (see hinny.h).
///
/// # Safety
/// `ns` is a namespace handle; `changes_json` is a NUL-terminated string; `error` is NULL or a
/// place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_apply(ns: *mut HinnyNamespace, changes_json: *const c_char, error: *mut *mut c_char) -> c_int {
    // SAFETY: as documented.
    unsafe {
        guard(error, -1, || {
            let value = parse(text(changes_json, "the changes")?, "the changes")?;
            let changes: Vec<Change> = hinny_json::changes(&value)?;
            handle(ns, "the namespace")?.0.apply(changes).map_err(io)?;
            Ok(0)
        })
    }
}

/// Deletes the document with a key.
///
/// # Safety
/// `ns` is a namespace handle; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_delete(ns: *mut HinnyNamespace, key: u64, error: *mut *mut c_char) -> c_int {
    // SAFETY: as documented.
    unsafe {
        guard(error, -1, || {
            handle(ns, "the namespace")?.0.delete(key).map_err(io)?;
            Ok(0)
        })
    }
}

/// Changes part of a document (see hinny.h): 1 if it was changed, 0 if there is no such
/// document or the condition didn't hold, -1 on an error.
///
/// # Safety
/// `ns` is a namespace handle; `patch_json` is a NUL-terminated string; `condition_json` is NULL
/// or one; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_patch(ns: *mut HinnyNamespace, key: u64, patch_json: *const c_char, condition_json: *const c_char, error: *mut *mut c_char) -> c_int {
    // SAFETY: as documented.
    unsafe {
        guard(error, -1, || {
            let ns = handle(ns, "the namespace")?;
            let patch = patch(&parse(text(patch_json, "the patch")?, "the patch")?, None)?;
            let changed = if condition_json.is_null() {
                ns.0.patch(key, &patch)
            } else {
                ns.0.patch_if(key, &patch, &filter(&parse(text(condition_json, "the condition")?, "the condition")?)?)
            };
            Ok(changed.map_err(io)? as c_int)
        })
    }
}

/// Changes part of every document that passes a filter (see hinny.h); how many, or -1 on an
/// error.
///
/// # Safety
/// `ns` is a namespace handle; `filter_json` and `patch_json` are NUL-terminated strings;
/// `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_patch_where(ns: *mut HinnyNamespace, filter_json: *const c_char, patch_json: *const c_char, error: *mut *mut c_char) -> i64 {
    // SAFETY: as documented.
    unsafe {
        guard(error, -1, || {
            let ns = handle(ns, "the namespace")?;
            let filter = filter(&parse(text(filter_json, "the filter")?, "the filter")?)?;
            let patch = patch(&parse(text(patch_json, "the patch")?, "the patch")?, None)?;
            Ok(ns.0.patch_where(&filter, &patch).map_err(io)? as i64)
        })
    }
}

/// Adds or replaces a document unless the one there fails a condition (see hinny.h): 1 if it
/// was written, 0 if not, -1 on an error.
///
/// # Safety
/// `ns` is a namespace handle; `document_json` and `condition_json` are NUL-terminated strings;
/// `vector` is NULL or points to `dimensions` floats; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_upsert_if(ns: *mut HinnyNamespace, document_json: *const c_char, vector: *const f32, dimensions: usize, condition_json: *const c_char, error: *mut *mut c_char) -> c_int {
    // SAFETY: as documented.
    unsafe {
        guard(error, -1, || {
            let ns = handle(ns, "the namespace")?;
            let doc = document(&parse(text(document_json, "the document")?, "the document")?, floats(vector, dimensions))?;
            let condition = filter(&parse(text(condition_json, "the condition")?, "the condition")?)?;
            Ok(ns.0.upsert_if(doc, &condition).map_err(io)? as c_int)
        })
    }
}

/// Deletes the document with a key if it passes a condition (see hinny.h): 1 if it was
/// deleted, 0 if not, -1 on an error.
///
/// # Safety
/// `ns` is a namespace handle; `condition_json` is a NUL-terminated string; `error` is NULL or a
/// place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_delete_if(ns: *mut HinnyNamespace, key: u64, condition_json: *const c_char, error: *mut *mut c_char) -> c_int {
    // SAFETY: as documented.
    unsafe {
        guard(error, -1, || {
            let ns = handle(ns, "the namespace")?;
            let condition = filter(&parse(text(condition_json, "the condition")?, "the condition")?)?;
            Ok(ns.0.delete_if(key, &condition).map_err(io)? as c_int)
        })
    }
}

/// A document's attributes and text, or NULL (and no error) if there is none.
///
/// # Safety
/// `ns` is a namespace handle; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_get(ns: *mut HinnyNamespace, key: u64, error: *mut *mut c_char) -> *mut c_char {
    // SAFETY: as documented.
    unsafe {
        guard(error, std::ptr::null_mut(), || {
            // One snapshot for both, so they're of the same version of the document.
            let snapshot = handle(ns, "the namespace")?.0.snapshot();
            let Some(attrs) = snapshot.get(key) else { return Ok(std::ptr::null_mut()) };
            let text = snapshot.text(key).unwrap_or_default();
            Ok(give(hinny_json::stored(&attrs, &text).to_string()))
        })
    }
}

/// Deletes the documents that pass a filter; how many, or -1 on an error.
///
/// # Safety
/// `ns` is a namespace handle; `filter_json` is a NUL-terminated string; `error` is NULL or a
/// place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_delete_where(ns: *mut HinnyNamespace, filter_json: *const c_char, error: *mut *mut c_char) -> i64 {
    // SAFETY: as documented.
    unsafe {
        guard(error, -1, || {
            let ns = handle(ns, "the namespace")?;
            let filter = filter(&parse(text(filter_json, "the filter")?, "the filter")?)?;
            Ok(ns.0.delete_where(&filter).map_err(io)? as i64)
        })
    }
}

/// How many documents pass a filter (every document, with NULL); -1 on an error.
///
/// # Safety
/// `ns` is a namespace handle; `filter_json` is NULL or a NUL-terminated string; `error` is NULL
/// or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_count(ns: *mut HinnyNamespace, filter_json: *const c_char, error: *mut *mut c_char) -> i64 {
    // SAFETY: as documented.
    unsafe {
        guard(error, -1, || {
            let ns = handle(ns, "the namespace")?;
            let filter = if filter_json.is_null() { None } else { Some(filter(&parse(text(filter_json, "the filter")?, "the filter")?)?) };
            Ok(ns.0.count(filter.as_ref()) as i64)
        })
    }
}

/// How many documents have each value of an attribute, as JSON (see hinny.h).
///
/// # Safety
/// `ns` is a namespace handle; `attribute` is a NUL-terminated string; `filter_json` is NULL or
/// one; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_count_by(ns: *mut HinnyNamespace, attribute: *const c_char, filter_json: *const c_char, limit: usize, error: *mut *mut c_char) -> *mut c_char {
    // SAFETY: as documented.
    unsafe {
        guard(error, std::ptr::null_mut(), || {
            let ns = handle(ns, "the namespace")?;
            let name = text(attribute, "the attribute")?;
            let filter = if filter_json.is_null() { None } else { Some(filter(&parse(text(filter_json, "the filter")?, "the filter")?)?) };
            Ok(give(hinny_json::counts(&ns.0.count_by(name, filter.as_ref(), limit)).to_string()))
        })
    }
}

/// Searches (see hinny.h).
///
/// # Safety
/// `ns` is a namespace handle; `query_json` is a NUL-terminated string; `vector` is NULL or
/// points to `dimensions` floats; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_query(ns: *mut HinnyNamespace, query_json: *const c_char, vector: *const f32, dimensions: usize, error: *mut *mut c_char) -> *mut c_char {
    // SAFETY: as documented.
    unsafe {
        guard(error, std::ptr::null_mut(), || {
            let value = parse(text(query_json, "the query")?, "the query")?;
            let q = hinny_json::query(&handle(ns, "the namespace")?.0, &value, floats(vector, dimensions))?;
            let hits = q.run().map_err(io)?;
            Ok(give(hinny_json::hits(&hits).to_string()))
        })
    }
}

/// One of the namespace's maintenance calls, by name.
unsafe fn maintain(ns: *mut HinnyNamespace, error: *mut *mut c_char, f: impl FnOnce(&Namespace) -> std::io::Result<()>) -> c_int {
    // SAFETY: the caller's, as documented on each function.
    unsafe {
        guard(error, -1, || {
            f(&handle(ns, "the namespace")?.0).map_err(io)?;
            Ok(0)
        })
    }
}

/// Makes every write so far durable.
///
/// # Safety
/// `ns` is a namespace handle; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_sync(ns: *mut HinnyNamespace, error: *mut *mut c_char) -> c_int {
    // SAFETY: as documented.
    unsafe { maintain(ns, error, Namespace::sync) }
}

/// Writes what's in memory out as a segment.
///
/// # Safety
/// `ns` is a namespace handle; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_flush(ns: *mut HinnyNamespace, error: *mut *mut c_char) -> c_int {
    // SAFETY: as documented.
    unsafe { maintain(ns, error, Namespace::flush) }
}

/// Merges everything into one segment.
///
/// # Safety
/// `ns` is a namespace handle; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_optimize(ns: *mut HinnyNamespace, error: *mut *mut c_char) -> c_int {
    // SAFETY: as documented.
    unsafe { maintain(ns, error, Namespace::optimize) }
}

/// Counts, as JSON.
///
/// # Safety
/// `ns` is a namespace handle; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_stats(ns: *mut HinnyNamespace, error: *mut *mut c_char) -> *mut c_char {
    // SAFETY: as documented.
    unsafe {
        guard(error, std::ptr::null_mut(), || {
            Ok(give(hinny_json::stats(&handle(ns, "the namespace")?.0.stats()).to_string()))
        })
    }
}

/// Reads every file against its checksums.
///
/// # Safety
/// `ns` is a namespace handle; `error` is NULL or a place for a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hinny_verify(ns: *mut HinnyNamespace, error: *mut *mut c_char) -> *mut c_char {
    // SAFETY: as documented.
    unsafe {
        guard(error, std::ptr::null_mut(), || {
            let v = handle(ns, "the namespace")?.0.verify().map_err(io)?;
            Ok(give(hinny_json::verified(&v).to_string()))
        })
    }
}
