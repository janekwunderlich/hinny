//! Deterministic I/O failures, for tests (the `failpoints` feature; without it, nothing at all).
//! Each place the engine writes or synchronizes calls `check` with its name; a test arms one name
//! to fail on its nth call, runs a workload, and checks what survives (`tests/faults.rs`).

use std::io;

#[cfg(feature = "failpoints")]
mod imp {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    struct State {
        armed: Option<(String, usize)>,
        hits: BTreeMap<String, usize>,
    }

    static STATE: Mutex<State> = Mutex::new(State { armed: None, hits: BTreeMap::new() });

    pub fn check(point: &str) -> std::io::Result<()> {
        let mut s = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let n = s.hits.entry(point.to_string()).or_insert(0);
        *n += 1;
        let n = *n;
        if s.armed.as_ref().is_some_and(|(p, at)| p == point && *at == n) {
            s.armed = None;
            return Err(std::io::Error::other(format!("injected failure at {point} (call {n})")));
        }
        Ok(())
    }

    /// Fails the `nth` call (from 1) to `point` from now on, once; resets the counts.
    pub fn fail_at(point: &str, nth: usize) {
        let mut s = STATE.lock().unwrap_or_else(|e| e.into_inner());
        s.hits.clear();
        s.armed = Some((point.to_string(), nth));
    }

    /// Disarms, and returns how often each point was reached since the last `fail_at` or `reset`.
    pub fn reset() -> BTreeMap<String, usize> {
        let mut s = STATE.lock().unwrap_or_else(|e| e.into_inner());
        s.armed = None;
        std::mem::take(&mut s.hits)
    }
}

#[cfg(feature = "failpoints")]
pub use imp::{fail_at, reset};

#[inline(always)]
pub(crate) fn check(point: &str) -> io::Result<()> {
    #[cfg(feature = "failpoints")]
    return imp::check(point);
    #[cfg(not(feature = "failpoints"))]
    {
        let _ = point;
        Ok(())
    }
}

/// The point for an atomic write of `path`: by the kind of file (manifest, del, schema, …).
pub(crate) fn kind(path: &std::path::Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("del") => "write del",
        _ if path.file_name().is_some_and(|n| n == "manifest") => "write manifest",
        _ if path.file_name().is_some_and(|n| n == "schema") => "write schema",
        _ => "write other",
    }
}
