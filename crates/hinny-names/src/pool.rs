//! Running one query on several cores without spinning.
//!
//! rayon's idle workers spin and `sched_yield` before sleeping: an empty task every millisecond cost
//! ~200% CPU on macOS, against ~3–4% with Grand Central Dispatch (rayon issues #642 and #576). Our
//! queries are short bursts a person types, so the pool must cost nothing between them: this hands
//! each burst to GCD's `dispatch_apply`, whose threads park in the kernel, at user-interactive
//! priority (performance cores). Workers take chunks from a shared counter, so a chunk full of
//! matches (a whole node_modules) doesn't hold up the rest.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::power;

unsafe extern "C" {
    fn dispatch_get_global_queue(identifier: isize, flags: usize) -> *mut c_void;
    fn dispatch_apply_f(iterations: usize, queue: *mut c_void, context: *mut c_void, work: extern "C" fn(*mut c_void, usize));
}

/// QOS_CLASS_USER_INTERACTIVE (sys/qos.h).
const USER_INTERACTIVE: isize = 0x21;

/// Workers for a query: one per performance core on power (more only lands on efficiency cores
/// and makes the slowest chunk the tail), four on battery.
pub fn workers() -> usize {
    if power::on_battery() { power::BATTERY_SEARCH_THREADS } else { power::performance_cores() }
}

/// Workers for a full scan, which reads every name and is bound by each core's memory bandwidth:
/// on power 12 (6 workers took 1.5–3.5 ms against 1.1–2.0 ms for 12, for ~8% more CPU per keystroke;
/// parked workers don't spin, so the extra cores cost little), four on battery.
/// `HINNY_NAMES_SCAN_WORKERS` overrides it (for measuring).
pub fn scan_workers() -> usize {
    let default = if power::on_battery() {
        power::BATTERY_SEARCH_THREADS
    } else {
        std::thread::available_parallelism().map_or(8, |n| n.get()).min(power::POWER_SEARCH_THREADS)
    };
    std::env::var("HINNY_NAMES_SCAN_WORKERS").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Calls `f(chunk)` for every chunk in `0..chunks` on `workers` threads, returning when all are done.
pub fn for_each_chunk<F: Fn(usize) + Sync>(chunks: usize, workers: usize, f: &F) {
    if chunks == 0 {
        return;
    }
    if workers <= 1 || chunks == 1 {
        (0..chunks).for_each(f);
        return;
    }
    struct Job<'a, F> {
        next: AtomicUsize,
        chunks: usize,
        f: &'a F,
    }
    extern "C" fn work<F: Fn(usize) + Sync>(context: *mut c_void, _worker: usize) {
        let job = unsafe { &*(context as *const Job<F>) };
        loop {
            let chunk = job.next.fetch_add(1, Ordering::Relaxed);
            if chunk >= job.chunks {
                break;
            }
            (job.f)(chunk);
        }
    }
    let job = Job { next: AtomicUsize::new(0), chunks, f };
    unsafe {
        let queue = dispatch_get_global_queue(USER_INTERACTIVE, 0);
        // dispatch_apply blocks until every iteration has run, so `job` outlives them all.
        dispatch_apply_f(workers.min(chunks), queue, (&job as *const Job<F>) as *mut c_void, work::<F>);
    }
}
