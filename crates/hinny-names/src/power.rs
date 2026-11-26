//! Battery or power, and what the index does differently on battery.
//!
//! Measured on a 6 performance + 12 efficiency core Mac:
//! - search: 18 threads 1.5 ms and 22 ms of CPU (idle threads spin), 12 threads 1.55 ms and 17.5 ms,
//!   4 threads 3.3 ms and 13 ms;
//! - build: 18 threads 25 s and 248 s of CPU, 6 threads 26 s and 83 s (more threads only contend
//!   in the kernel), 3 threads 45 s and 61 s.
//! So on power: searches on up to 12 threads, builds on one per performance core. On battery:
//! searches on 4, builds on 3 at background priority (efficiency cores, throttled disk reads), and
//! file events batched a little longer.

use std::ffi::{c_char, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOPSCopyPowerSourcesInfo() -> *const c_void;
    fn IOPSGetProvidingPowerSourceType(snapshot: *const c_void) -> *const c_void;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringGetCString(s: *const c_void, buffer: *mut c_char, size: isize, encoding: u32) -> u8;
    fn CFRelease(value: *const c_void);
}

unsafe extern "C" {
    fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
}

/// Quality-of-service classes (sys/qos.h).
#[derive(Clone, Copy)]
pub enum Qos {
    UserInteractive = 0x21,
    UserInitiated = 0x19,
    Utility = 0x11,
    /// Efficiency cores only, with throttled disk I/O.
    Background = 0x09,
}

pub fn set_thread_qos(qos: Qos) {
    unsafe { pthread_set_qos_class_self_np(qos as u32, 0) };
}

/// Last check (seconds since 1970) and its answer (bit 0: on battery).
static CHECKED: AtomicU64 = AtomicU64::new(0);

/// Whether the Mac runs on battery right now. Cached for 20 s (asking IOKit costs ~0.1 ms).
/// `HINNY_NAMES_POWER=battery` or `=ac` overrides it (for testing).
pub fn on_battery() -> bool {
    match std::env::var("HINNY_NAMES_POWER").as_deref() {
        Ok("battery") => return true,
        Ok("ac") => return false,
        _ => {}
    }
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let cached = CHECKED.load(Ordering::Relaxed);
    if cached != 0 && now.saturating_sub(cached >> 1) < 20 {
        return cached & 1 == 1;
    }
    let battery = unsafe {
        let snapshot = IOPSCopyPowerSourcesInfo();
        if snapshot.is_null() {
            false
        } else {
            let kind = IOPSGetProvidingPowerSourceType(snapshot);
            let mut buf = [0 as c_char; 64];
            let ok = !kind.is_null() && CFStringGetCString(kind, buf.as_mut_ptr(), buf.len() as isize, 0x0800_0100) != 0;
            CFRelease(snapshot);
            ok && std::ffi::CStr::from_ptr(buf.as_ptr()).to_bytes() == b"Battery Power"
        }
    };
    CHECKED.store((now << 1) | battery as u64, Ordering::Relaxed);
    battery
}

/// Threads a search uses on battery: 3.3 ms instead of 1.5 ms, for about half the CPU.
pub const BATTERY_SEARCH_THREADS: usize = 4;
/// On power: as fast as all 18 threads, with less spinning.
pub const POWER_SEARCH_THREADS: usize = 12;
pub const BATTERY_BUILD_THREADS: usize = 3;

/// Performance cores (more build threads than these only contend in the kernel).
pub fn performance_cores() -> usize {
    let mut n: i32 = 0;
    let mut size = std::mem::size_of::<i32>();
    let ok = unsafe { libc::sysctlbyname(c"hw.perflevel0.physicalcpu".as_ptr(), (&mut n as *mut i32).cast(), &mut size, std::ptr::null_mut(), 0) } == 0;
    if ok && n > 0 { n as usize } else { std::thread::available_parallelism().map_or(4, |n| n.get() / 2).max(2) }
}

/// The pool searches run on: all cores on power, four on battery.
pub fn search_pool() -> &'static rayon::ThreadPool {
    static FULL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    static BATTERY: OnceLock<rayon::ThreadPool> = OnceLock::new();
    if on_battery() {
        BATTERY.get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(BATTERY_SEARCH_THREADS)
                .thread_name(|i| format!("hinny-names-search-{i}"))
                .start_handler(|_| set_thread_qos(Qos::UserInteractive))
                .build()
                .expect("search pool")
        })
    } else {
        FULL.get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(std::thread::available_parallelism().map_or(8, |n| n.get()).min(POWER_SEARCH_THREADS))
                .thread_name(|i| format!("hinny-names-search-{i}"))
                .start_handler(|_| set_thread_qos(Qos::UserInteractive))
                .build()
                .expect("search pool")
        })
    }
}

/// A pool for building: one thread per performance core on power; on battery 3 at background
/// priority (efficiency cores, throttled reads). RAYON_NUM_THREADS overrides the count.
pub fn build_pool() -> rayon::ThreadPool {
    let battery = on_battery();
    let qos = if battery { Qos::Background } else { Qos::UserInitiated };
    let threads = std::env::var("RAYON_NUM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(if battery { BATTERY_BUILD_THREADS } else { performance_cores() });
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("hinny-names-build-{i}"))
        .start_handler(move |_| set_thread_qos(qos))
        .build()
        .expect("build pool")
}

/// How long FSEvents batches events before delivering them: longer on battery (fewer wake-ups),
/// still well under the 1 s freshness target.
pub fn event_latency() -> f64 {
    if on_battery() { 0.3 } else { 0.05 }
}
