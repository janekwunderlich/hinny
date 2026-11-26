//! Hinny names: the names and metadata of everything under a folder.
//!
//! - `walk`: reads directories with `getattrlistbulk` on all cores.
//! - `store`: a column store in one file, memory-mapped copy-on-write (instant start, patched in place).
//! - `search`: SIMD substring scan over one folded name array, ranked, on all cores.
//! - `fast`: search as you type: repeats, word-start lists, refining the last keystroke's matches,
//!   else a scan on a pool that parks between keystrokes (`pool`); `starts` builds the lists.
//! - `watch`: FSEvents, resumed from the saved event ID, so changes made while off are caught up on.
//! - `power`: battery or power, and what changes on battery (fewer search threads, background builds).

pub mod fast;
pub mod ffi;
pub mod fold;
pub mod localized;
pub mod pool;
pub mod power;
pub mod search;
pub mod service;
pub mod starts;
pub mod store;
pub mod walk;
pub mod watch;

pub use search::{Hit, search};
pub use store::{Index, Traits};
