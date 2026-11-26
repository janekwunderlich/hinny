//! When a query must give up: a deadline, a flag another thread raises, or both. Searches check it
//! between segments, MaxScore windows, term-at-a-time lists and vector clusters, and stop early;
//! the query then reports the stop as an error, so partial results never escape.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// A handle that cancels the queries given it (`Query::cancel`), from any thread.
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn new() -> Cancel {
        Cancel::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stop {
    pub deadline: Option<Instant>,
    pub cancel: Option<Cancel>,
}

impl Stop {
    pub const NEVER: Stop = Stop { deadline: None, cancel: None };

    /// Whether to give up now: cancelled, or past the deadline.
    #[inline]
    pub fn stopped(&self) -> bool {
        self.cancel.as_ref().is_some_and(Cancel::is_cancelled) || self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    /// The error a stopped query returns: TimedOut past its deadline, Interrupted if cancelled.
    pub fn error(&self) -> std::io::Error {
        if self.cancel.as_ref().is_some_and(Cancel::is_cancelled) {
            std::io::Error::new(std::io::ErrorKind::Interrupted, "the query was cancelled")
        } else {
            std::io::Error::new(std::io::ErrorKind::TimedOut, "the query ran past its deadline")
        }
    }
}
