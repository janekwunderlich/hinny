//! The name index as a running service: the mapped index, search as you type, FSEvents, and the
//! journal/compaction loop, in one place for the CLI (`hinny-names watch`) and the app (`ffi`).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::fast::Searcher;
use crate::power;
use crate::search::Hit;
use crate::starts::WordStarts;
use crate::store::Index;
use crate::watch::{self, Watcher};

pub fn now() -> u32 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as u32)
}

pub struct Service {
    pub path: PathBuf,
    pub index: Arc<RwLock<Index>>,
    searcher: Arc<Mutex<Searcher>>,
    watcher: Mutex<Option<Watcher>>,
    saver: Mutex<Option<JoinHandle<()>>>,
    stop: Arc<AtomicBool>,
}

impl Service {
    /// Opens an index (replaying its journal) and the word-start lists that belong to it.
    pub fn open(path: &Path) -> std::io::Result<Service> {
        let index = Index::open_with_journal(path)?;
        let searcher = Searcher::for_index(&index, path);
        let index = Arc::new(RwLock::new(index));
        // Read the search columns in now, at low priority, instead of on the first keystroke. In 4 MB
        // steps, each under its own read lock: held for the whole ~700 MB it kept the watcher's
        // write lock waiting, and searches queue behind a waiting writer.
        let warm = index.clone();
        let _ = std::thread::Builder::new().name("hinny-names-prefault".into()).spawn(move || {
            power::set_thread_qos(power::Qos::Utility);
            let (mut at, build) = (0, warm.read().unwrap().build_id);
            loop {
                let ix = warm.read().unwrap();
                // Compacted meanwhile: a new file, read in by its own reopen.
                if ix.build_id != build {
                    break;
                }
                match ix.prefault_part(at, 4 << 20) {
                    Some(next) => at = next,
                    None => break,
                }
            }
        });
        Ok(Service {
            path: path.to_path_buf(),
            index,
            searcher: Arc::new(Mutex::new(searcher)),
            watcher: Mutex::new(None),
            saver: Mutex::new(None),
            stop: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<Hit> {
        // Lock order everywhere: searcher, then index.
        let t0 = Instant::now();
        let mut searcher = self.searcher.lock().unwrap();
        let index = self.index.read().unwrap();
        let t1 = Instant::now();
        let hits = searcher.search(&index, query, limit, now());
        // HINNY_NAMES_DEBUG=1: where a search's time went (no query text).
        if std::env::var_os("HINNY_NAMES_DEBUG").is_some() {
            eprintln!("[hinny-names] waited {:.3} ms for locks, searched {:.3} ms via {:?}", (t1 - t0).as_secs_f64() * 1000.0, t1.elapsed().as_secs_f64() * 1000.0, searcher.last);
        }
        hits
    }

    /// Follows file system changes, appends them to the journal every 5 s, and compacts by the
    /// policy in `should_compact`. `log` receives one line per notable event.
    pub fn watch(&self, log: impl Fn(String) + Send + Sync + 'static) -> bool {
        let mut slot = self.watcher.lock().unwrap();
        if slot.is_some() {
            return true;
        }
        let log = Arc::new(log);
        let started = Instant::now();
        let events_log = log.clone();
        let Some(watcher) = watch::watch_index(self.index.clone(), move |applied, events| {
            if applied.history_done {
                events_log(format!("caught up in {:.0} ms", started.elapsed().as_secs_f64() * 1000.0));
            }
            if applied.added + applied.removed + applied.updated > 0 {
                events_log(format!("{} events: +{} −{} ~{}", events.len(), applied.added, applied.removed, applied.updated));
            }
        }) else {
            return false;
        };
        *slot = Some(watcher);

        let (index, searcher, stop, path) = (self.index.clone(), self.searcher.clone(), self.stop.clone(), self.path.clone());
        let saver = std::thread::Builder::new()
            .name("hinny-names-saver".into())
            .spawn(move || {
                power::set_thread_qos(power::Qos::Utility);
                let mut last_compaction = Instant::now();
                while !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_secs(5));
                    let compacted = {
                        let mut ix = index.write().unwrap();
                        if let Err(e) = ix.flush_journal() {
                            log(format!("journal write failed: {e}"));
                        }
                        let (bytes, changes) = ix.journal.as_ref().map_or((0, 0), |j| (j.bytes, j.changes));
                        if !should_compact(bytes, changes, ix.count, last_compaction.elapsed(), power::on_battery()) {
                            continue;
                        }
                        last_compaction = Instant::now();
                        let t = Instant::now();
                        match ix.compact_to(&path).and_then(|n| Index::open_with_journal(&path).map(|fresh| (n, fresh))) {
                            Ok((n, fresh)) => {
                                *ix = fresh;
                                log(format!("compacted {n} entries in {:.1} s (journal was {:.1} MB, {changes} changes)", t.elapsed().as_secs_f64(), bytes as f64 / 1e6));
                                true
                            }
                            Err(e) => {
                                log(format!("compaction failed: {e}"));
                                false
                            }
                        }
                    };
                    if compacted {
                        // Entry numbers changed: new word-start lists, then a new searcher with them. Built
                        // from a second, lock-free map of the file just written (seconds): under the
                        // index lock, the watcher's next write would wait on it and searches behind that.
                        // Searches meanwhile scan instead of using the old build's lists.
                        if let Err(e) = Index::open(&path).and_then(|built| WordStarts::build(&built, &WordStarts::path_for(&path), now())) {
                            log(format!("word starts failed: {e}"));
                        }
                        let mut s = searcher.lock().unwrap();
                        let ix = index.read().unwrap();
                        *s = Searcher::for_index(&ix, &path);
                    }
                }
            })
            .ok();
        *self.saver.lock().unwrap() = saver;
        true
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.watcher.lock().unwrap().take();
        if let Ok(mut ix) = self.index.write() {
            let _ = ix.flush_journal();
        }
    }
}

/// Rewriting the whole index (722 MB for 9 M entries, 1.7 s) only pays off when the journal has grown:
/// past 2 % of the entries or 64 MB, at most hourly, and only on power. On battery it waits for the
/// charger unless the journal reaches 256 MB.
pub fn should_compact(journal_bytes: u64, changes: usize, count: usize, since_last: Duration, on_battery: bool) -> bool {
    const MB: u64 = 1_000_000;
    let grown = journal_bytes > 64 * MB || changes > count / 50;
    let huge = journal_bytes > 256 * MB;
    huge || (grown && !on_battery && since_last > Duration::from_secs(3600))
}

/// Builds an index of everything under `root` (and its word-start lists). Blocking: 26 s for 9 M
/// entries on power, longer at background priority on battery.
pub fn build(root: &Path, path: &Path) -> std::io::Result<usize> {
    let since = unsafe { watch::FSEventsGetCurrentEventId() };
    let builder = power::build_pool().install(|| crate::store::Builder::walk(root));
    builder.write(path, root, since)?;
    let _ = std::fs::remove_file(crate::store::Journal::path_for(path));
    let index = Index::open(path)?;
    power::build_pool().install(|| WordStarts::build(&index, &WordStarts::path_for(path), now()))?;
    Ok(builder.len())
}
