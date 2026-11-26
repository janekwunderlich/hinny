//! A live text index: what's new is searchable at once, what's old is in immutable segments.
//!
//! The shape of an LSM tree, as Lucene and Turbopuffer use it for search:
//! - **Writes** come in batches (`write`; `upsert` and `delete` are batches of one). A batch is one
//!   write-ahead log record, so it's all or nothing after a crash, and one in-memory *chunk*,
//!   searched by scoring its postings directly. Chunks merge in memory two at a time when they're
//!   about the same size (a binomial heap of them), so there are about log₂ n.
//! - **A checkpoint** writes the memtable (its chunks) out as an immutable, memory-mapped segment,
//!   saves deletions, names the segments in the manifest (written whole, then renamed into place),
//!   and deletes the logs whose changes are in segments. When memory fills it happens in the
//!   background.
//! - **Deletes** stamp a passage with the sequence number of the batch that deleted it; a changed
//!   item is a delete and a put.
//! - **Merges** run in the background, by size tier, and rewrite segments that are mostly deleted.
//!   They work from the posting lists, so text is never indexed again.
//! - **Scores are global:** idf and average lengths come from all segments together (deleted
//!   passages count until merged away, as in Lucene), and segments' block bounds come from their
//!   impacts under those statistics, so a passage scores the same in whichever segment it is.
//!
//! **Versions (MVCC):** every batch gets the next sequence number, and the index publishes an
//! immutable `View` after it: the segments, the chunks, and that number. A query takes the current
//! view (a reference count, never the writer's lock) and sees exactly the batches up to its number:
//! chunks are immutable once published, and a view skips passages stamped deleted at or before its
//! number (Hekaton's end timestamps). So queries never wait for writes or writes for queries, and a
//! view kept with `Reader::view` is a snapshot as long as it's held. Its segments stay mapped while
//! it holds them, even after a merge deletes their files.
//!
//! Keys identify items (an email, a file) and can have many passages; versions let a caller tell
//! which items changed since it last synced.
//!
//! **Processes:** one writer at a time holds an exclusive lock on the index (`open`; another
//! process gets `WouldBlock`). Any number of readers open read-only snapshots (`open_read_only`),
//! which never change files, and reopen when `on_disk` changes.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;

use rayon::prelude::*;

use crate::analysis::Analyzer;
use crate::attrs::{Filter, Kind, Value};
use crate::bitmap::{Bitmap, Deletes};
use crate::norms;
use crate::search::{Clause, Query, Searcher, Sink, TopK, weight};
use crate::segment::{self, MAX_FIELDS, MemIndex, Segment};
use crate::tokenize::tokens;
use crate::vector::Metric;

/// A segment as the index has it: the file, and its deletions (stamped; see `Deletes`).
struct SegEntry {
    id: u64,
    segment: Arc<Segment>,
    deletes: Deletes,
    /// Deletions not yet saved.
    dirty: std::sync::atomic::AtomicBool,
    /// Which of the segment's deletion files the manifest names (`del_path`; 0: none, or the
    /// one without a number that formats before 7 wrote).
    del: std::sync::atomic::AtomicU64,
    /// The highest number a deletion file of this segment was written under, named by a
    /// manifest or not (a save can fail after its manifest is in place): the next is above it.
    del_written: std::sync::atomic::AtomicU64,
    /// Live passages with stored text in each of the segment's blob files (in `Segment::blobs`
    /// order): counted once, when a merge first asks, and kept current by the writer's deletions,
    /// so that choosing what a merge rewrites doesn't read every passage of the index each time.
    live_text: std::sync::OnceLock<Vec<std::sync::atomic::AtomicU64>>,
}

impl SegEntry {
    fn new(id: u64, segment: Segment, deletes: Deletes, dirty: bool) -> SegEntry {
        SegEntry { id, segment: Arc::new(segment), deletes, dirty: dirty.into(), del: 0.into(), del_written: 0.into(), live_text: std::sync::OnceLock::new() }
    }

    /// See `live_text`. Only the writer calls this and deletes, so the count and the deletions
    /// that follow it can't cross.
    fn live_text(&self) -> &[std::sync::atomic::AtomicU64] {
        self.live_text.get_or_init(|| {
            let mut live = vec![0u64; self.segment.blobs().count()];
            for d in 0..self.segment.docs {
                if let Some(i) = self.segment.text_blob(d).filter(|_| !self.deletes.deleted(d)) {
                    live[i] += 1;
                }
            }
            live.into_iter().map(std::sync::atomic::AtomicU64::new).collect()
        })
    }

    /// Deletes a passage as of `seq` (the writer's only way to): whether it was live.
    fn delete(&self, doc: u32, seq: u64) -> bool {
        let new = self.deletes.delete(doc, seq);
        if new {
            self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
            if let (Some(live), Some(i)) = (self.live_text.get(), self.segment.text_blob(doc)) {
                live[i].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        new
    }
}

/// Passages in memory, immutable once published (only their deletion stamps change).
struct Chunk {
    index: MemIndex,
    deletes: Deletes,
    keys: HashMap<u64, Vec<u32>>,
    /// Some item has more than one passage here (live or not).
    repeats: bool,
}

impl Chunk {
    fn new(index: MemIndex) -> Chunk {
        let mut keys: HashMap<u64, Vec<u32>> = HashMap::new();
        for (d, &k) in index.keys.iter().enumerate() {
            keys.entry(k).or_default().push(d as u32);
        }
        let repeats = keys.len() < index.len();
        Chunk { deletes: Deletes::new(index.len()), index, keys, repeats }
    }

    /// `a` then `b` as one chunk, with their deletions. (The new chunk is seen only by views from
    /// after it's published, so a deletion so far is a deletion for all of them.)
    fn concat(a: &Chunk, b: &Chunk) -> Chunk {
        let mut c = Chunk::new(a.index.concat(&b.index));
        for d in 0..a.index.len() as u32 {
            if a.deletes.deleted(d) {
                c.deletes.mark(d);
            }
        }
        let base = a.index.len() as u32;
        for d in 0..b.index.len() as u32 {
            if b.deletes.deleted(d) {
                c.deletes.mark(base + d);
            }
        }
        c
    }
}

/// A memtable: chunks, oldest first, and the log files holding their changes.
struct Memtable {
    chunks: Vec<Arc<Chunk>>,
    logs: Vec<PathBuf>,
    /// Bytes of the log records applied to it: about what its text, vectors and attributes take.
    bytes: usize,
}

impl Memtable {
    fn len(&self) -> usize {
        self.chunks.iter().map(|c| c.index.len()).sum()
    }
}

struct Merge {
    inputs: Vec<u64>,
    output: u64,
    /// Each input's deletions when the merge started: the passages it left out.
    dropped: Vec<Bitmap>,
    handle: JoinHandle<std::io::Result<usize>>,
}

/// A frozen memtable being written out as segment `id`.
struct Flush {
    id: u64,
    handle: JoinHandle<std::io::Result<()>>,
}

/// A document: an item (key) with its passages (one text per field each) and attributes. Its
/// version tells a caller syncing from elsewhere whether it changed.
#[derive(Clone, Debug, Default)]
pub struct Doc {
    pub key: u64,
    pub version: u64,
    pub passages: Vec<Vec<String>>,
    pub attrs: Vec<(String, Value)>,
    /// An embedding of the document, stored as the index's metric stores it: normalized for
    /// cosine, as it is for Euclidean distance and dot product (`vector::Metric`).
    pub vector: Option<Vec<f32>>,
}

/// What `View::document` finds of an item.
#[derive(Clone, Debug)]
pub struct Whole {
    pub version: u64,
    pub attrs: Vec<(String, Value)>,
    /// One text per field, if the index kept text.
    pub text: Option<Vec<String>>,
    pub vector: Option<Vec<f32>>,
    pub passages: usize,
}

/// One change in a batch.
#[derive(Clone, Copy, Debug)]
pub enum Op<'a> {
    /// Adds the document or replaces the one with its key.
    Put(&'a Doc),
    Delete(u64),
}

/// A search: words of `text` in `fields` (field, boost).
pub struct Request<'a> {
    pub text: &'a str,
    pub fields: &'a [(u8, f32)],
    /// Distinct items to return (each with its best passage).
    pub k: usize,
    /// Every word must match (in some field).
    pub all: bool,
    /// For words of 4+ letters, also this many of the most common longer words they start
    /// ("invoice" → "invoices"; German compounds): the recall prefix search gave FTS5.
    pub expand: usize,
    /// The last word is being typed: this many of its most common completions.
    pub prefix_last: usize,
    /// Only documents whose attributes pass this.
    pub filter: Option<&'a Filter>,
    /// Weighted terms for sparse fields (field, term, weight): scored by dot product with the
    /// documents' weights, added to the text's BM25. (`all` doesn't make them required.)
    pub sparse: &'a [(u8, String, f32)],
    /// Give up early when this says so (none: never).
    pub stop: Option<&'a crate::stop::Stop>,
}

/// A result. `score` is higher for a better one, whatever was searched: BM25 for text; for
/// vectors the cosine, the squared distance negated, or the dot product, by the index's
/// metric (`vector::Metric`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Found {
    pub key: u64,
    pub passage: u32,
    pub score: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub segments: usize,
    pub passages: usize,
    pub deleted: usize,
    pub memory: usize,
    pub merging: bool,
    /// Chunks in memory (writes not yet in a segment, merged two by two).
    pub chunks: usize,
    /// The newest batch's sequence number.
    pub seq: u64,
    /// Writes not yet in a segment, as logged (the active memtable and any being written out).
    pub memtable_bytes: usize,
    /// The log files on disk.
    pub log_bytes: u64,
    /// A memtable is being written out as a segment.
    pub flushing: bool,
    /// Segments a merge replaced that snapshots still hold (their files stay mapped until the
    /// last such snapshot is dropped).
    pub pinned_segments: usize,
    pub pinned_bytes: usize,
    /// Segments whose vector scan caches a search has built, and their memory (built on a
    /// segment's first approximate search, kept while it's open).
    pub vector_caches: usize,
    pub vector_cache_bytes: usize,
    /// Conditions whose matches segments keep from earlier queries (at most 32 a segment, a bit
    /// a document each), and their memory.
    pub kept_filters: usize,
    pub kept_filter_bytes: usize,
    /// Generations of centroids among the segments (segments of one are searched as one; more
    /// than one while segments of an earlier one wait to be merged), and how many centroids the
    /// one trained on the most vectors has (0: no segment has clusters yet).
    pub vector_generations: usize,
    pub vector_clusters: usize,
}

/// Why a vector search missed neighbours a reference ranking has (`View::nearest_misses`), and
/// the work it did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Misses {
    /// In a cluster the search didn't scan (more probing would have found them).
    pub probe: usize,
    /// In a scanned cluster, near enough to be returned, but skipped on their estimates (a looser
    /// bound or a deeper rerank would have kept them).
    pub prune: usize,
    /// Farther than the k-th result at the precision the index keeps (f16): the reference's
    /// precision, not the search, decides these, and an exact search misses them too.
    pub precision: usize,
    /// Codes estimated, and vectors compared at full precision.
    pub estimated: usize,
    pub reranked: usize,
}

/// `Misses`' mark for a segment compared exactly (its filter's matches, not clusters).
const EXACT: u32 = u32::MAX;

/// A filtered vector search compares a filter's matches exactly, rather than probing clusters,
/// when that is the cheaper: reading a row at full precision costs about what scanning this
/// many codes under the filter's bits does (a million vectors of 256 dimensions on an M1 Max:
/// 100 ns a row, where rows are far apart, against 3 ns a code), and how many codes a search
/// for k among the matches scans is what the segment's measured budget for that depth says.
const EXACT_CODES: usize = 32;

/// For segments without measured budgets (from before there were): up to this many matches, or
/// a 32nd of the segment, are compared exactly.
const EXACT_FILTERED: usize = 4096;

/// Up to this share of a segment's documents passing, a scan looks every block's documents up in
/// the filter's bits from its first list on and skips what doesn't pass; above it, that costs
/// more than it saves unless the top k's threshold rules little out (`vector::search_lists`).
const MASKED_SHARE: f64 = 1.0 / 16.0;

/// Vectors in segments below which searching them at once isn't worth handing work to threads.
const PARALLEL_VECTORS: usize = 10_000;

/// Threads for running a query's parts at once (a hybrid query's retrievers, segments' vectors),
/// apart from rayon's global pool, where a merge's k-means can hold every thread for a while.
pub(crate) fn query_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism().map_or(2, |n| n.get()).clamp(2, 8);
        rayon::ThreadPoolBuilder::new().num_threads(n).thread_name(|i| format!("hinny-query-{i}")).build().expect("query threads")
    })
}

/// What queries see: the index as of one batch, immutable. Cheap to take (`Index::view`,
/// `Reader::view`) and safe to keep and share between threads: a snapshot for as long as it's held.
pub struct View {
    seq: u64,
    fields: usize,
    dims: usize,
    /// How vectors are compared.
    metric: Metric,
    analyzers: Vec<Analyzer>,
    segments: Vec<Arc<SegEntry>>,
    chunks: Vec<Arc<Chunk>>,
    /// How many passages of each segment, then of each chunk, are deleted as this view sees them:
    /// counted when the view is made (everything stamped by then is at or before its sequence
    /// number), so a query never counts, however old the view and whatever was deleted since.
    deleted: Vec<usize>,
}

/// What `View::verify` read and found whole.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Verified {
    /// Segments read against their checksums.
    pub segments: usize,
    /// Segments from before there were checksums (format 7 and earlier): not checked. A merge
    /// (`optimize`) rewrites them with checksums.
    pub unchecked_segments: usize,
    /// Files of stored text read (each block against its checksum, where it has one).
    pub blobs: usize,
}

/// A handle for taking views from another thread, without the index itself.
#[derive(Clone)]
pub struct Reader(Arc<arc_swap::ArcSwap<View>>);

impl Reader {
    /// The newest view: every batch written so far, kept as long as it's held.
    pub fn view(&self) -> Arc<View> {
        self.0.load_full()
    }

    /// Makes `view` the newest: for a reader that has read the index's files again.
    pub fn adopt(&self, view: Arc<View>) {
        self.0.store(view);
    }

    /// The newest view for a moment (a query): no shared reference count to bump, so readers on
    /// many cores don't contend (RocksDB found one shared count capped read scaling at ~8 cores).
    pub fn current(&self) -> arc_swap::Guard<Arc<View>> {
        self.0.load()
    }
}

pub struct Index {
    dir: PathBuf,
    fields: usize,
    /// Vector dimensions (0 until the first vector).
    dims: usize,
    /// How vectors are compared: cosine unless `set_metric` said otherwise, while the index
    /// held no vectors. In the manifest, so a log replayed later stores its vectors the same way.
    metric: Metric,
    segments: Vec<Arc<SegEntry>>,
    /// Memtables, oldest first: the last is written to; one before it is frozen while a background
    /// thread writes it out as a segment (`flush`). Both are searched.
    mems: Vec<Memtable>,
    flush: Option<Flush>,
    /// The write-ahead log the active memtable writes to; none when read-only.
    wal: Option<BufWriter<File>>,
    /// A partial WAL append must never be followed by an acknowledged record behind its torn tail.
    wal_failed: bool,
    next_log: u64,
    /// The writer's exclusive lock (released when the index is dropped, after its last checkpoint).
    _lock: Option<File>,
    next_id: u64,
    /// What tells this index from every other, one of the same name in the same place later
    /// included: made up when it is created (and when it is forked: a fork is its own index),
    /// kept in the manifest. 0: read from a manifest that has none (before format 7).
    id: u128,
    merge: Option<Merge>,
    /// The newest batch's sequence number (1 before any: deletions loaded from disk are stamped 1).
    seq: u64,
    /// The current view, for readers.
    published: Arc<arc_swap::ArcSwap<View>>,
    /// While replaying a log, views are published once at the end.
    replaying: bool,
    /// Checkpoint after this many passages in memory.
    pub flush_passages: usize,
    /// Checkpoint after this many bytes in memory too (as logged), so a few large documents
    /// can't grow the memtable without bound: writes not yet in segments stay under about four
    /// times this (the active memtable and one being written out, each up to twice it).
    pub flush_bytes: usize,
    /// The largest batch `write` accepts, encoded as it's logged (bytes). Larger ones are refused
    /// before anything is logged or applied: split them, or raise this.
    pub max_batch_bytes: usize,
    /// Merge this many segments of a size tier into one.
    pub merge_factor: usize,
    /// Merging in the background (off: only when asked, for tests).
    pub auto_merge: bool,
    /// Keep passages' text, for `text` (in blob files; see `blob`). Off: only the index has it.
    pub store_text: bool,
    /// How each field's text becomes terms (recorded in the manifest; see `set_analyzers`).
    analyzers: Vec<Analyzer>,
    /// Segments merges replaced, for `stats` to count while snapshots still hold them.
    retired: Vec<std::sync::Weak<Segment>>,
}

const MANIFEST: &str = "manifest";
/// The index's format: what its manifest, log and segments are. 1: versions before 0.2 (a
/// manifest that says "fugu-text 1", a log without a header). 2: logs with a header, records
/// checked with FNV-1a. 3: log records checked with CRC32 (hardware on ARM and x86), and batches.
/// 4: the manifest ends with a CRC32 of itself, and deletion files carry a header and a CRC32
/// (raw ones from before still read, so an older index needs no rewrite). 5: segments of
/// version 8 (attributes as columns of codes, vectors in lists a generation shares, checksums),
/// and attributes that are floats, in the log and in segments. 6: attributes of several strings.
/// 7: deletion files numbered and named by the manifest. 8: vectors compared by another metric
/// than cosine, which the manifest names and segments of version 9 carry. This version reads
/// 1–8; anything newer is refused (see DESIGN.md, "Formats").
///
/// It writes 8 only for an index of such a metric. One whose vectors are cosine's holds nothing
/// a build that reads 7 would take for something else, so it is written as 7 (`COSINE_FORMAT`),
/// byte for byte what it was; under another metric the same build would normalize the log's
/// vectors when replaying them and score segments' as cosines, so it must refuse, and by this
/// number it does.
pub const FORMAT: u32 = 8;
/// The format of an index whose vectors are compared by cosine (or that has none), and of
/// every log: a log's records are what they were, vectors as they were given, and it is the
/// manifest that says how to take them.
const COSINE_FORMAT: u32 = 7;
/// `flush_bytes`' default.
pub const FLUSH_BYTES: usize = 256 << 20;

/// `max_batch_bytes`' default: a gibibyte, a quarter of what a log record's u32 length can say.
pub const MAX_BATCH_BYTES: usize = 1 << 30;

/// The log's header: these 8 bytes, then the format (u32).
const WAL_MAGIC: &[u8; 8] = b"fugu-log";
/// A deletion file's header: these 8 bytes and its version (u32), then the bitmap and a CRC32 of
/// all before it. Files from before format 4 are the bitmap alone.
const DEL_MAGIC: &[u8; 8] = b"fugu-del";
const DEL_VERSION: u32 = 1;

/// The error for files a newer hinny wrote.
pub(crate) fn too_new(what: &str, found: u32, reads: u32) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{what} is format {found}, written by a newer hinny; this one reads up to {reads}"))
}
/// Chunks stop merging at this many passages (a merge copies the older chunk: at 8,192, a few
/// milliseconds), so a 20,000-passage memtable has about 3 + log₂ 8,192 chunks at most.
const CHUNK_CAP: usize = 8192;
const WAL: &str = "wal";
const LOCK: &str = "lock";
const PUT: u8 = 1;
const DELETE: u8 = 2;
/// A batch: its changes, each a PUT or DELETE record's bytes with a length before it. One record,
/// so one checksum: a torn batch is dropped whole.
const BATCH: u8 = 3;

impl Index {
    /// Opens (or creates) the index in `dir` for writing, replaying its log. Only one process may
    /// write: if another has it open, this fails with `ErrorKind::WouldBlock`.
    pub fn open(dir: &Path, fields: usize) -> std::io::Result<Index> {
        Self::open_validated(dir, fields, || Ok(()))
    }

    /// Removes the index in `dir` and everything it holds. `WouldBlock` if it is open for
    /// writing anywhere, `NotFound` if there is no such directory. Renamed first to `aside` (a
    /// path beside it that nothing else uses), so the index is gone whole from that moment
    /// even if removing its files stops halfway.
    pub fn remove(dir: &Path, aside: &Path) -> std::io::Result<()> {
        let lock = OpenOptions::new().write(true).open(dir.join(LOCK));
        match &lock {
            Ok(lock) => match lock.try_lock() {
                Ok(()) => {}
                Err(std::fs::TryLockError::WouldBlock) => {
                    return Err(std::io::Error::new(std::io::ErrorKind::WouldBlock, format!("{} is open for writing (in another process, or by a handle in this one); close it first", dir.display())));
                }
                Err(std::fs::TryLockError::Error(e)) => return Err(std::io::Error::new(e.kind(), e.to_string())),
            },
            // No lock file: never opened for writing, or not there at all.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && dir.is_dir() => {}
            Err(e) => return Err(std::io::Error::new(e.kind(), format!("{}: {e}", dir.display()))),
        }
        let _ = std::fs::remove_dir_all(aside);
        std::fs::rename(dir, aside)?;
        if let Some(parent) = dir.parent() {
            sync_dir(parent)?;
        }
        drop(lock);
        std::fs::remove_dir_all(aside)
    }

    /// Validate namespace metadata under the writer lock, before replay or garbage collection.
    pub(crate) fn open_validated(dir: &Path, fields: usize, validate: impl FnOnce() -> std::io::Result<()>) -> std::io::Result<Index> {
        assert!(fields <= MAX_FIELDS);
        create_dir_all_synced(dir)?;
        let lock = OpenOptions::new().create(true).truncate(false).write(true).open(dir.join(LOCK))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(std::io::Error::new(std::io::ErrorKind::WouldBlock, format!("{} is already open for writing (in another process, or by another handle in this one)", dir.display())));
            }
            Err(std::fs::TryLockError::Error(e)) => return Err(e),
        }
        validate()?;
        let (segments, next_id, analyzers, id, metric) = read_manifest(dir)?;
        // (A manifest without one, from before format 7, gets it when it is next written.)
        let id = if id == 0 { new_id() } else { id };
        let logs = log_files(dir)?;
        let contents: Vec<Vec<u8>> = logs.iter().map(|(_, p)| std::fs::read(p)).collect::<std::io::Result<_>>()?;
        // Validate before cleanup or constructing an Index: its Drop would otherwise checkpoint
        // a partially replayed index when opening an unsupported WAL returns an error.
        let records = contents.iter().map(|c| log_records(c)).collect::<std::io::Result<Vec<_>>>()?;
        if !dir.join(MANIFEST).try_exists()? {
            // A first flush that crashes before publication must still have a manifest from
            // which recovery can distinguish its orphaned output from committed data.
            write_atomic(&dir.join(MANIFEST), sealed(format!("fugudb-index {COSINE_FORMAT}\nnext 1\nid {id:032x}\n")).as_bytes())?;
        }
        // Files no manifest names: a crash during a flush or merge left them. (Only the writer
        // gets here, so they aren't another process's work in progress.) Blob files count as
        // named while a named segment points into them.
        let named: Vec<u64> = segments.iter().map(|(id, _, _, _)| *id).collect();
        let blobs: std::collections::HashSet<u64> = segments.iter().flat_map(|(_, s, _, _)| s.blobs().map(|(id, _)| id)).collect();
        for entry in std::fs::read_dir(dir)?.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let id = name.strip_prefix("seg-").and_then(|r| r.split('.').next()).and_then(|n| n.parse::<u64>().ok());
            let unused_blob = crate::blob::id_of(&name).is_some_and(|id| !blobs.contains(&id));
            // A deletion file of a named segment that isn't the one the manifest names: written
            // before a manifest that never was, or left when the next one took its place.
            let old_deletions = name.ends_with(".del") && id.is_some_and(|id| segments.iter().any(|(i, _, _, del)| *i == id && entry.path() != del_path(dir, id, *del)));
            if name.ends_with(".tmp") || id.is_some_and(|id| !named.contains(&id)) || unused_blob || old_deletions {
                let _ = std::fs::remove_file(entry.path());
            }
        }
        // New changes go to a log of their own; the old ones go once their changes are in a segment.
        let next_log = logs.last().map_or(1, |(n, _)| n + 1);
        let current = dir.join(format!("{WAL}-{next_log}"));
        let mut wal = BufWriter::new(OpenOptions::new().create(true).append(true).open(&current)?);
        write_log_header(&mut wal)?;
        let mut paths: Vec<PathBuf> = logs.into_iter().map(|(_, p)| p).collect();
        paths.push(current);
        let mut index = Index::assemble(dir, fields, segments, next_id, Some(wal), Some(lock), paths, next_log + 1, analyzers, metric);
        index.id = id;
        index.replaying = true;
        for r in records {
            index.replay(r);
        }
        index.replaying = false;
        index.publish();
        Ok(index)
    }

    /// A read-only snapshot of the index in `dir`, as of now (empty if there's none yet). It
    /// changes no files and takes no lock, so it can sit beside a writer in another process.
    /// Writes and checkpoints on it fail with `ErrorKind::PermissionDenied`.
    pub fn open_read_only(dir: &Path, fields: usize) -> std::io::Result<Index> {
        assert!(fields <= MAX_FIELDS);
        // Manifest, segments and log must be of one moment: the writer renames a new manifest in
        // and then deletes the logs that are in segments, and deletes merged-away segments. If the
        // manifest changed while reading, or a file it named is gone, read again.
        let mut last_error = None;
        for _ in 0..50 {
            let before = std::fs::read(dir.join(MANIFEST)).ok();
            let (segments, next_id, analyzers, id, metric) = match read_manifest(dir) {
                Ok(r) => r,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    last_error = Some(e);
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
                Err(e) => return Err(e),
            };
            // A log deleted since the listing (its changes are in a segment now): read again.
            let Ok(logs) = log_files(dir).and_then(|l| l.into_iter().map(|(_, p)| std::fs::read(p)).collect::<std::io::Result<Vec<_>>>()) else { continue };
            if std::fs::read(dir.join(MANIFEST)).ok() != before {
                continue;
            }
            let mut index = Index::assemble(dir, fields, segments, next_id, None, None, Vec::new(), 0, analyzers, metric);
            index.id = id;
            index.auto_merge = false;
            index.replaying = true;
            for c in &logs {
                index.replay(log_records(c)?);
            }
            index.replaying = false;
            index.publish();
            return Ok(index);
        }
        Err(last_error.unwrap_or_else(|| std::io::Error::other(format!("{} kept changing while being opened", dir.display()))))
    }

    /// What's on disk now, as a number that changes whenever a writer changes the index (its
    /// manifest or log): a reader whose snapshot was taken at another value is out of date.
    pub fn on_disk(dir: &Path) -> u64 {
        let mut state = fnv(&std::fs::read(dir.join(MANIFEST)).unwrap_or_default()) as u64;
        for (n, p) in log_files(dir).unwrap_or_default() {
            let m = std::fs::metadata(p).ok();
            let modified = m.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos() as u64);
            state = state.rotate_left(13) ^ n ^ (m.map_or(0, |m| m.len()) << 32) ^ modified.rotate_left(17);
        }
        state
    }

    /// For tests: stops as a crash would, without the final checkpoint, releasing the lock as the
    /// operating system does when a process dies.
    #[doc(hidden)]
    pub fn simulate_crash(mut self) {
        drop(self._lock.take());
        std::mem::forget(self);
    }

    /// Whether this is a read-only snapshot.
    pub fn read_only(&self) -> bool {
        self.wal.is_none()
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(dir: &Path, fields: usize, segments: Vec<(u64, Segment, Bitmap, u64)>, next_id: u64, wal: Option<BufWriter<File>>, lock: Option<File>, logs: Vec<PathBuf>, next_log: u64, mut analyzers: Vec<Analyzer>, metric: Metric) -> Index {
        let dims = segments.iter().find_map(|(_, s, _, _)| s.vector_dims()).unwrap_or(0);
        analyzers.resize(fields, Analyzer::PLAIN);
        let segments: Vec<Arc<SegEntry>> = segments
            .into_iter()
            .map(|(id, segment, deleted, del)| {
                let deletes = Deletes::from_bitmap(&deleted, segment.docs as usize);
                let entry = SegEntry::new(id, segment, deletes, false);
                entry.del.store(del, std::sync::atomic::Ordering::Relaxed);
                entry.del_written.store(del, std::sync::atomic::Ordering::Relaxed);
                Arc::new(entry)
            })
            .collect();
        let empty = Arc::new(View { seq: 1, fields, dims, metric, analyzers: analyzers.clone(), segments: Vec::new(), chunks: Vec::new(), deleted: Vec::new() });
        let mut index = Index {
            dir: dir.to_path_buf(),
            fields,
            dims,
            metric,
            segments,
            mems: vec![Memtable { chunks: Vec::new(), logs, bytes: 0 }],
            flush: None,
            wal,
            wal_failed: false,
            next_log,
            _lock: lock,
            next_id,
            id: 0,
            merge: None,
            seq: 1,
            published: Arc::new(arc_swap::ArcSwap::new(empty)),
            replaying: false,
            flush_passages: 20_000,
            flush_bytes: FLUSH_BYTES,
            max_batch_bytes: MAX_BATCH_BYTES,
            merge_factor: 4,
            auto_merge: true,
            store_text: false,
            analyzers,
            retired: Vec::new(),
        };
        index.publish();
        index
    }

    /// Makes the index's current state the view readers get.
    fn publish(&mut self) {
        if self.replaying {
            return;
        }
        let chunks: Vec<Arc<Chunk>> = self.mems.iter().flat_map(|m| m.chunks.iter().cloned()).collect();
        let deleted = self.segments.iter().map(|s| s.deletes.count()).chain(chunks.iter().map(|c| c.deletes.count())).collect();
        let view = View { seq: self.seq, fields: self.fields, dims: self.dims, metric: self.metric, analyzers: self.analyzers.clone(), segments: self.segments.clone(), chunks, deleted };
        self.published.store(Arc::new(view));
    }

    /// The newest view (every batch written so far).
    pub fn view(&self) -> Arc<View> {
        self.published.load_full()
    }

    /// A handle that takes views without the index: for queries on other threads, which then
    /// never wait for this one's writes.
    pub fn reader(&self) -> Reader {
        Reader(self.published.clone())
    }

    /// How each field's text becomes terms.
    pub fn analyzers(&self) -> &[Analyzer] {
        &self.analyzers
    }

    /// Sets how fields are analyzed. Their analysis is part of what the index's segments mean, so
    /// it can only change while the index is empty; it's saved in the manifest at once, so a log
    /// replayed later is read the same way.
    pub fn set_analyzers(&mut self, mut analyzers: Vec<Analyzer>) -> std::io::Result<()> {
        analyzers.resize(self.fields, Analyzer::PLAIN);
        if analyzers == self.analyzers {
            return Ok(());
        }
        // An index made with an earlier stopword list, opened with today's default: it keeps its own.
        let holds_data = !self.segments.is_empty() || self.in_memory() > 0;
        if holds_data && analyzers.iter().zip(&self.analyzers).all(|(a, b)| a.same_but_revision(b)) {
            return Ok(());
        }
        if self.read_only() {
            return Err(read_only());
        }
        if !self.segments.is_empty() || self.in_memory() > 0 {
            let codes = |a: &[Analyzer]| a.iter().map(Analyzer::code).collect::<Vec<_>>().join(", ");
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("the index's fields are analyzed as {}; that can't change once it holds data (asked for {})", codes(&self.analyzers), codes(&analyzers))));
        }
        self.analyzers = analyzers;
        self.persist()?;
        self.publish();
        Ok(())
    }

    /// How the index's vectors are compared.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Sets how vectors are compared. A metric is part of what stored vectors are (cosine's are
    /// normalized, the others' as they came) and of what its segments' probe budgets were
    /// measured for, so it can only change while the index holds no vectors; it's saved in the
    /// manifest at once, so a log replayed later is read the same way.
    pub fn set_metric(&mut self, metric: Metric) -> std::io::Result<()> {
        if metric == self.metric {
            return Ok(());
        }
        if self.read_only() {
            return Err(read_only());
        }
        if self.dims != 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("the index's vectors are compared by {}; that can't change once it holds vectors (asked for {})", self.metric.name(), metric.name())));
        }
        let before = std::mem::replace(&mut self.metric, metric);
        if let Err(e) = self.persist() {
            self.metric = before;
            return Err(e);
        }
        self.publish();
        Ok(())
    }

    fn in_memory(&self) -> usize {
        self.mems.iter().map(Memtable::len).sum()
    }

    // MARK: Changes

    /// Replaces item `key` with these passages (one text per field each), without attributes.
    pub fn put(&mut self, key: u64, version: u64, passages: &[Vec<String>]) -> std::io::Result<()> {
        self.upsert(&Doc { key, version, passages: passages.to_vec(), attrs: Vec::new(), vector: None })
    }

    /// Replaces the document with `doc.key` (or adds it).
    pub fn upsert(&mut self, doc: &Doc) -> std::io::Result<()> {
        self.write(&[Op::Put(doc)])
    }

    pub fn delete(&mut self, key: u64) -> std::io::Result<()> {
        self.write(&[Op::Delete(key)])
    }

    /// Applies a batch: all of it or (after a crash mid-write) none of it, visible to queries at
    /// once, all at once. On return the WAL is in the OS's hands and survives a process crash.
    /// Call `sync` for a checked storage synchronization boundary, or `checkpoint` to also flush
    /// memory into segments. An OS crash or power loss before that boundary may lose writes.
    pub fn write(&mut self, ops: &[Op]) -> std::io::Result<()> {
        self.check_writable()?;
        let _expected = self.admit(ops)?;
        // An empty index takes its first vector's dimensions, which the rest of the batch must have.
        let mut dims = self.dims;
        for op in ops {
            if let Op::Put(doc) = op {
                if let Some(v) = &doc.vector {
                    if dims == 0 {
                        dims = v.len();
                    }
                    if v.len() != dims || v.is_empty() || !v.iter().all(|x| x.is_finite()) {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("vector of {} dimensions; this namespace has {}", v.len(), dims)));
                    }
                    // Kept as it comes, in half precision, which ends at 65,504: beyond it a
                    // component would be stored as infinity. (Cosine's are normalized first.)
                    if self.metric != Metric::Cosine && let Some(x) = v.iter().find(|x| x.abs() > crate::vector::LARGEST_COMPONENT) {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("document {} has a vector with a component of {x}; vectors compared by {} are kept as they come, and a component can be at most {} in size", doc.key, self.metric.name(), crate::vector::LARGEST_COMPONENT)));
                    }
                }
            }
        }
        if ops.is_empty() {
            return Ok(());
        }
        let record = if let [op] = ops { self.encode(op) } else {
            let mut r = vec![BATCH];
            r.extend_from_slice(&(ops.len() as u32).to_le_bytes());
            for op in ops {
                let e = self.encode(op);
                r.extend_from_slice(&(e.len() as u32).to_le_bytes());
                r.extend_from_slice(&e);
            }
            r
        };
        debug_assert_eq!(record.len(), _expected, "admit's size is the logged size");
        self.log(&record)?;
        let store = self.store_text;
        self.apply(ops.iter().map(|op| (*op, store)));
        self.mems.last_mut().unwrap().bytes += record.len();
        self.maybe_flush()
    }

    /// Refuses a batch that couldn't be logged faithfully, before anything is: every length must
    /// fit the width it's written with (u32 for texts, strings, vectors, passages and records,
    /// u16 for attribute names and counts), and the whole within `max_batch_bytes`. Returns the
    /// size it will be logged at.
    fn admit(&self, ops: &[Op]) -> std::io::Result<usize> {
        let refuse = |what: String| Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, what));
        let fits32 = |n: usize| n <= u32::MAX as usize;
        let mut total = if ops.len() > 1 { 5 + 4 * ops.len() } else { 0 };
        if !fits32(ops.len()) {
            return refuse(format!("a batch of {} changes; at most {}", ops.len(), u32::MAX));
        }
        for op in ops {
            total += match op {
                Op::Delete(_) => 9,
                Op::Put(doc) => {
                    if !fits32(doc.passages.len()) {
                        return refuse(format!("document {} has {} passages; at most {}", doc.key, doc.passages.len(), u32::MAX));
                    }
                    if doc.attrs.len() > u16::MAX as usize {
                        return refuse(format!("document {} has {} attributes; at most {}", doc.key, doc.attrs.len(), u16::MAX));
                    }
                    let mut n = 1 + 8 + 8 + 4 + 2 + 1 + 1;
                    for p in &doc.passages {
                        n += 1;
                        for text in p.iter().take(self.fields) {
                            if !fits32(text.len()) {
                                return refuse(format!("document {} has a text of {} bytes; at most {}", doc.key, text.len(), u32::MAX));
                            }
                            n += 4 + text.len();
                        }
                    }
                    for (name, value) in &doc.attrs {
                        if name.len() > u16::MAX as usize {
                            return refuse(format!("document {} has an attribute name of {} bytes; at most {}", doc.key, name.len(), u16::MAX));
                        }
                        n += 2 + name.len() + 1 + match value {
                            Value::Float(x) if x.is_nan() => return refuse(format!("document {} has an attribute {name:?} that is not a number", doc.key)),
                            // The one integer a column can't hold: it is how "no value" is kept
                            // (and it had been stored as that, silently).
                            Value::Int(crate::attrs::NO_INT) => return refuse(format!("document {} has an attribute {name:?} of -2^63, the one integer an attribute can't be", doc.key)),
                            Value::Int(_) | Value::Float(_) => 8,
                            Value::Str(t) if !fits32(t.len()) => return refuse(format!("document {} has a string attribute of {} bytes; at most {}", doc.key, t.len(), u32::MAX)),
                            Value::Str(t) => 4 + t.len(),
                            Value::Tags(tags) => {
                                if let Some(t) = tags.iter().find(|t| !fits32(t.len())) {
                                    return refuse(format!("document {} has a tag of {} bytes; at most {}", doc.key, t.len(), u32::MAX));
                                }
                                if !fits32(tags.len()) {
                                    return refuse(format!("document {} has {} tags in {name:?}; at most {}", doc.key, tags.len(), u32::MAX));
                                }
                                4 + tags.iter().map(|t| 4 + t.len()).sum::<usize>()
                            }
                        };
                    }
                    if let Some(v) = &doc.vector {
                        n += 4 + 4 * v.len();
                    }
                    n
                }
            };
            if total > self.max_batch_bytes || !fits32(total) {
                break;
            }
        }
        if total > self.max_batch_bytes || !fits32(total) {
            return refuse(format!("a batch of more than {} bytes as logged; at most {} (split it, or raise max_batch_bytes)", total, self.max_batch_bytes.min(u32::MAX as usize)));
        }
        Ok(total)
    }

    /// A change's log record (PUT or DELETE).
    fn encode(&self, op: &Op) -> Vec<u8> {
        let doc = match op {
            Op::Delete(key) => {
                let mut r = vec![DELETE];
                r.extend_from_slice(&key.to_le_bytes());
                return r;
            }
            Op::Put(doc) => doc,
        };
        let mut record = vec![PUT];
        record.extend_from_slice(&doc.key.to_le_bytes());
        record.extend_from_slice(&doc.version.to_le_bytes());
        record.extend_from_slice(&(doc.passages.len() as u32).to_le_bytes());
        for p in &doc.passages {
            record.push(p.len().min(self.fields) as u8);
            for text in p.iter().take(self.fields) {
                record.extend_from_slice(&(text.len() as u32).to_le_bytes());
                record.extend_from_slice(text.as_bytes());
            }
        }
        record.extend_from_slice(&(doc.attrs.len() as u16).to_le_bytes());
        for (name, value) in &doc.attrs {
            record.extend_from_slice(&(name.len() as u16).to_le_bytes());
            record.extend_from_slice(name.as_bytes());
            match value {
                Value::Int(x) => {
                    record.push(Kind::Int as u8);
                    record.extend_from_slice(&x.to_le_bytes());
                }
                Value::Float(x) => {
                    record.push(Kind::Float as u8);
                    record.extend_from_slice(&x.to_bits().to_le_bytes());
                }
                Value::Str(t) => {
                    record.push(Kind::Str as u8);
                    record.extend_from_slice(&(t.len() as u32).to_le_bytes());
                    record.extend_from_slice(t.as_bytes());
                }
                Value::Tags(tags) => {
                    record.push(Kind::Tags as u8);
                    record.extend_from_slice(&(tags.len() as u32).to_le_bytes());
                    for t in tags {
                        record.extend_from_slice(&(t.len() as u32).to_le_bytes());
                        record.extend_from_slice(t.as_bytes());
                    }
                }
            }
        }
        match &doc.vector {
            Some(v) => {
                record.push(1);
                record.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for x in v {
                    record.extend_from_slice(&x.to_le_bytes());
                }
            }
            None => record.push(0),
        }
        // Whether its text is stored: the log, not the opener, says (a reader replaying it doesn't
        // know how the writer was set up).
        record.push(self.store_text as u8);
        record
    }

    fn log(&mut self, payload: &[u8]) -> std::io::Result<()> {
        let wal = self.wal.as_mut().ok_or_else(read_only)?;
        let result = (|| {
            crate::faults::check("wal append")?;
            wal.write_all(&(payload.len() as u32).to_le_bytes())?;
            wal.write_all(&crc32fast::hash(payload).to_le_bytes())?;
            if let Err(e) = crate::faults::check("wal torn") {
                // As a crash mid-append leaves it: half the record on disk, then the error.
                wal.write_all(&payload[..payload.len() / 2])?;
                wal.flush()?;
                return Err(e);
            }
            wal.write_all(payload)?;
            wal.flush()
        })();
        if result.is_err() {
            self.wal_failed = true;
        }
        result
    }

    fn check_writable(&self) -> std::io::Result<()> {
        if self.read_only() {
            return Err(read_only());
        }
        if self.wal_failed {
            return Err(std::io::Error::other("a WAL I/O operation failed; close and reopen the index before writing or syncing"));
        }
        Ok(())
    }

    fn flush_wal(&mut self) -> std::io::Result<()> {
        self.check_writable()?;
        let result = crate::faults::check("wal flush").and_then(|()| self.wal.as_mut().unwrap().flush());
        if result.is_err() {
            self.wal_failed = true;
        }
        result
    }

    /// Applies a batch as the next sequence number: deletes stamp what they replace, puts go into
    /// a new chunk, and the result is published as one view.
    fn apply<'a>(&mut self, ops: impl Iterator<Item = (Op<'a>, bool)>) {
        self.seq += 1;
        let seq = self.seq;
        let mut mem = MemIndex::with_analyzers(self.fields, &self.analyzers);
        // This batch's own passages by key (a later change to a key in it replaces an earlier one).
        let mut mine: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut gone: Vec<u32> = Vec::new();
        for (op, store) in ops {
            let key = match op {
                Op::Put(d) => d.key,
                Op::Delete(k) => k,
            };
            self.stamp_deleted(key, seq);
            if let Some(docs) = mine.remove(&key) {
                gone.extend(docs);
            }
            let Op::Put(d) = op else { continue };
            if let Some(v) = &d.vector {
                if self.dims == 0 {
                    self.dims = v.len();
                }
            }
            let mut texts: Vec<&str> = Vec::with_capacity(self.fields);
            // A document without text is still stored (its attributes can be queried).
            let empty = vec![Vec::new()];
            let passages = if d.passages.is_empty() { &empty } else { &d.passages };
            for (i, p) in passages.iter().enumerate() {
                texts.clear();
                texts.extend(p.iter().map(String::as_str));
                // The first passage carries the document's vector.
                let vector = if i == 0 { d.vector.clone().and_then(|mut v| self.metric.prepare(&mut v).then_some(v)) } else { None };
                let doc = mem.add_full(d.key, d.version, i as u32, &texts, &d.attrs, vector);
                if store && p.iter().enumerate().any(|(f, t)| !t.is_empty() && !self.analyzers.get(f).is_some_and(|a| a.sparse)) {
                    // A sparse field's weighted terms aren't text to keep.
                    let kept: Vec<String> = p.iter().enumerate().map(|(f, t)| if self.analyzers.get(f).is_some_and(|a| a.sparse) { String::new() } else { t.clone() }).collect();
                    mem.texts[doc as usize] = Some(kept.into());
                }
                mine.entry(d.key).or_default().push(doc);
            }
        }
        if !mem.is_empty() {
            let mut chunk = Chunk::new(mem);
            for d in gone {
                chunk.deletes.mark(d);
            }
            self.add_chunk(chunk);
        }
        self.publish();
    }

    /// Stamps every live passage of `key` (in segments and memory) deleted as of `seq`.
    fn stamp_deleted(&mut self, key: u64, seq: u64) {
        for s in &self.segments {
            for d in s.segment.docs_with_key(key) {
                s.delete(d, seq);
            }
        }
        for m in &self.mems {
            for c in &m.chunks {
                if let Some(docs) = c.keys.get(&key) {
                    for &d in docs {
                        c.deletes.delete(d, seq);
                    }
                }
            }
        }
    }

    /// Adds a chunk to the active memtable, merging the newest two while the older is no bigger
    /// than twice the newer and together they're at most `CHUNK_CAP` passages: about log₂ n chunks
    /// for small ones, and no write ever copies a big one (merging copies the older). The chunks
    /// merged stay as they are in views already taken.
    fn add_chunk(&mut self, chunk: Chunk) {
        let chunks = &mut self.mems.last_mut().unwrap().chunks;
        chunks.push(Arc::new(chunk));
        while chunks.len() >= 2 {
            let n = chunks.len();
            let (older, newer) = (chunks[n - 2].index.len(), chunks[n - 1].index.len());
            if older > 2 * newer || older + newer > CHUNK_CAP {
                break;
            }
            let b = chunks.pop().unwrap();
            let a = chunks.pop().unwrap();
            chunks.push(Arc::new(Chunk::concat(&a, &b)));
        }
    }

    fn replay(&mut self, (log, format): (&[u8], u32)) {
        // Logs of format 3 on are checked with CRC32, older ones with FNV-1a.
        let check = |b: &[u8]| if format >= 3 { crc32fast::hash(b) } else { fnv(b) };
        let mut at = 0;
        while at + 8 <= log.len() {
            let len = u32::from_le_bytes(log[at..at + 4].try_into().unwrap()) as usize;
            let sum = u32::from_le_bytes(log[at + 4..at + 8].try_into().unwrap());
            // A torn last record (crash mid-write) ends the log.
            if at + 8 + len > log.len() || check(&log[at + 8..at + 8 + len]) != sum {
                break;
            }
            let r = &log[at + 8..at + 8 + len];
            at += 8 + len;
            let mut changes: Vec<(Decoded, bool)> = Vec::new();
            if r.first() == Some(&BATCH) {
                let n = u32::from_le_bytes(r[1..5].try_into().unwrap()) as usize;
                let mut p = 5;
                for _ in 0..n {
                    let l = u32::from_le_bytes(r[p..p + 4].try_into().unwrap()) as usize;
                    match decode(&r[p + 4..p + 4 + l]) {
                        Some(c) => changes.push(c),
                        None => return,
                    }
                    p += 4 + l;
                }
            } else {
                match decode(r) {
                    Some(c) => changes.push(c),
                    None => break,
                }
            }
            let ops: Vec<(Op, bool)> = changes
                .iter()
                .map(|(c, store)| {
                    (
                        match c {
                            Decoded::Put(d) => Op::Put(d),
                            Decoded::Delete(k) => Op::Delete(*k),
                        },
                        *store,
                    )
                })
                .collect();
            self.apply(ops.into_iter());
            self.mems.last_mut().unwrap().bytes += len;
        }
    }

    // MARK: Checkpoints and merges

    /// Synchronizes every WAL needed to recover the current writes, without building segments.
    /// Buffered writes are visible immediately; this is their explicit durability boundary.
    /// Uses file synchronization and, on Unix, directory synchronization. Durability depends on
    /// the filesystem and device honoring those requests; non-Unix directory entries are not
    /// synchronized. A read-only index cannot sync. Errors must be handled by the caller.
    pub fn sync(&mut self) -> std::io::Result<()> {
        self.flush_wal()?;
        // A frozen or replayed WAL can hold acknowledged writes absent from the manifest.
        // Synchronizing just the current WAL would not cover them.
        for mem in &self.mems {
            for path in &mem.logs {
                crate::faults::check("wal sync")?;
                OpenOptions::new().write(true).open(path)?.sync_all()?;
            }
        }
        sync_dir(&self.dir)
    }

    /// Writes what's in memory as a segment and waits for it, saves deletions and the manifest,
    /// and deletes the logs that are in segments now; then applies a finished merge or starts one
    /// if a tier is full. (Filling memory does this in the background.)
    pub fn checkpoint(&mut self) -> std::io::Result<()> {
        self.check_writable()?;
        self.finish_flush()?;
        if self.mems.last().unwrap().len() > 0 {
            self.freeze()?;
            self.finish_flush()?;
        } else {
            self.persist()?;
            // Nothing in memory: logs before the current one hold only what segments have.
            let active = self.mems.last_mut().unwrap();
            let current = active.logs.pop();
            let retired = active.logs.drain(..).collect::<Vec<_>>();
            active.logs.extend(current);
            let dir = self.dir.clone();
            self.retire_logs(retired, |p| std::fs::remove_file(p).and_then(|()| sync_dir(&dir)))?;
        }
        self.sync()?;
        self.poll_merge()?;
        if self.auto_merge {
            self.start_merge(false);
        }
        Ok(())
    }

    /// Deletes every document that passes `filter` now, as one batch (all of them or, if it
    /// fails, none); how many that was. Found from the filter's columns, as a count is.
    pub fn delete_where(&mut self, filter: &Filter) -> std::io::Result<u64> {
        self.check_writable()?;
        let keys = self.view().keys(filter);
        if !keys.is_empty() {
            self.write(&keys.iter().map(|&k| Op::Delete(k)).collect::<Vec<_>>())?;
        }
        Ok(keys.len() as u64)
    }

    /// What a copy of this index is made of, as of now: the manifest's bytes and the names of
    /// the files it names (segments, their deletion files, files of stored text), each with its
    /// size. What is in memory is written out first. None of the named files is ever written
    /// again, so they can be copied at leisure; one that has gone meanwhile (merged away) means
    /// the manifest has moved on, and the copy starts over from the new one.
    pub fn files(&mut self) -> std::io::Result<(Vec<u8>, Vec<(String, u64)>)> {
        self.check_writable()?;
        self.checkpoint()?;
        let mut paths: Vec<PathBuf> = Vec::new();
        let mut blobs: Vec<u64> = self.segments.iter().flat_map(|s| s.segment.blobs().map(|(id, _)| id)).collect();
        blobs.sort_unstable();
        blobs.dedup();
        paths.extend(blobs.into_iter().map(|id| crate::blob::path(&self.dir, id)));
        for s in &self.segments {
            paths.push(seg_path(&self.dir, s.id));
            let deletions = del_path(&self.dir, s.id, s.del.load(std::sync::atomic::Ordering::Relaxed));
            if deletions.exists() {
                paths.push(deletions);
            }
        }
        let files = paths.iter().map(|p| Ok((p.file_name().unwrap_or_default().to_string_lossy().into_owned(), std::fs::metadata(p)?.len()))).collect::<std::io::Result<Vec<_>>>()?;
        Ok((std::fs::read(self.dir.join(MANIFEST))?, files))
    }

    /// Makes `to` (a directory that isn't there yet, or is empty) an index of its own holding
    /// what this one holds now: every write acknowledged before the call, and none after.
    /// From then on the two are apart: a write, a delete or a merge in one changes nothing in
    /// the other, and either can be deleted.
    ///
    /// Nothing is copied where the filesystem can link: a segment and a file of stored text are
    /// never changed once written, and a deletion file and the manifest are replaced whole, so
    /// both indexes can name the same file until one of them merges it away (which removes its
    /// own name for it, not the file). What's in memory is written out first (a checkpoint), so
    /// a fork costs that plus a link per file: milliseconds, whatever the index's size. Where
    /// links can't be made (another filesystem, one without them), the files are copied.
    ///
    /// The directory is complete once its manifest is there, which is written last.
    pub fn fork(&mut self, to: &Path) -> std::io::Result<()> {
        self.check_writable()?;
        if std::fs::read_dir(to).is_ok_and(|mut entries| entries.next().is_some()) {
            return Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, format!("{} is there already, and not empty", to.display())));
        }
        self.checkpoint()?;
        create_dir_all_synced(to)?;
        let place = |name: PathBuf| -> std::io::Result<()> {
            let (from, into) = (self.dir.join(&name), to.join(&name));
            crate::faults::check("fork link")?;
            match std::fs::hard_link(&from, &into) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound || e.kind() == std::io::ErrorKind::AlreadyExists => Err(e),
                // No link to be had (across filesystems, or one that has none): a copy.
                Err(_) => std::fs::copy(&from, &into).and_then(|_| File::open(&into)?.sync_all()),
            }
        };
        let name = |path: PathBuf| PathBuf::from(path.file_name().unwrap_or_default());
        let mut blobs: Vec<u64> = self.segments.iter().flat_map(|s| s.segment.blobs().map(|(id, _)| id)).collect();
        blobs.sort_unstable();
        blobs.dedup();
        for id in blobs {
            place(name(crate::blob::path(&self.dir, id)))?;
        }
        for s in &self.segments {
            place(name(seg_path(&self.dir, s.id)))?;
            let deletions = del_path(&self.dir, s.id, s.del.load(std::sync::atomic::Ordering::Relaxed));
            if deletions.exists() {
                place(name(deletions))?;
            }
        }
        // The manifest as this index's is (everything is saved: the checkpoint above), under an
        // identity of its own: a fork is another index from here on.
        let mut manifest = self.manifest_head(new_id());
        for s in &self.segments {
            let del = s.del.load(std::sync::atomic::Ordering::Relaxed);
            manifest.push_str(&if del == 0 { format!("segment {}\n", s.id) } else { format!("segment {} {del}\n", s.id) });
        }
        write_atomic(&to.join(MANIFEST), sealed(manifest).as_bytes())
    }

    /// A manifest's lines before its segments, for an index of identity `id`: as format 7
    /// wrote them while the metric is cosine, else as format 8 with the metric named (`FORMAT`).
    fn manifest_head(&self, id: u128) -> String {
        let analysis = self.analyzers.iter().map(Analyzer::code).collect::<Vec<_>>().join(" ");
        match self.metric {
            Metric::Cosine => format!("fugudb-index {COSINE_FORMAT}\nnext {}\nid {id:032x}\nanalysis {analysis}\n", self.next_id),
            metric => format!("fugudb-index {FORMAT}\nnext {}\nid {id:032x}\nanalysis {analysis}\nmetric {}\n", self.next_id, metric.name()),
        }
    }

    /// What tells this index from every other (see the field), as it is written: 32 hex
    /// digits; empty for one read from a manifest that has none.
    pub fn id(&self) -> String {
        if self.id == 0 { String::new() } else { format!("{:032x}", self.id) }
    }

    /// After a write: installs a finished background flush, and freezes the memtable once it's
    /// full. If the previous one is still being written when this one is twice full, waits for
    /// it (memory stays bounded).
    fn maybe_flush(&mut self) -> std::io::Result<()> {
        if self.flush.as_ref().is_some_and(|f| f.handle.is_finished()) {
            self.finish_flush()?;
        }
        let mem = self.mems.last().unwrap();
        let (active, bytes) = (mem.len(), mem.bytes);
        if active < self.flush_passages && bytes < self.flush_bytes {
            return Ok(());
        }
        if self.mems.len() > 1 {
            if active < 2 * self.flush_passages && bytes < 2 * self.flush_bytes {
                return Ok(());
            }
            self.finish_flush()?;
        }
        self.freeze()
    }

    /// Freezes the active memtable (writes go to a new one, with a log of its own) and starts
    /// writing it out in the background.
    fn freeze(&mut self) -> std::io::Result<()> {
        let n = self.next_log;
        self.next_log += 1;
        let path = self.dir.join(format!("{WAL}-{n}"));
        let mut wal = BufWriter::new(OpenOptions::new().create(true).append(true).open(&path)?);
        write_log_header(&mut wal)?;
        self.flush_wal()?;
        self.wal = Some(wal);
        self.mems.push(Memtable { chunks: Vec::new(), logs: vec![path], bytes: 0 });
        self.start_flush()
    }

    /// Writes the oldest memtable out as a segment, on a thread of its own. Queries keep
    /// searching it in memory meanwhile, and deletes still reach it.
    fn start_flush(&mut self) -> std::io::Result<()> {
        if self.flush.is_some() || self.mems.len() < 2 {
            return Ok(());
        }
        let id = self.next_id;
        self.next_id += 1;
        let frozen: Vec<Arc<Chunk>> = self.mems[0].chunks.clone();
        let path = seg_path(&self.dir, id);
        let (generation, total, metric) = (self.generation(), self.vectors(), self.metric);
        let handle = std::thread::Builder::new()
            .name("hinny-flush".into())
            .spawn(move || {
                let parts: Vec<&MemIndex> = frozen.iter().map(|c| &c.index).collect();
                build_pool(false).install(|| segment::write_chunks_with(&path, &parts, id, generation.as_deref(), total, metric))
            })?;
        self.flush = Some(Flush { id, handle });
        Ok(())
    }

    /// Waits for a background flush and puts its segment in place of the frozen memtable: with
    /// the deletions that reached it meanwhile, named in the manifest, its logs deleted.
    fn finish_flush(&mut self) -> std::io::Result<()> {
        // A flush that failed earlier: the frozen memtable is still here; try again.
        if self.flush.is_none() && self.mems.len() > 1 {
            self.start_flush()?;
        }
        let Some(f) = self.flush.take() else { return Ok(()) };
        let path = seg_path(&self.dir, f.id);
        match f.handle.join() {
            Ok(Ok(())) => {}
            failed => {
                let _ = std::fs::remove_file(&path);
                let _ = std::fs::remove_file(crate::blob::path(&self.dir, f.id));
                return Err(match failed {
                    Ok(Err(e)) => e,
                    _ => std::io::Error::other("writing a segment panicked"),
                });
            }
        }
        let segment = Segment::open(&path)?;
        let frozen = &self.mems[0];
        // Deletions so far, including those that reached the frozen chunks during the flush: the
        // segment is seen only by views from after it's installed.
        let mut deletes = Deletes::new(segment.docs as usize);
        let mut d = 0u32;
        for c in &frozen.chunks {
            for i in 0..c.index.len() as u32 {
                if c.deletes.deleted(i) {
                    deletes.mark(d);
                }
                d += 1;
            }
        }
        let dirty = deletes.count() > 0;
        self.segments.push(Arc::new(SegEntry::new(f.id, segment, deletes, dirty)));
        if let Err(e) = self.persist() {
            self.segments.pop();
            // Keep the frozen memory/logs for retry. The manifest rename may have succeeded
            // before a directory-sync error, so leave the candidate files on disk as well.
            return Err(e);
        }
        let frozen = self.mems.remove(0);
        self.publish();
        let dir = self.dir.clone();
        self.retire_logs(frozen.logs, |p| std::fs::remove_file(p).and_then(|()| sync_dir(&dir)))?;
        if self.auto_merge {
            self.poll_merge()?;
            self.start_merge(false);
            // Flushes that outrun merges wait for them: a tier with twice what a merge takes is a
            // merge behind, and every segment more is one more for each search to go through.
            // (A flush assigns its vectors to centroids there already are, so it takes a
            // fraction of what it did when it clustered them, and writing no longer waits on
            // it long enough for merges to keep up by themselves: a million vectors written
            // flat out left 31 segments.)
            while self.tiers().values().any(|ids| ids.len() >= 2 * self.merge_factor) {
                if self.merge.is_none() && !self.start_merge(false) {
                    break;
                }
                self.wait_merge()?;
                self.start_merge(false);
            }
        }
        Ok(())
    }

    /// The segments by size tier (each `merge_factor` times the one before).
    fn tiers(&self) -> HashMap<u32, Vec<u64>> {
        let mut by_tier: HashMap<u32, Vec<u64>> = HashMap::new();
        for s in &self.segments {
            by_tier.entry(((s.segment.docs.max(4096) as f64 / 4096.0).log(self.merge_factor.max(2) as f64)) as u32).or_default().push(s.id);
        }
        by_tier
    }

    /// Waits for the flush and the merges under way, and starts and waits for the merges the
    /// policy still wants, until it wants none: the segments as writing leaves them once it
    /// pauses. (Merges are started and finished by writes; nothing merges while nothing writes.)
    pub fn settle(&mut self) -> std::io::Result<()> {
        loop {
            self.finish_flush()?;
            self.wait_merge()?;
            if !self.auto_merge || !self.start_merge(false) {
                return Ok(());
            }
        }
    }

    /// Persist each unlink before attempting the next one, leaving a suffix of the WAL history
    /// after a crash. Removing a newer log while an older one survives can replay stale values
    /// over the committed segment. Any failure prevents subsequent writes or retirement.
    fn retire_logs(&mut self, logs: Vec<PathBuf>, mut remove_and_sync: impl FnMut(&Path) -> std::io::Result<()>) -> std::io::Result<()> {
        for path in logs {
            if let Err(e) = crate::faults::check("wal remove").and_then(|()| remove_and_sync(&path)) {
                self.wal_failed = true;
                return Err(e);
            }
        }
        Ok(())
    }

    /// Saves changed deletion bitmaps and the manifest (atomically: written whole, then renamed).
    fn persist(&mut self) -> std::io::Result<()> {
        // A bitmap can delete a previously durable version in favor of a replacement in *any*
        // active WAL. The WAL must be durable before either that deletion or a manifest is.
        self.sync()?;
        use std::sync::atomic::Ordering::Relaxed;
        // A segment's changed deletions go into a file of the next number, which the manifest
        // then names: no file a manifest names is ever written again, so a manifest and the
        // files it names are one state of the index, whoever reads or copies them and whenever.
        // The file before goes once the manifest that no longer names it is in place.
        let mut manifest = self.manifest_head(self.id);
        let mut renamed: Vec<(&SegEntry, u64)> = Vec::new();
        for s in &self.segments {
            let mut del = s.del.load(Relaxed);
            if s.dirty.load(Relaxed) {
                // Above every number tried before, so a save that failed halfway (its manifest
                // may be in place) never has its file written over.
                del = s.del_written.load(Relaxed).max(del) + 1;
                s.del_written.store(del, Relaxed);
                write_atomic(&del_path(&self.dir, s.id, del), &deletion_file(&s.deletes.to_bitmap()))?;
                renamed.push((s, del));
            }
            manifest.push_str(&if del == 0 { format!("segment {}\n", s.id) } else { format!("segment {} {del}\n", s.id) });
        }
        write_atomic(&self.dir.join(MANIFEST), sealed(manifest).as_bytes())?;
        for (s, del) in renamed {
            let before = s.del.swap(del, Relaxed);
            s.dirty.store(false, Relaxed);
            // The one before, those of saves that failed between, and one from before format 7.
            for number in (before..del).chain([0]) {
                let _ = std::fs::remove_file(del_path(&self.dir, s.id, number));
            }
        }
        Ok(())
    }

    /// The namespace's generation of centroids, as the segment that carries the one trained on
    /// the most vectors (the newest of those): what flushes and merges build their vectors' lists
    /// into, until one of them has outgrown it and trains the next (`vector::build`). Segments
    /// of an earlier generation move to it when they're merged.
    fn generation(&self) -> Option<Arc<Segment>> {
        self.segments.iter().filter_map(|s| s.segment.vector_generation().map(|g| ((g.trained_on, g.id), &s.segment))).max_by_key(|(order, _)| *order).map(|(_, s)| s.clone())
    }

    /// About how many vectors the namespace holds (those in segments, deleted or not, and in
    /// memory): what a generation of centroids is sized for.
    fn vectors(&self) -> usize {
        self.segments.iter().map(|s| s.segment.vector_count()).sum::<usize>() + self.mems.iter().flat_map(|m| &m.chunks).map(|c| c.index.vectors.iter().flatten().count()).sum::<usize>()
    }

    /// Starts a background merge if a size tier has `merge_factor` segments (or, with `force`,
    /// of every segment), or a segment is mostly deleted.
    pub fn start_merge(&mut self, force: bool) -> bool {
        if self.read_only() || self.merge.is_some() || self.segments.is_empty() {
            return false;
        }
        let deleted = |s: &SegEntry| s.deletes.count();
        let mut pick: Vec<u64> = Vec::new();
        if force {
            if self.segments.len() > 1 || self.segments.iter().any(|s| deleted(s) > 0) {
                pick = self.segments.iter().map(|s| s.id).collect();
            }
        } else {
            if let Some((_, ids)) = self.tiers().into_iter().filter(|(_, ids)| ids.len() >= self.merge_factor).min_by_key(|(t, _)| *t) {
                pick = ids.into_iter().take(self.merge_factor * 2).collect();
            } else if let Some(s) = self.segments.iter().find(|s| s.segment.docs >= 1000 && deleted(s) * 10 > s.segment.docs as usize * 3) {
                pick = vec![s.id];
            }
        }
        if pick.is_empty() {
            return false;
        }
        let chosen: Vec<&Arc<SegEntry>> = self.segments.iter().filter(|s| pick.contains(&s.id)).collect();
        let pick: Vec<u64> = chosen.iter().map(|s| s.id).collect();
        let dropped: Vec<Bitmap> = chosen.iter().map(|s| s.deletes.to_bitmap()).collect();
        let inputs: Vec<(Arc<Segment>, Bitmap)> = chosen.iter().zip(&dropped).map(|(s, b)| (s.segment.clone(), b.clone())).collect();
        let rewrite = self.blobs_to_rewrite(force);
        let output = self.next_id;
        self.next_id += 1;
        let path = seg_path(&self.dir, output);
        let (generation, total) = (self.generation(), self.vectors());
        let handle = std::thread::Builder::new()
            .name("hinny-text-merge".into())
            .spawn(move || {
                background_priority();
                let refs: Vec<(&Segment, &Bitmap)> = inputs.iter().map(|(s, d)| (s.as_ref(), d)).collect();
                build_pool(true).install(|| segment::merge(&path, &refs, &rewrite, output, generation.as_deref(), total))
            })
            .expect("merge thread");
        self.merge = Some(Merge { inputs: pick, output, dropped, handle });
        true
    }

    /// Puts a finished merge in place of its inputs. Returns whether one was applied.
    pub fn poll_merge(&mut self) -> std::io::Result<bool> {
        if !self.merge.as_ref().is_some_and(|m| m.handle.is_finished()) {
            return Ok(false);
        }
        self.finish_merge()
    }

    /// Waits for a running merge and applies it.
    pub fn wait_merge(&mut self) -> std::io::Result<bool> {
        if self.merge.is_none() {
            return Ok(false);
        }
        self.finish_merge()
    }

    fn finish_merge(&mut self) -> std::io::Result<bool> {
        let m = self.merge.take().unwrap();
        let path = seg_path(&self.dir, m.output);
        match m.handle.join() {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                let _ = std::fs::remove_file(&path);
                return Err(e);
            }
            Err(_) => return Err(std::io::Error::other("merge panicked")),
        }
        let segment = Segment::open(&path)?;
        // The merged passages' deletions now, including those that landed during the merge: the
        // inputs' stamps, through the merge's renumbering (inputs in order, minus what it dropped).
        let mut deletes = Deletes::new(segment.docs as usize);
        let mut new = 0u32;
        let inputs: Vec<&Arc<SegEntry>> = m.inputs.iter().filter_map(|id| self.segments.iter().find(|s| s.id == *id)).collect();
        for (s, dropped) in inputs.iter().zip(&m.dropped) {
            for d in 0..s.segment.docs {
                if dropped.get(d) {
                    continue;
                }
                if s.deletes.deleted(d) {
                    deletes.mark(new);
                }
                new += 1;
            }
        }
        let dirty = deletes.count() > 0;
        let previous = self.segments.clone();
        let at = self.segments.iter().position(|s| m.inputs.contains(&s.id)).unwrap_or(self.segments.len());
        self.segments.retain(|s| !m.inputs.contains(&s.id));
        let merged = Arc::new(SegEntry::new(m.output, segment, deletes, dirty));
        // An empty result (everything deleted) needn't stay.
        if merged.segment.docs > 0 {
            self.segments.insert(at.min(self.segments.len()), merged);
        }
        if let Err(e) = self.persist() {
            self.segments = previous;
            // Neither set of files is disposable until publication is known to have succeeded.
            return Err(e);
        }
        self.publish();
        // Snapshots from before keep the inputs mapped: count them until they're let go.
        self.retired.retain(|w| w.strong_count() > 0);
        self.retired.extend(previous.iter().filter(|e| m.inputs.contains(&e.id)).map(|e| Arc::downgrade(&e.segment)));
        // The inputs' files, with the deletion file each had last.
        let previous_inputs: Vec<(u64, u64)> = previous.iter().filter(|e| m.inputs.contains(&e.id)).map(|e| (e.id, e.del.load(std::sync::atomic::Ordering::Relaxed))).collect();
        drop(previous);
        for e in previous_inputs {
            let _ = std::fs::remove_file(seg_path(&self.dir, e.0));
            let _ = std::fs::remove_file(del_path(&self.dir, e.0, e.1));
            let _ = std::fs::remove_file(del_path(&self.dir, e.0, 0));
        }
        if self.segments.iter().all(|s| s.id != m.output) {
            let _ = std::fs::remove_file(&path);
        }
        self.remove_unused_blobs();
        Ok(true)
    }

    /// Blob files whose text a merge should move: those with at least half their records dead
    /// (deleted, or replaced), or with `all`, any with a dead one. Text in other blob files stays
    /// put, and merges copy only references to it.
    fn blobs_to_rewrite(&self, all: bool) -> std::collections::HashSet<u64> {
        let mut live: HashMap<u64, (u64, u64)> = HashMap::new(); // id → (live records, records)
        for s in &self.segments {
            for ((id, records), here) in s.segment.blobs().zip(s.live_text()) {
                live.entry(id).or_insert((0, records)).0 += here.load(std::sync::atomic::Ordering::Relaxed);
            }
        }
        live.into_iter().filter(|(_, (l, n))| if all { l < n } else { l * 2 <= *n }).map(|(id, _)| id).collect()
    }

    /// Deletes blob files no segment points into any more. (A view that has one mapped keeps
    /// reading it.)
    fn remove_unused_blobs(&self) {
        let used: std::collections::HashSet<u64> = self.segments.iter().flat_map(|s| s.segment.blobs().map(|(id, _)| id)).collect();
        for entry in std::fs::read_dir(&self.dir).into_iter().flatten().flatten() {
            if let Some(id) = crate::blob::id_of(&entry.file_name().to_string_lossy()) {
                if !used.contains(&id) {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }

    // MARK: Reading (the newest view)

    pub fn stats(&self) -> Stats {
        let mut s = self.view().stats();
        s.merging = self.merge.is_some();
        s.flushing = self.flush.is_some();
        s.memtable_bytes = self.mems.iter().map(|m| m.bytes).sum();
        s.log_bytes = self.mems.iter().flat_map(|m| &m.logs).filter_map(|p| std::fs::metadata(p).ok()).map(|m| m.len()).sum();
        let pinned: Vec<Arc<Segment>> = self.retired.iter().filter_map(std::sync::Weak::upgrade).collect();
        s.pinned_segments = pinned.len();
        s.pinned_bytes = pinned.iter().map(|g| g.bytes().len()).sum();
        s
    }

    pub fn versions(&self) -> HashMap<u64, u64> {
        self.view().versions()
    }

    pub fn without_vectors(&self) -> Vec<u64> {
        self.view().without_vectors()
    }

    pub fn term_weights(&self, text: &str, fields: &[(u8, f32)]) -> Vec<(u8, String, f32)> {
        self.view().term_weights(text, fields)
    }

    pub fn search(&self, r: &Request) -> Vec<Found> {
        self.view().search(r)
    }

    pub fn query(&self, filter: Option<&Filter>, order_by: Option<(&str, bool)>, limit: usize) -> Vec<Found> {
        self.view().query(filter, order_by, limit)
    }

    pub fn get(&self, key: u64) -> Option<Vec<(String, Value)>> {
        self.view().get(key)
    }

    pub fn count(&self, filter: Option<&Filter>) -> u64 {
        self.view().count(filter)
    }

    pub fn count_by(&self, name: &str, filter: Option<&Filter>, limit: usize) -> Vec<(Value, u64)> {
        self.view().count_by(name, filter, limit)
    }

    pub fn text(&self, key: u64) -> Option<Vec<String>> {
        self.view().text(key)
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn nearest(&self, q: &[f32], k: usize, filter: Option<&Filter>, nprobe: usize) -> Vec<Found> {
        self.view().nearest(q, k, filter, nprobe)
    }

    pub fn nearest_with_stats(&self, q: &[f32], k: usize, filter: Option<&Filter>, nprobe: usize) -> (Vec<Found>, usize, usize) {
        self.view().nearest_with_stats(q, k, filter, nprobe)
    }

    pub fn nearest_exact(&self, q: &[f32], k: usize, filter: Option<&Filter>) -> Vec<Found> {
        self.view().nearest_exact(q, k, filter)
    }

    pub fn nearest_misses(&self, q: &[f32], k: usize, filter: Option<&Filter>, nprobe: usize, truth: &[u64]) -> (Vec<Found>, Misses) {
        self.view().nearest_misses(q, k, filter, nprobe, truth)
    }
}

/// A log record's change, decoded.
enum Decoded {
    Put(Doc),
    Delete(u64),
}

/// A PUT or DELETE record (and whether a PUT's text is stored); none if it isn't one.
fn decode(r: &[u8]) -> Option<(Decoded, bool)> {
    let u64_at = |i: usize| u64::from_le_bytes(r[i..i + 8].try_into().unwrap());
    match r.first()? {
        &PUT => {
            let (key, version) = (u64_at(1), u64_at(9));
            let n = u32::from_le_bytes(r[17..21].try_into().unwrap()) as usize;
            let mut p = 21;
            let mut passages = Vec::with_capacity(n);
            for _ in 0..n {
                let fields = r[p] as usize;
                p += 1;
                let mut texts = Vec::with_capacity(fields);
                for _ in 0..fields {
                    let l = u32::from_le_bytes(r[p..p + 4].try_into().unwrap()) as usize;
                    texts.push(String::from_utf8_lossy(&r[p + 4..p + 4 + l]).into_owned());
                    p += 4 + l;
                }
                passages.push(texts);
            }
            let mut attrs = Vec::new();
            if p + 2 <= r.len() {
                let n = u16::from_le_bytes([r[p], r[p + 1]]) as usize;
                p += 2;
                for _ in 0..n {
                    let l = u16::from_le_bytes([r[p], r[p + 1]]) as usize;
                    let name = String::from_utf8_lossy(&r[p + 2..p + 2 + l]).into_owned();
                    p += 2 + l;
                    let kind = r[p];
                    p += 1;
                    if kind == Kind::Int as u8 {
                        attrs.push((name, Value::Int(u64_at(p) as i64)));
                        p += 8;
                    } else if kind == Kind::Float as u8 {
                        attrs.push((name, Value::Float(f64::from_bits(u64_at(p)))));
                        p += 8;
                    } else if kind == Kind::Tags as u8 {
                        let n = u32::from_le_bytes(r[p..p + 4].try_into().unwrap()) as usize;
                        p += 4;
                        let mut tags = Vec::with_capacity(n.min(1024));
                        for _ in 0..n {
                            let l = u32::from_le_bytes(r[p..p + 4].try_into().unwrap()) as usize;
                            tags.push(String::from_utf8_lossy(&r[p + 4..p + 4 + l]).into_owned());
                            p += 4 + l;
                        }
                        attrs.push((name, Value::Tags(tags)));
                    } else {
                        let l = u32::from_le_bytes(r[p..p + 4].try_into().unwrap()) as usize;
                        attrs.push((name, Value::Str(String::from_utf8_lossy(&r[p + 4..p + 4 + l]).into_owned())));
                        p += 4 + l;
                    }
                }
            }
            let mut vector = None;
            if p < r.len() && r[p] == 1 {
                let n = u32::from_le_bytes(r[p + 1..p + 5].try_into().unwrap()) as usize;
                vector = Some((0..n).map(|i| f32::from_le_bytes(r[p + 5 + i * 4..p + 9 + i * 4].try_into().unwrap())).collect::<Vec<f32>>());
                p += 5 + n * 4;
            } else if p < r.len() {
                p += 1;
            }
            // Records from before stored text have no flag (their text, if any, is among the
            // attributes).
            let store = r.get(p) == Some(&1);
            Some((Decoded::Put(Doc { key, version, passages, attrs, vector }), store))
        }
        &DELETE => Some((Decoded::Delete(u64_at(1)), false)),
        _ => None,
    }
}

impl View {
    /// The batch this view is as of.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    /// How vectors are compared.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn analyzers(&self) -> &[Analyzer] {
        &self.analyzers
    }

    /// Reads every file of this view against the checksums written with it: each segment (its
    /// header is also checked whenever it's opened) and each file of stored text. An error names
    /// the first damaged file. It reads everything, so it takes as long as that takes; queries
    /// and writes go on meanwhile.
    pub fn verify(&self) -> std::io::Result<Verified> {
        let mut out = Verified::default();
        let mut seen = std::collections::HashSet::new();
        for s in &self.segments {
            s.segment.verify().map_err(|e| std::io::Error::new(e.kind(), format!("segment {} is damaged: {e}", s.id)))?;
            if s.segment.has_checksums() {
                out.segments += 1;
            } else {
                out.unchecked_segments += 1;
            }
            for (id, blob) in s.segment.blob_files() {
                if seen.insert(id) {
                    blob.verify().map_err(|e| std::io::Error::new(e.kind(), format!("stored text file {id} is damaged: {e}")))?;
                    out.blobs += 1;
                }
            }
        }
        Ok(out)
    }

    pub fn stats(&self) -> Stats {
        let memory: usize = self.chunks.iter().map(|c| c.index.len()).sum();
        Stats {
            segments: self.segments.len(),
            passages: self.segments.iter().map(|s| s.segment.docs as usize).sum::<usize>() + memory,
            deleted: self.deleted.iter().sum(),
            memory,
            merging: false,
            chunks: self.chunks.len(),
            seq: self.seq,
            vector_caches: self.segments.iter().filter(|s| s.segment.vector_cache_bytes() > 0).count(),
            vector_cache_bytes: self.segments.iter().map(|s| s.segment.vector_cache_bytes()).sum(),
            kept_filters: self.segments.iter().map(|s| s.segment.kept_filters().0).sum(),
            kept_filter_bytes: self.segments.iter().map(|s| s.segment.kept_filters().1).sum(),
            vector_generations: {
                let mut ids: Vec<u64> = self.segments.iter().filter_map(|s| s.segment.vector_generation().map(|g| g.id)).collect();
                ids.sort_unstable();
                ids.dedup();
                ids.len()
            },
            vector_clusters: self.segments.iter().filter_map(|s| s.segment.vector_generation()).max_by_key(|g| (g.trained_on, g.id)).map_or(0, |g| g.centroids.len() / self.dims.next_power_of_two().max(64)),
            ..Stats::default()
        }
    }

    fn in_memory(&self) -> usize {
        self.chunks.iter().map(|c| c.index.len()).sum()
    }

    /// Every live item's version, for a caller to find what changed.
    pub fn versions(&self) -> HashMap<u64, u64> {
        let mut out = HashMap::new();
        for s in &self.segments {
            for d in 0..s.segment.docs {
                if s.deletes.visible(d, self.seq) {
                    out.insert(s.segment.key(d), s.segment.version(d));
                }
            }
        }
        for c in &self.chunks {
            for (d, (&key, &version)) in c.index.keys.iter().zip(&c.index.versions).enumerate() {
                if c.deletes.visible(d as u32, self.seq) {
                    out.insert(key, version);
                }
            }
        }
        out
    }

    /// Live documents without a vector (to embed what was indexed before embeddings were on).
    pub fn without_vectors(&self) -> Vec<u64> {
        let mut with = std::collections::HashSet::new();
        for s in &self.segments {
            for d in 0..s.segment.docs {
                if s.deletes.visible(d, self.seq) && s.segment.has_vector(d) {
                    with.insert(s.segment.key(d));
                }
            }
        }
        for c in &self.chunks {
            for (d, &key) in c.index.keys.iter().enumerate() {
                if c.deletes.visible(d as u32, self.seq) && c.index.vectors.get(d).is_some_and(|v| v.is_some()) {
                    with.insert(key);
                }
            }
        }
        let mut out: Vec<u64> = self.versions().into_keys().filter(|k| !with.contains(k)).collect();
        out.sort_unstable();
        out
    }

    /// The query's terms in each of these fields (as each field analyzes them) and how informative
    /// each is there: its BM25 idf times the field's boost (0 for a term in no document). For
    /// snippets.
    pub fn term_weights(&self, text: &str, fields: &[(u8, f32)]) -> Vec<(u8, String, f32)> {
        let docs = (self.segments.iter().map(|s| s.segment.docs as u64).sum::<u64>() + self.in_memory() as u64).max(1);
        let mut out: Vec<(u8, String, f32)> = Vec::new();
        for t in tokens(text) {
            for &(f, boost) in fields {
                let Some(term) = self.analyzers[f as usize].term(&t) else { continue };
                if out.iter().any(|(of, ot, _)| *of == f && *ot == term) {
                    continue;
                }
                let w = match self.df(f, &term) {
                    0 => 0.0,
                    df => crate::search::idf(docs, df) * boost,
                };
                out.push((f, term.into_owned(), w));
            }
        }
        out
    }

    /// How many passages have `term` in `field`, across segments and memory.
    fn df(&self, field: u8, term: &str) -> u64 {
        self.segments.iter().filter_map(|s| s.segment.lookup(field, term)).map(|i| i.df as u64).sum::<u64>() + self.chunks.iter().map(|c| c.index.get(field, term).map_or(0, |p| p.docs.len() as u64)).sum::<u64>()
    }

    /// The `n` most common words starting with `prefix` (longer than it), across fields and segments.
    fn completions(&self, prefix: &str, fields: &[(u8, f32)], n: usize) -> Vec<String> {
        let mut counts: HashMap<String, u64> = HashMap::new();
        for &(f, _) in fields {
            for s in &self.segments {
                for (word, info) in s.segment.with_prefix(f, prefix) {
                    *counts.entry(word).or_default() += info.df as u64;
                }
            }
            for c in &self.chunks {
                for (word, df) in c.index.with_prefix(f, prefix) {
                    *counts.entry(word).or_default() += df as u64;
                }
            }
        }
        counts.remove(prefix);
        let mut words: Vec<(String, u64)> = counts.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        words.into_iter().take(n).map(|(w, _)| w).collect()
    }

    pub fn search(&self, r: &Request) -> Vec<Found> {
        // Collection statistics over everything (deleted passages included until merged away).
        let docs = (self.segments.iter().map(|s| s.segment.docs as u64).sum::<u64>() + self.in_memory() as u64).max(1);
        let caches: Vec<[f32; 256]> = (0..self.fields)
            .map(|f| {
                let total = self.segments.iter().map(|s| s.segment.total_tokens[f]).sum::<u64>() + self.chunks.iter().map(|c| c.index.totals[f]).sum::<u64>();
                norms::tf_cache((total as f32 / docs as f32).max(1e-3))
            })
            .collect();

        // Each word: itself and its expansions, in each field, with global weights.
        // Each word once, with how many times the query has it.
        let mut words: Vec<String> = Vec::new();
        let mut counts: Vec<f32> = Vec::new();
        for w in tokens(r.text) {
            match words.iter().position(|x| *x == w) {
                Some(i) => counts[i] += 1.0,
                None => {
                    words.push(w);
                    counts.push(1.0);
                }
            }
        }
        let mut groups: Vec<Vec<(u8, String, f32)>> = Vec::new();
        for (i, word) in words.iter().enumerate() {
            // A word the query repeats counts that many times (BM25 with query term frequency, as
            // Lucene scores a repeated clause). Long queries depend on it: on BEIR ArguAna, whose
            // queries are whole arguments, counting each word once cost 5–6 points of nDCG@10.
            let qtf = counts[i];
            let mut terms = vec![word.clone()];
            let more = if i + 1 == words.len() && r.prefix_last > 0 { r.prefix_last } else if word.chars().count() >= 4 { r.expand } else { 0 };
            if more > 0 {
                terms.extend(self.completions(word, r.fields, more));
            }
            // Each field's term for the word (its analyzer's: stemmed, or none for a stopword).
            let mut group: Vec<(u8, String, f32)> = Vec::new();
            let (mut any_term, mut stopword) = (false, false);
            for t in &terms {
                for &(f, boost) in r.fields {
                    // Text queries don't search sparse fields (their terms come weighted).
                    if self.analyzers[f as usize].sparse {
                        continue;
                    }
                    let Some(term) = self.analyzers[f as usize].term(t) else {
                        stopword = true;
                        continue;
                    };
                    any_term = true;
                    let df = self.df(f, &term);
                    if df > 0 && !group.iter().any(|(gf, gt, _)| *gf == f && *gt == term) {
                        group.push((f, term.into_owned(), weight(docs, df, boost) * qtf));
                    }
                }
            }
            // A stopword isn't part of the query where it is one; when all words must match, a
            // stopword in any field searched isn't required (whether a word is one is a matter of
            // language, not of field).
            if !any_term || (stopword && r.all) {
                continue;
            }
            if group.is_empty() {
                if r.all {
                    return Vec::new();
                }
            } else {
                groups.push(group);
            }
        }
        // Sparse terms: each its own group, weighted by the query, scored by dot product.
        let required = groups.len();
        for (f, term, w) in r.sparse {
            if *w > 0.0 && self.analyzers.get(*f as usize).is_some_and(|a| a.sparse) && self.df(*f, term) > 0 {
                groups.push(vec![(*f, term.clone(), *w / crate::analysis::SPARSE_SCALE)]);
            }
        }
        if groups.is_empty() || r.k == 0 {
            return Vec::new();
        }
        // With sparse terms too, "all words" requires the text's words, not the sparse terms (the
        // query's expansion): `all` for the searches' own intersection, else `required` leading
        // groups checked per candidate.
        let all = r.all && groups.len() == required;
        let required = if r.all { required } else { 0 };
        let linear: Vec<bool> = self.analyzers.iter().map(|a| a.sparse).collect();

        // The best k items, each at its best passage.
        let mut top = TopK::items(r.k);
        let stop = r.stop.unwrap_or(&crate::stop::Stop::NEVER);
        'segments: for (ordinal, s) in self.segments.iter().enumerate() {
            if stop.stopped() {
                break;
            }
            let mut q = Query { all, ..Default::default() };
            // The required groups present here (they come first).
            let mut text = 0;
            for (i, g) in groups.iter().enumerate() {
                let clauses: Vec<Clause> = g.iter().filter_map(|(f, t, w)| s.segment.lookup(*f, t).map(|info| Clause { info, weight: *w })).collect();
                if clauses.is_empty() {
                    if i < required && !all {
                        continue 'segments; // a required word no passage here has
                    }
                    q.impossible |= all;
                } else {
                    text += (i < required) as usize;
                    q.groups.push(clauses);
                }
            }
            let searcher = Searcher::with_caches(&s.segment, caches.clone()).linear(linear.clone());
            let has_all = (required > 0 && !all && !q.impossible).then(|| searcher.count_groups(&q.groups[..text]));
            let bound = r.filter.map(|f| f.bind(&s.segment.attr_columns()));
            // The filter's passages as bits, when a search will ask about more passages than
            // making them costs: until k pass, it asks about nearly every passage it scores.
            // From columns that is a pass over their codes (worth it from a 128th of the
            // segment in postings on); from strings as they were before version 8, a passage
            // at a time, and worth it only if a sample says a 64th pass (FiQA, hybrid with a
            // 1% filter: 0.31 → 0.26 ms).
            let postings = searcher.postings(&q);
            let allowed = bound.as_ref().filter(|_| !q.groups.is_empty()).and_then(|b| match s.segment.by_column(b) {
                true => (postings >= s.segment.docs as u64 / 128).then(|| s.segment.matching_kept(b)),
                false => s.segment.matching_at_most(b, s.segment.docs as usize / 64),
            });
            if allowed.as_ref().is_some_and(|a| a.count() == 0) {
                continue;
            }
            let check = |d: u32| {
                has_all.as_ref().is_none_or(|h| h.count(d as usize) == text as u32)
                    && match &allowed {
                        Some(a) => a.get(d),
                        None => bound.as_ref().is_none_or(|b| b.matches(&s.segment.row(d))),
                    }
            };
            let key = |d: u32| s.segment.key(d);
            top.source(s.segment.repeats_keys());
            let filtered = r.filter.is_some() || has_all.is_some();
            let mut sink = Sink { top: &mut top, deleted: Some((&s.deletes, self.seq)), filter: filtered.then_some(&check as &dyn Fn(u32) -> bool), ordinal: ordinal as u32, stop, key: Some(&key) };
            // Far fewer pass than the lists are long: from the passages that pass, not the lists.
            match allowed.as_ref().filter(|a| a.count() as u64 * 4 <= postings) {
                Some(few) => searcher.search_among(&q, few, &mut sink),
                None => searcher.search_into(&q, &mut sink),
            }
        }
        for (j, c) in self.chunks.iter().enumerate() {
            self.search_memory(c, (self.segments.len() + j) as u64, &groups, if all { groups.len() } else { required }, &caches, r.filter, &mut top);
        }

        top.into_sorted()
            .into_iter()
            .map(|hit| {
                let (key, passage) = self.locate(hit.doc);
                Found { key, passage, score: hit.score }
            })
            .collect()
    }

    /// Documents by attributes alone (no text): those passing `filter`, ordered by an integer
    /// attribute (`order_by`: name, descending) or by key, one per item (at its first passage in
    /// that order), at most `limit`, in memory that grows with the limit, not the matches.
    /// (Without `order_by`, keys are ordered as i64, as they always were: keys from 2^63 on come
    /// first.)
    ///
    /// A segment's passages aren't all looked at when the order says which can't be among the
    /// first. By key: passages are walked in the order of keys until `limit` items are kept and
    /// the next key is past them. By a column stored as codes: zones are read best first (each
    /// knows its smallest and largest code) until `limit` items are kept; then only passages
    /// whose code is at least as good as the worst kept are looked at, found by one comparison
    /// a passage that skips every zone that has none. A million documents, the 10 newest by a
    /// time that grows with the key: 129 ms when every passage was offered (each newer than
    /// all before it, so each was kept for a while), 0.1 ms.
    pub fn query(&self, filter: Option<&Filter>, order_by: Option<(&str, bool)>, limit: usize) -> Vec<Found> {
        use crate::attrs::Row;
        if limit == 0 {
            return Vec::new();
        }
        let descending = order_by.is_some_and(|(_, desc)| desc);
        // Ascending in this is the order asked for: `!v` reverses i64's order exactly. A missing
        // value sorts as the smallest integer.
        let sort_value = |v: Option<i64>, key: u64| {
            let v = match order_by {
                Some(_) => v.unwrap_or(i64::MIN),
                None => key as i64,
            };
            if descending { !v } else { v }
        };
        let mut first = FirstItems::new(limit);
        for s in &self.segments {
            let columns = s.segment.attr_columns();
            let bound = filter.map(|f| f.bind(&columns));
            let order = order_by.and_then(|(n, _)| columns.iter().position(|(c, k)| c == n && matches!(k, Kind::Int | Kind::Float)));
            // A filter's matches as a bitmap, a column at a time.
            let matching = bound.as_ref().map(|b| s.segment.matching_kept(b));
            let seg = &s.segment;
            let n = seg.docs as usize;
            let matches = matching.as_ref().map_or(n, |m| m.count());
            if matches == 0 {
                continue;
            }
            let passes = |d: u32| matching.as_ref().is_none_or(|m| m.get(d));
            // A passage that matches, if it is live.
            let offer = |first: &mut FirstItems, d: u32| {
                if s.deletes.visible(d, self.seq) {
                    let key = seg.key(d);
                    first.offer((sort_value(order.and_then(|c| seg.row(d).int(c)), key), key, seg.passage(d)));
                }
            };
            // By key, when matches are common enough that the first `limit` come soon in the
            // order of keys (about limit × passages / matches along, if they are spread evenly;
            // given up for the matches themselves if they turn out to be late).
            if order_by.is_none() && limit.saturating_mul(n) <= matches.saturating_mul(matches) {
                // Keys from 2^63 on come first.
                let (mut lo, mut hi) = (0, n);
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    if seg.key(seg.by_key(mid)) >> 63 == 0 { lo = mid + 1 } else { hi = mid }
                }
                // Eight times as far as they would come if spread evenly, then no further:
                // they are late in this order (the newest of keys that grow with time).
                let far = match &matching {
                    Some(_) => (8 * limit.saturating_mul(n) / matches).clamp(1024, matches.max(1024)),
                    None => n,
                };
                let whole = 'walk: {
                    for (walked, i) in (lo..n).chain(0..lo).enumerate() {
                        let d = seg.by_key(i);
                        if first.bar().is_some_and(|bar| seg.key(d) as i64 > bar.0) {
                            break 'walk true;
                        }
                        if walked >= far {
                            break 'walk false;
                        }
                        if passes(d) {
                            offer(&mut first, d);
                        }
                    }
                    true
                };
                if whole {
                    continue;
                }
            }
            // By a column of codes, unless the matches are so few that offering each is less
            // than finding the zones to read. (And not for the smallest first of a column some
            // documents have no value in: those come first, and have no code to compare.)
            let codes = order.and_then(|c| seg.int_codes(c)).filter(|(l, _)| matches > 4096 && l.width > 0 && (descending || l.present as usize == l.docs));
            if let Some((layout, bytes)) = codes {
                use crate::attrs::ZONE;
                // The best a zone's passages can be in the order asked for (as `sort_value`
                // has it: the smaller the better).
                let mut zones: Vec<(i64, usize)> = (0..n.div_ceil(ZONE))
                    .map(|z| {
                        let (smallest, largest) = layout.zone(bytes, z);
                        let best = match descending {
                            true if largest == 0 => sort_value(None, 0),
                            true => sort_value(Some(layout.value(largest)), 0),
                            false => sort_value(Some(layout.value(smallest.max(1))), 0),
                        };
                        (best, z)
                    })
                    .collect();
                zones.sort_unstable();
                let mut whole = 0;
                for &(_, z) in &zones {
                    if first.bar().is_some() {
                        break;
                    }
                    for d in (z * ZONE) as u32..((z + 1) * ZONE).min(n) as u32 {
                        if passes(d) {
                            offer(&mut first, d);
                        }
                    }
                    whole += 1;
                }
                let Some(bar) = first.bar().filter(|_| whole < zones.len()) else { continue };
                // What is at least as good as the worst kept. (If that one has no value, any
                // other without one may come before it: every match, below.)
                if bar.0 != sort_value(None, 0) || !descending {
                    let run = match descending {
                        true => layout.codes_between(!bar.0, i64::MAX),
                        false => layout.codes_between(i64::MIN, bar.0),
                    };
                    if let Some(run) = run {
                        let good = Bitmap::of_words(n, layout.in_runs(bytes, &[run]));
                        match &matching {
                            Some(m) => good.and(m).ones().for_each(|d| offer(&mut first, d)),
                            None => good.ones().for_each(|d| offer(&mut first, d)),
                        }
                    }
                    continue;
                }
            }
            match &matching {
                Some(m) => m.ones().for_each(|d| offer(&mut first, d)),
                None => (0..seg.docs).for_each(|d| offer(&mut first, d)),
            }
        }
        for c in &self.chunks {
            let columns = c.index.attr_columns();
            let bound = filter.map(|f| f.bind(&columns));
            let order = order_by.and_then(|(n, _)| columns.iter().position(|(c, k)| c == n && matches!(k, Kind::Int | Kind::Float)));
            for d in 0..c.index.len() as u32 {
                if !c.deletes.visible(d, self.seq) {
                    continue;
                }
                let row = c.index.row(d);
                if bound.as_ref().is_some_and(|b| !b.matches(&row)) {
                    continue;
                }
                let key = c.index.keys[d as usize];
                first.offer((sort_value(order.and_then(|c| row.int(c)), key), key, c.index.passages[d as usize]));
            }
        }
        first.into_sorted().into_iter().map(|(_, key, passage)| Found { key, passage, score: 0.0 }).collect()
    }

    /// How many items pass `filter` (every item, without one). An item counts at its first
    /// passage. A segment without deletions or items of several passages is counted from the
    /// filter's bits alone; in others, and in memory, each match is looked at.
    /// The keys of the documents that pass `filter`: found as `count` counts them, from the
    /// filter's columns.
    pub fn keys(&self, filter: &Filter) -> Vec<u64> {
        let mut keys = Vec::new();
        for s in &self.segments {
            let seg = &s.segment;
            let matching = seg.matching_kept(&filter.bind(&seg.attr_columns()));
            keys.extend(matching.ones().filter(|&d| s.deletes.visible(d, self.seq) && seg.passage(d) == 0).map(|d| seg.key(d)));
        }
        for c in &self.chunks {
            let bound = filter.bind(&c.index.attr_columns());
            keys.extend((0..c.index.len() as u32).filter(|&d| c.deletes.visible(d, self.seq) && c.index.passages[d as usize] == 0 && bound.matches(&c.index.row(d))).map(|d| c.index.keys[d as usize]));
        }
        keys
    }

    pub fn count(&self, filter: Option<&Filter>) -> u64 {
        let mut items = 0;
        for (ordinal, s) in self.segments.iter().enumerate() {
            let seg = &s.segment;
            let bound = filter.map(|f| f.bind(&seg.attr_columns()));
            let matching = bound.as_ref().map(|b| seg.matching_kept(b));
            let whole = self.deleted[ordinal] == 0 && !seg.repeats_keys();
            let item = |d: &u32| s.deletes.visible(*d, self.seq) && seg.passage(*d) == 0;
            items += match (&matching, whole) {
                (Some(m), true) => m.count(),
                (None, true) => seg.docs as usize,
                (Some(m), false) => m.ones().filter(item).count(),
                (None, false) => (0..seg.docs).filter(item).count(),
            } as u64;
        }
        for c in &self.chunks {
            let bound = filter.map(|f| f.bind(&c.index.attr_columns()));
            items += (0..c.index.len() as u32).filter(|&d| c.deletes.visible(d, self.seq) && c.index.passages[d as usize] == 0 && bound.as_ref().is_none_or(|b| b.matches(&c.index.row(d)))).count() as u64;
        }
        items
    }

    /// How many items have each value of the attribute `name`, among those that pass `filter`:
    /// the most frequent first (of equally frequent, the smaller value), at most `limit` (0:
    /// every value). Items without the attribute aren't counted.
    ///
    /// For a string column a segment counts per code and looks each string up once; without a
    /// filter, deletions or items of several passages, the counts are the ones its dictionary
    /// keeps, and no document is read. It holds a count per distinct value, so it is for
    /// attributes of few values (a kind, an owner, a status), not for ones every item has its
    /// own of.
    pub fn count_by(&self, name: &str, filter: Option<&Filter>, limit: usize) -> Vec<(Value, u64)> {
        use crate::attrs::Row;
        let mut ints: HashMap<i64, u64> = HashMap::new();
        let mut floats: HashMap<i64, u64> = HashMap::new();
        let mut strings: HashMap<String, u64> = HashMap::new();
        for (ordinal, s) in self.segments.iter().enumerate() {
            let seg = &s.segment;
            let columns = seg.attr_columns();
            let Some(c) = columns.iter().position(|(n, _)| n == name) else { continue };
            let bound = filter.map(|f| f.bind(&columns));
            let matching = bound.as_ref().map(|b| seg.matching_kept(b));
            let whole = self.deleted[ordinal] == 0 && !seg.repeats_keys();
            // Every item that passes, at its first passage.
            let each = |visit: &mut dyn FnMut(u32)| {
                let mut item = |d: u32| {
                    if whole || s.deletes.visible(d, self.seq) && seg.passage(d) == 0 {
                        visit(d)
                    }
                };
                match &matching {
                    Some(m) => m.ones().for_each(&mut item),
                    None => (0..seg.docs).for_each(&mut item),
                }
            };
            match (columns[c].1, seg.codes(c)) {
                (Kind::Str, Some((layout, bytes))) => {
                    let mut by_code = vec![0u64; layout.top as usize + 1];
                    if matching.is_none() && whole {
                        for code in 1..=layout.top {
                            by_code[code as usize] = layout.count(bytes, code);
                        }
                    } else {
                        each(&mut |d| {
                            if let Some(n) = by_code.get_mut(layout.code(bytes, d as usize) as usize) {
                                *n += 1;
                            }
                        });
                    }
                    for code in 1..=layout.top {
                        if let Some(string) = layout.string(bytes, code).filter(|_| by_code[code as usize] > 0) {
                            *strings.entry(string.to_string()).or_default() += by_code[code as usize];
                        }
                    }
                }
                (Kind::Str, None) => each(&mut |d| {
                    if let Some(string) = seg.row(d).str(c) {
                        *strings.entry(string.to_string()).or_default() += 1;
                    }
                }),
                // A document counts once for each of its tags.
                (Kind::Tags, Some((layout, bytes))) => {
                    let mut by_code = vec![0u64; layout.top as usize + 1];
                    if matching.is_none() && whole {
                        for code in 1..=layout.top {
                            by_code[code as usize] = layout.count(bytes, code);
                        }
                    } else {
                        each(&mut |d| {
                            for code in layout.tags(bytes, d as usize) {
                                if let Some(n) = by_code.get_mut(code as usize) {
                                    *n += 1;
                                }
                            }
                        });
                    }
                    for code in 1..=layout.top {
                        if let Some(string) = layout.string(bytes, code).filter(|_| by_code[code as usize] > 0) {
                            *strings.entry(string.to_string()).or_default() += by_code[code as usize];
                        }
                    }
                }
                (Kind::Tags, None) => {}
                (Kind::Int, _) => each(&mut |d| {
                    if let Some(x) = seg.row(d).int(c) {
                        *ints.entry(x).or_default() += 1;
                    }
                }),
                (Kind::Float, _) => each(&mut |d| {
                    if let Some(x) = seg.row(d).int(c) {
                        *floats.entry(x).or_default() += 1;
                    }
                }),
            }
        }
        for chunk in &self.chunks {
            let columns = chunk.index.attr_columns();
            let Some(c) = columns.iter().position(|(n, _)| n == name) else { continue };
            let bound = filter.map(|f| f.bind(&columns));
            for d in 0..chunk.index.len() as u32 {
                let row = chunk.index.row(d);
                if !chunk.deletes.visible(d, self.seq) || chunk.index.passages[d as usize] != 0 || bound.as_ref().is_some_and(|b| !b.matches(&row)) {
                    continue;
                }
                if let Some(x) = row.int(c) {
                    *if columns[c].1 == Kind::Float { &mut floats } else { &mut ints }.entry(x).or_default() += 1;
                } else if let Some(string) = row.str(c) {
                    *strings.entry(string.to_string()).or_default() += 1;
                } else {
                    row.some_tag(c, &mut |tag| {
                        *strings.entry(tag.to_string()).or_default() += 1;
                        false
                    });
                }
            }
        }
        // Among the equally frequent, the smaller value (a column is of one kind; floats are
        // in order as the integers they are kept as).
        let mut ints: Vec<(i64, u64)> = ints.into_iter().collect();
        ints.sort_unstable();
        let mut floats: Vec<(i64, u64)> = floats.into_iter().collect();
        floats.sort_unstable();
        let mut strings: Vec<(String, u64)> = strings.into_iter().collect();
        strings.sort_unstable();
        let mut counts: Vec<(Value, u64)> = ints
            .into_iter()
            .map(|(x, n)| (Value::Int(x), n))
            .chain(floats.into_iter().map(|(x, n)| (Value::Float(crate::attrs::code_float(x)), n)))
            .chain(strings.into_iter().map(|(s, n)| (Value::Str(s), n)))
            .collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1));
        if limit > 0 {
            counts.truncate(limit);
        }
        counts
    }

    /// Where `key`'s first live passage is: a chunk's (Ok) or a segment's (Err), newest first.
    fn find(&self, key: u64) -> Option<Result<(&Chunk, u32), (&SegEntry, u32)>> {
        for c in self.chunks.iter().rev() {
            if let Some(&d) = c.keys.get(&key).and_then(|docs| docs.iter().find(|&&d| c.deletes.visible(d, self.seq))) {
                return Some(Ok((c, d)));
            }
        }
        for s in self.segments.iter().rev() {
            if let Some(d) = s.segment.docs_with_key(key).into_iter().find(|&d| s.deletes.visible(d, self.seq)) {
                return Some(Err((s, d)));
            }
        }
        None
    }

    /// Whether the item with `key` passes `filter` (None: there is no such item).
    pub fn passes(&self, key: u64, filter: &Filter) -> Option<bool> {
        Some(match self.find(key)? {
            Ok((c, d)) => filter.bind(&c.index.attr_columns()).matches(&c.index.row(d)),
            Err((s, d)) => filter.bind(&s.segment.attr_columns()).matches(&s.segment.row(d)),
        })
    }

    /// An item as it could be written again, for changing part of it: its version, its
    /// attributes, its stored text (its first passage's, if the index kept it), its vector, and
    /// how many passages it has.
    pub fn document(&self, key: u64) -> Option<Whole> {
        let at = self.find(key)?;
        let (version, attrs, vector, passages) = match at {
            Ok((c, d)) => {
                let vector = c.index.vectors.get(d as usize).cloned().flatten().map(|v| v.iter().map(|&h| crate::vector::f16_to_f32(h)).collect());
                (c.index.versions[d as usize], c.index.attributes(d), vector, c.keys.get(&key).map_or(1, |docs| docs.iter().filter(|&&d| c.deletes.visible(d, self.seq)).count()))
            }
            Err((s, d)) => (s.segment.version(d), s.segment.attributes(d), s.segment.vector(d), s.segment.docs_with_key(key).into_iter().filter(|&d| s.deletes.visible(d, self.seq)).count()),
        };
        Some(Whole { version, attrs, text: Self::text_at(at), vector, passages })
    }

    /// An item's attributes (from its first live passage), or None if it isn't in the index.
    pub fn get(&self, key: u64) -> Option<Vec<(String, Value)>> {
        Some(match self.find(key)? {
            Ok((c, d)) => c.index.attributes(d),
            Err((s, d)) => s.segment.attributes(d),
        })
    }

    /// An item's attributes and, if `with_text`, its stored text (both from its first live
    /// passage), found once: what `get` and `text` give, for a result.
    pub fn hydrate(&self, key: u64, with_text: bool) -> Option<(Vec<(String, Value)>, Option<Vec<String>>)> {
        let at = self.find(key)?;
        let attrs = match at {
            Ok((c, d)) => c.index.attributes(d),
            Err((s, d)) => s.segment.attributes(d),
        };
        Some((attrs, if with_text { Self::text_at(at) } else { None }))
    }

    /// An item's stored text, one per field (its first passage's), if the index stored it.
    pub fn text(&self, key: u64) -> Option<Vec<String>> {
        Self::text_at(self.find(key)?)
    }

    fn text_at(at: Result<(&Chunk, u32), (&SegEntry, u32)>) -> Option<Vec<String>> {
        match at {
            Ok((c, d)) => {
                if let Some(t) = c.index.texts.get(d as usize).cloned().flatten() {
                    return Some(t.to_vec());
                }
                // A log from before blob files: text among the attributes.
                let mut out: Vec<String> = Vec::new();
                for (n, v) in c.index.attributes(d) {
                    if let (Some(i), Value::Str(t)) = (n.strip_prefix(segment::LEGACY_TEXT).and_then(|i| i.parse::<usize>().ok()), v) {
                        if out.len() <= i {
                            out.resize(i + 1, String::new());
                        }
                        out[i] = t;
                    }
                }
                (!out.is_empty()).then_some(out)
            }
            Err((s, d)) => s.segment.stored(d),
        }
    }

    /// The k documents whose vectors are nearest to `q` by the index's metric, passing `filter`.
    /// A score is higher for a nearer one: the cosine, the squared Euclidean distance negated,
    /// or the dot product (`vector::Metric`). Each segment probes its clusters nearest first
    /// (by dot product: those that promise the greatest product first, `vector::search_lists`);
    /// `nprobe` is how many it probes
    /// before stopping once there are k results (more: slower and closer to exact; 0: the budget
    /// each segment calibrated for recall@10 of 0.997 when it was built). Filtered-out
    /// and deleted documents never enter the results, and probing goes on until k pass.
    pub fn nearest(&self, q: &[f32], k: usize, filter: Option<&Filter>, nprobe: usize) -> Vec<Found> {
        self.nearest_with_stats(q, k, filter, nprobe).0
    }

    /// `nearest`, with how many codes were estimated and how many vectors read at full precision.
    pub fn nearest_with_stats(&self, q: &[f32], k: usize, filter: Option<&Filter>, nprobe: usize) -> (Vec<Found>, usize, usize) {
        self.nearest_stopping(q, k, filter, nprobe, &crate::stop::Stop::NEVER)
    }

    /// `nearest`, giving up between segments and clusters when `stop` says so.
    pub fn nearest_until(&self, q: &[f32], k: usize, filter: Option<&Filter>, nprobe: usize, stop: &crate::stop::Stop) -> Vec<Found> {
        self.nearest_stopping(q, k, filter, nprobe, stop).0
    }

    fn nearest_stopping(&self, q: &[f32], k: usize, filter: Option<&Filter>, nprobe: usize, stop: &crate::stop::Stop) -> (Vec<Found>, usize, usize) {
        let (found, estimated, reranked, _) = self.nearest_traced(q, k, filter, nprobe, stop, false);
        (found, estimated, reranked)
    }

    /// Why `nearest` missed what it missed: of `truth` (a reference ranking's top k keys), those
    /// it didn't return, by cause (see `Misses`). For calibrating and checking the search, not
    /// for queries: it traces which clusters were scanned and looks each miss up.
    pub fn nearest_misses(&self, q: &[f32], k: usize, filter: Option<&Filter>, nprobe: usize, truth: &[u64]) -> (Vec<Found>, Misses) {
        let (found, estimated, reranked, probed) = self.nearest_traced(q, k, filter, nprobe, &crate::stop::Stop::NEVER, true);
        let probed: std::collections::HashSet<(u32, u32)> = probed.unwrap_or_default().into_iter().collect();
        let mut m = Misses { estimated, reranked, ..Misses::default() };
        let mut qn = q.to_vec();
        if found.is_empty() || !self.metric.prepare(&mut qn) {
            return (found, m);
        }
        let got: std::collections::HashSet<u64> = found.iter().map(|f| f.key).collect();
        // What the k-th result was ordered by (for cosine similarity s, the squared distance 2 − 2s).
        let kth = if found.len() < k { f32::INFINITY } else { self.metric.key(found[found.len() - 1].score) };
        for &key in truth.iter().filter(|key| !got.contains(key)) {
            match self.find(key) {
                // Unflushed vectors are compared exactly.
                None | Some(Ok(_)) => m.precision += 1,
                Some(Err((s, d))) => {
                    let ordinal = self.segments.iter().position(|e| std::ptr::eq(&**e, s)).unwrap() as u32;
                    let far = s.segment.vector_key(d, &qn).is_none_or(|key| key > kth);
                    if far || probed.contains(&(ordinal, EXACT)) {
                        m.precision += 1;
                    } else if s.segment.vector_cluster(d).is_some_and(|c| probed.contains(&(ordinal, c))) {
                        m.prune += 1;
                    } else {
                        m.probe += 1;
                    }
                }
            }
        }
        (found, m)
    }

    /// `nearest_stopping`, and if `trace`, the (segment, cluster) pairs it scanned (`EXACT` for a
    /// segment compared exactly).
    #[allow(clippy::type_complexity)]
    fn nearest_traced(&self, q: &[f32], k: usize, filter: Option<&Filter>, nprobe: usize, stop: &crate::stop::Stop, trace: bool) -> (Vec<Found>, usize, usize, Option<Vec<(u32, u32)>>) {
        let mut q = q.to_vec();
        if self.dims == 0 || q.len() != self.dims || k == 0 || !self.metric.prepare(&mut q) {
            return (Vec::new(), 0, 0, None);
        }
        let fresh = || if trace { crate::vector::Nearest::traced(k) } else { crate::vector::Nearest::new(k) };
        let mut top = fresh();
        let with_vectors: Vec<(usize, &SegEntry)> = self.segments.iter().enumerate().filter(|(_, s)| s.segment.vector_count() > 0).map(|(o, s)| (o, &**s)).collect();
        // The query rotated once per rotation (segments of a namespace share one).
        let mut rotated: Vec<(u64, Vec<f32>)> = Vec::new();
        for (_, s) in &with_vectors {
            if let Some(rotation) = s.segment.rotation() {
                if !rotated.iter().any(|(r, _)| *r == rotation.seed) {
                    rotated.push((rotation.seed, rotation.apply(&q)));
                }
            }
        }
        // Segments of one generation are searched as one (`vector::search_lists`); a segment with
        // clusters of its own (from before generations), or with none, on its own.
        let mut groups: Vec<(u64, Vec<(usize, &SegEntry)>)> = Vec::new();
        for &(o, s) in &with_vectors {
            let id = s.segment.vector_generation().map_or(0, |g| g.id);
            match groups.iter_mut().find(|(g, _)| id != 0 && *g == id) {
                Some((_, members)) => members.push((o, s)),
                None => groups.push((id, vec![(o, s)])),
            }
        }
        let search = |members: &[(usize, &SegEntry)], top: &mut crate::vector::Nearest| {
            if stop.stopped() {
                return;
            }
            // Per segment: its filter, and whether its matches are few enough to compare exactly.
            struct Part<'a> {
                ordinal: usize,
                s: &'a SegEntry,
                bound: Option<crate::attrs::Bound>,
                /// The filter's documents as bits, once they are made: before probing if few
                /// pass, else by the scan when it wants them (`lazy`: they can be made from
                /// columns).
                allowed: std::cell::OnceCell<Arc<Bitmap>>,
                lazy: bool,
                passing: f64,
                /// The filter's share of the segment's documents.
                share: f64,
                /// Few enough pass that the scan looks the bits up a block of codes at a time.
                masked: bool,
            }
            let mut parts: Vec<Part> = Vec::with_capacity(members.len());
            for &(ordinal, s) in members {
                let bound = filter.map(|f| f.bind(&s.segment.attr_columns()));
                let docs = s.segment.docs.max(1) as f64;
                // The filter's documents as bits, now, if a sample says few pass: they may be
                // compared exactly, and a scan would look them up from its first list on. From
                // columns that is a pass over their codes (0.05 ms a million documents); from
                // strings as they were before version 8, a document at a time, so only for what
                // may be compared exactly. When more pass, a scan asks for them if it finds it
                // needs them (`vector::Among`), and most don't.
                let by_column = bound.as_ref().is_some_and(|b| s.segment.by_column(b));
                let share = bound.as_ref().map(|b| s.segment.sample_share(b));
                let few = EXACT_FILTERED.max(s.segment.vector_count() / 32);
                let allowed = bound.as_ref().filter(|_| share.is_some_and(|f| if by_column { f <= MASKED_SHARE } else { f * docs <= 2.0 * few as f64 })).map(|b| s.segment.matching_kept(b));
                // Compared exactly, one by one, if that is cheaper than probing clusters for
                // them (FiQA, a filter of 1%: 0.32 → 0.03 ms): the codes of as many clusters as
                // the depth of k among the matches takes (`vector::depth_for`).
                let exact = |matches: usize| match s.segment.vector_budgets().and_then(|(budgets, base, clusters, vectors)| Some((budgets.probes(clusters, vectors, crate::vector::depth_for(k, matches as f64 / docs))?.max(base), clusters, vectors))) {
                    Some((probes, clusters, vectors)) => (matches * EXACT_CODES) as f64 <= probes as f64 * vectors as f64 / clusters.max(1) as f64,
                    None => matches <= few,
                };
                if let Some(a) = allowed.as_ref().filter(|a| exact(a.count())) {
                    if let Some(p) = &mut top.probed {
                        p.push((ordinal as u32, EXACT));
                    }
                    for d in a.ones() {
                        if s.deletes.visible(d, self.seq) {
                            if let Some(key) = s.segment.vector_key(d, &q) {
                                top.reranked += 1;
                                top.offer(key, (ordinal as u64) << 32 | d as u64);
                            }
                        }
                    }
                    continue;
                }
                // The share of vectors a search can return here, which deepens its budget: the
                // filter's (counted, or from a sample), less what's deleted.
                let filtered = match (&allowed, share) {
                    (Some(a), _) => a.count() as f64 / docs,
                    (None, Some(f)) => f,
                    (None, None) => 1.0,
                };
                let masked = allowed.is_some() && filtered <= MASKED_SHARE;
                let allowed = allowed.map_or_else(std::cell::OnceCell::new, std::cell::OnceCell::from);
                parts.push(Part { ordinal, s, bound, allowed, lazy: by_column, passing: filtered * (1.0 - self.deleted[ordinal] as f64 / docs), share: filtered, masked });
            }
            // Probing clusters (for a segment from before version 8, only now its scan caches).
            let keeps: Vec<Box<dyn Fn(u32) -> bool + '_>> = parts
                .iter()
                .map(|part| {
                    Box::new(move |d: u32| {
                        part.s.deletes.visible(d, self.seq)
                            && match part.allowed.get() {
                                Some(a) => a.get(d),
                                None => part.bound.as_ref().is_none_or(|b| b.matches(&part.s.segment.row(d))),
                            }
                    }) as Box<dyn Fn(u32) -> bool + '_>
                })
                .collect();
            let makes: Vec<Box<dyn Fn() -> Arc<Bitmap> + '_>> = parts.iter().map(|part| Box::new(move || part.bound.as_ref().map_or_else(|| Arc::new(Bitmap::new(0)), |b| part.s.segment.matching_kept(b))) as Box<dyn Fn() -> Arc<Bitmap> + '_>).collect();
            let views: Vec<crate::vector::Member> = parts
                .iter()
                .zip(keeps.iter().zip(&makes))
                .filter_map(|(part, (keep, make))| {
                    let among = part.bound.as_ref().map(|_| crate::vector::Among { bits: &part.allowed, make: part.lazy.then_some(&**make), share: part.share, kept: part.passing, mask: part.masked });
                    Some(crate::vector::Member { view: part.s.segment.vector_view()?, ordinal: part.ordinal as u32, keep: &**keep, among })
                })
                .collect();
            let Some(first) = views.first() else { return };
            let Some((_, qr)) = rotated.iter().find(|(r, _)| *r == first.view.rotation.seed) else { return };
            if let [only] = &views[..] {
                // (One segment: with one cluster it's compared exactly.)
                let passing = parts.iter().find(|p| p.ordinal as u32 == only.ordinal).map_or(1.0, |p| p.passing);
                only.view.search_rotated(&q, qr, only.ordinal, only.keep, only.among, top, nprobe, passing, stop);
                return;
            }
            // What the members' filters and deletions keep, of all their vectors.
            let vectors: f64 = parts.iter().map(|p| p.s.segment.vector_count() as f64).sum();
            let passing = parts.iter().map(|p| p.passing * p.s.segment.vector_count() as f64).sum::<f64>() / vectors.max(1.0);
            crate::vector::search_lists(&views, &q, qr, top, nprobe, passing, stop);
        };
        let vectors: usize = with_vectors.iter().map(|(_, s)| s.segment.vector_count()).sum();
        if groups.len() > 1 && vectors >= PARALLEL_VECTORS {
            // Groups at once, each with its own top k, merged: without a shared threshold each
            // reads a few more vectors at full precision, but the query takes as long as its
            // largest group instead of all of them (FiQA in 5 segments: 0.31 → 0.18 ms).
            let tops: Vec<crate::vector::Nearest> = query_pool().install(|| {
                groups
                    .par_iter()
                    .map(|(_, members)| {
                        let mut t = fresh();
                        search(members, &mut t);
                        t
                    })
                    .collect()
            });
            for mut t in tops {
                top.estimated += t.estimated;
                top.reranked += t.reranked;
                if let (Some(all), Some(p)) = (&mut top.probed, t.probed.take()) {
                    all.extend(p);
                }
                for (d2, id) in t.into_sorted() {
                    top.offer(d2, id);
                }
            }
        } else {
            for (_, members) in &groups {
                search(members, &mut top);
            }
        }
        self.nearest_in_memory(&q, filter, &mut top);
        let (estimated, reranked, probed) = (top.estimated, top.reranked, top.probed.take());
        let found = top.into_sorted().into_iter().map(|(d2, id)| {
            let (key, passage) = self.locate(id);
            Found { key, passage, score: self.metric.score(d2) }
        }).collect();
        (found, estimated, reranked, probed)
    }

    /// The same as `nearest`, scoring every vector exactly (for checking recall).
    pub fn nearest_exact(&self, q: &[f32], k: usize, filter: Option<&Filter>) -> Vec<Found> {
        let mut q = q.to_vec();
        if k == 0 || self.dims == 0 || q.len() != self.dims || !self.metric.prepare(&mut q) {
            return Vec::new();
        }
        let mut top = crate::vector::Nearest::new(k);
        for (ordinal, s) in self.segments.iter().enumerate() {
            let bound = filter.map(|f| f.bind(&s.segment.attr_columns()));
            let keep = |d: u32| s.deletes.visible(d, self.seq) && bound.as_ref().is_none_or(|b| b.matches(&s.segment.row(d)));
            s.segment.vector_keys(&q, keep, |d, d2| top.offer(d2, (ordinal as u64) << 32 | d as u64));
        }
        self.nearest_in_memory(&q, filter, &mut top);
        top.into_sorted().into_iter().map(|(d2, id)| {
            let (key, passage) = self.locate(id);
            Found { key, passage, score: self.metric.score(d2) }
        }).collect()
    }

    /// The chunks' vectors, scored exactly (by the metric's key: `vector::Metric`).
    fn nearest_in_memory(&self, q: &[f32], filter: Option<&Filter>, top: &mut crate::vector::Nearest) {
        let dot = self.metric == Metric::Dot;
        let scan = |j: usize, c: &Chunk, top: &mut crate::vector::Nearest| {
            let ordinal = ((self.segments.len() + j) as u64) << 32;
            let bound = filter.map(|f| f.bind(&c.index.attr_columns()));
            for (d, v) in c.index.vectors.iter().enumerate() {
                let Some(v) = v else { continue };
                if !c.deletes.visible(d as u32, self.seq) || bound.as_ref().is_some_and(|b| !b.matches(&c.index.row(d as u32))) {
                    continue;
                }
                top.reranked += 1;
                top.offer(if dot { -crate::vector::dot_f16(q, v) } else { crate::vector::dist2_f16(q, v) }, ordinal | d as u64);
            }
        };
        let vectors: usize = self.chunks.iter().map(|c| c.index.len()).sum();
        if self.chunks.len() > 1 && vectors >= PARALLEL_VECTORS {
            // Unflushed vectors are scanned exactly, up to two memtables' worth while one flushes:
            // chunk by chunk at once, since a query waits on the scan (60,000 at 384 dimensions
            // had been most of the slowest queries while writing).
            let tops: Vec<crate::vector::Nearest> = query_pool().install(|| {
                self.chunks
                    .par_iter()
                    .enumerate()
                    .map(|(j, c)| {
                        let mut t = crate::vector::Nearest::new(top.k);
                        scan(j, c, &mut t);
                        t
                    })
                    .collect()
            });
            for t in tops {
                top.estimated += t.estimated;
                top.reranked += t.reranked;
                for (d2, id) in t.into_sorted() {
                    top.offer(d2, id);
                }
            }
        } else {
            for (j, c) in self.chunks.iter().enumerate() {
                scan(j, c, top);
            }
        }
    }

    /// (key, passage) for a result's (ordinal << 32 | document): segments, then chunks.
    fn locate(&self, id: u64) -> (u64, u32) {
        let (ordinal, doc) = ((id >> 32) as usize, id as u32);
        if ordinal >= self.segments.len() {
            let m = &self.chunks[ordinal - self.segments.len()].index;
            (m.keys[doc as usize], m.passages[doc as usize])
        } else {
            let s = &self.segments[ordinal].segment;
            (s.key(doc), s.passage(doc))
        }
    }

    /// A chunk, scored exhaustively.
    #[allow(clippy::too_many_arguments)]
    /// Scores a chunk's passages for `groups`, the first `required` of which each must have a term.
    fn search_memory(&self, c: &Chunk, ordinal: u64, groups: &[Vec<(u8, String, f32)>], required: usize, caches: &[[f32; 256]], filter: Option<&Filter>, top: &mut TopK) {
        // Term at a time into an array (a chunk is small, and every posting of it is scored).
        let n = c.index.len();
        let mut scores = vec![0f32; n];
        let mut matched = crate::search::Groups::new(if required > 0 { n } else { 0 });
        for (g, group) in groups.iter().enumerate() {
            for (f, t, w) in group {
                let Some(p) = c.index.get(*f, t) else { continue };
                let sparse = self.analyzers[*f as usize].sparse;
                let cache = &caches[*f as usize];
                for (&doc, &tf) in p.docs.iter().zip(&p.tfs) {
                    let tf = tf as f32;
                    scores[doc as usize] += if sparse { w * tf } else { w * (tf / (tf + cache[c.index.norm_id(*f as usize, doc) as usize])) };
                    if g < required {
                        matched.mark(doc as usize, g);
                    }
                }
            }
        }
        let need = required as u32;
        top.source(c.repeats);
        let bound = filter.map(|f| f.bind(&c.index.attr_columns()));
        let ordinal = ordinal << 32;
        for (d, &s) in scores.iter().enumerate() {
            if s > 0.0 && top.threshold().is_none_or(|t| s >= t) && (required == 0 || matched.count(d) == need) && c.deletes.visible(d as u32, self.seq) && bound.as_ref().is_none_or(|b| b.matches(&c.index.row(d as u32))) {
                top.offer_item(s, ordinal | d as u64, c.index.keys[d]);
            }
        }
    }
}

/// The first `k` items in the order of (value, key, passage), each at its first passage: a max-heap
/// of k entries, and where each item is in it. Whatever matches, scratch stays k entries.
struct FirstItems {
    k: usize,
    heap: Vec<(i64, u64, u32)>,
    at: HashMap<u64, usize>,
}

impl FirstItems {
    fn new(k: usize) -> FirstItems {
        let room = k.min(1024);
        FirstItems { k, heap: Vec::with_capacity(room), at: HashMap::with_capacity(room) }
    }

    /// The last of the kept, once `k` are: what a passage has to come before to be kept.
    fn bar(&self) -> Option<(i64, u64, u32)> {
        (self.heap.len() == self.k).then(|| self.heap[0])
    }

    fn offer(&mut self, e: (i64, u64, u32)) {
        // Full, and not before the last kept: neither a new item nor a better passage of a kept
        // one (which is no later than the last).
        if self.heap.len() == self.k && e >= self.heap[0] {
            return;
        }
        if let Some(&i) = self.at.get(&e.1) {
            if e < self.heap[i] {
                self.heap[i] = e;
                self.down(i);
            }
        } else if self.heap.len() < self.k {
            self.at.insert(e.1, self.heap.len());
            self.heap.push(e);
            self.up(self.heap.len() - 1);
        } else {
            self.at.remove(&self.heap[0].1);
            self.at.insert(e.1, 0);
            self.heap[0] = e;
            self.down(0);
        }
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.heap.swap(i, j);
        self.at.insert(self.heap[i].1, i);
        self.at.insert(self.heap[j].1, j);
    }

    fn up(&mut self, mut i: usize) {
        while i > 0 && self.heap[i] > self.heap[(i - 1) / 2] {
            self.swap(i, (i - 1) / 2);
            i = (i - 1) / 2;
        }
    }

    fn down(&mut self, mut i: usize) {
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut m = i;
            if l < self.heap.len() && self.heap[l] > self.heap[m] {
                m = l;
            }
            if r < self.heap.len() && self.heap[r] > self.heap[m] {
                m = r;
            }
            if m == i {
                return;
            }
            self.swap(i, m);
            i = m;
        }
    }

    fn into_sorted(mut self) -> Vec<(i64, u64, u32)> {
        self.heap.sort_unstable();
        self.heap
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        if self.read_only() {
            return;
        }
        if self.wal_failed {
            // Let builders finish before releasing the writer lock, but never publish beyond
            // a potentially torn WAL tail. Recovery will discard unreferenced builder output.
            if let Some(f) = self.flush.take() {
                let _ = f.handle.join();
            }
            if let Some(m) = self.merge.take() {
                let _ = m.handle.join();
            }
            return;
        }
        let _ = self.wait_merge();
        let _ = self.checkpoint();
        let _ = self.wait_merge();
    }
}

/// The log files in `dir`, oldest first: "wal" from before format 2 (as 0), then wal-1, wal-2, …
fn log_files(dir: &Path) -> std::io::Result<Vec<(u64, PathBuf)>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let e = entry?;
        let name = e.file_name().to_string_lossy().into_owned();
        let n = if name == WAL { Some(0) } else { name.strip_prefix("wal-").and_then(|n| n.parse::<u64>().ok()) };
        if let Some(n) = n {
            out.push((n, e.path()));
        }
    }
    out.sort();
    Ok(out)
}

/// The records of a log and its format: after its header, or all of it for a log from before
/// headers (format 1).
fn log_records(log: &[u8]) -> std::io::Result<(&[u8], u32)> {
    if log.len() >= 12 && &log[..8] == WAL_MAGIC {
        let v = u32::from_le_bytes(log[8..12].try_into().unwrap());
        if v > FORMAT {
            return Err(too_new("the write-ahead log", v, FORMAT));
        }
        return Ok((&log[12..], v));
    }
    Ok((log, 1))
}

fn write_log_header(wal: &mut BufWriter<File>) -> std::io::Result<()> {
    wal.write_all(WAL_MAGIC)?;
    wal.write_all(&COSINE_FORMAT.to_le_bytes())?;
    wal.flush()
}

fn read_only() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::PermissionDenied, "this index is open read-only")
}

/// The segments a manifest names, opened, with their deletions; the next segment id; the
/// fields' analyzers (plain for a manifest from before analyzers); the index's identity; and
/// how its vectors are compared (cosine for a manifest that doesn't say: every one before
/// format 8).
#[allow(clippy::type_complexity)]
fn read_manifest(dir: &Path) -> std::io::Result<(Vec<(u64, Segment, Bitmap, u64)>, u64, Vec<Analyzer>, u128, Metric)> {
    let mut segments = Vec::new();
    let mut analyzers = Vec::new();
    let mut metric: Option<Metric> = None;
    let text = match std::fs::read_to_string(dir.join(MANIFEST)) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // A missing manifest is only initialization if there is no finalized index data.
            // Guessing which files are orphans here would risk deleting committed documents.
            if dir.try_exists()? {
                for entry in std::fs::read_dir(dir)? {
                    let name = entry?.file_name();
                    let name = name.to_string_lossy();
                    if name.starts_with("seg-") || crate::blob::id_of(&name).is_some() {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "missing manifest with existing index data; recovery required"));
                    }
                }
            }
            return Ok((segments, 1, analyzers, 0, Metric::Cosine));
        }
        Err(e) => return Err(e),
    };
    let invalid = || std::io::Error::new(std::io::ErrorKind::InvalidData, format!("invalid manifest in {}", dir.display()));
    // Format 4 on: the last line is a CRC32 of everything before it, so a manifest damaged in a
    // way that still parses is refused rather than believed.
    let version = match text.lines().next().and_then(|l| l.split_once(' ')) {
        Some(("fugudb-index", v)) => v.parse::<u32>().unwrap_or(0),
        _ => 1,
    };
    if version > FORMAT {
        return Err(too_new(&format!("the index in {}", dir.display()), version, FORMAT));
    }
    let text = if version >= 4 {
        let body = text.strip_suffix('\n').and_then(|t| t.rsplit_once('\n')).ok_or_else(invalid)?;
        let crc = body.1.strip_prefix("crc32 ").and_then(|c| u32::from_str_radix(c, 16).ok()).ok_or_else(invalid)?;
        let covered = &text[..body.0.len() + 1];
        if crc32fast::hash(covered.as_bytes()) != crc {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("the manifest in {} fails its checksum", dir.display())));
        }
        covered.to_string()
    } else {
        text
    };
    let mut next_id = None;
    let mut index_id = 0u128;
    let mut analysis_seen = false;
    let mut ids = std::collections::HashSet::new();
    if !text.ends_with('\n') {
        return Err(invalid());
    }
    {
        match text.lines().next().and_then(|l| l.split_once(' ')) {
            Some(("fugu-text", "1")) => {}
            Some(("fugudb-index", v)) => {
                let v: u32 = v.parse().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "manifest"))?;
                if v > FORMAT {
                    return Err(too_new(&format!("the index in {}", dir.display()), v, FORMAT));
                }
                if v == 0 {
                    return Err(invalid());
                }
            }
            _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{} has no hinny manifest", dir.display()))),
        }
        for line in text.lines().skip(1) {
            match line.split_once(' ') {
                Some(("next", n)) if next_id.is_none() => next_id = Some(n.parse::<u64>().map_err(|_| invalid())?),
                Some(("id", hex)) if index_id == 0 => index_id = u128::from_str_radix(hex, 16).ok().filter(|id| *id != 0).ok_or_else(invalid)?,
                Some(("analysis", codes)) if !analysis_seen => {
                    analysis_seen = true;
                    analyzers = codes.split_whitespace().map(|c| Analyzer::from_code(c).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("unknown analyzer {c:?} in {}", dir.display())))).collect::<std::io::Result<_>>()?;
                }
                // Format 8: how vectors are compared, when it isn't by cosine.
                Some(("metric", name)) if metric.is_none() && version >= 8 => {
                    metric = Some(Metric::from_name(name).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("unknown metric {name:?} in {}", dir.display())))?);
                }
                // "segment ID", or from format 7 "segment ID N": its deletions are in the file
                // numbered N (`del_path`).
                Some(("segment", rest)) => {
                    let (id, del) = rest.split_once(' ').unwrap_or((rest, "0"));
                    let id: u64 = id.parse().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "manifest"))?;
                    let del: u64 = del.parse().map_err(|_| invalid())?;
                    if id == 0 || !ids.insert(id) {
                        return Err(invalid());
                    }
                    let segment = Segment::open(&seg_path(dir, id))?;
                    let deleted = match std::fs::read(del_path(dir, id, del)) {
                        Ok(b) => read_deletions(&b, segment.docs as usize).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("invalid deletion bitmap for segment {id}")))?,
                        // Before format 7 a segment without deletions had no file, and none was
                        // named: a missing one is no deletions. A file the manifest names must be
                        // there (or this manifest has just been replaced: a reader reads again).
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound && del == 0 => Bitmap::new(segment.docs as usize),
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                            return Err(std::io::Error::new(std::io::ErrorKind::NotFound, format!("the manifest in {} names {}, which is missing", dir.display(), del_path(dir, id, del).display())));
                        }
                        Err(e) => return Err(e),
                    };
                    segments.push((id, segment, deleted, del));
                }
                _ => return Err(invalid()),
            }
        }
    }
    let next_id = next_id.filter(|n| *n > ids.iter().copied().max().unwrap_or(0)).ok_or_else(invalid)?;
    // Vectors are stored for one metric and mean something else under another: a segment of
    // another than the index's (a file from elsewhere) is refused, not searched.
    let metric = metric.unwrap_or_default();
    if let Some((id, other)) = segments.iter().find_map(|(id, s, _, _)| s.vector_metric().filter(|m| *m != metric).map(|m| (*id, m))) {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("segment {id} in {} holds vectors compared by {}; the index's are compared by {}", dir.display(), other.name(), metric.name())));
    }
    Ok((segments, next_id, analyzers, index_id, metric))
}

/// An identity for a new index: 128 bits no other index will have. From the hasher's keys,
/// which the standard library takes from the operating system's randomness for each process,
/// stirred with the time and a count, so two made in one process differ too.
fn new_id() -> u128 {
    use std::hash::{BuildHasher, Hasher};
    static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let half = |salt: u64| {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(salt);
        h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
        h.write_u64(COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        h.write_u32(std::process::id());
        h.finish()
    };
    (((half(1) as u128) << 64) | half(2) as u128).max(1)
}

fn seg_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("seg-{id:08}.swt"))
}

/// A manifest with its checksum line.
fn sealed(mut manifest: String) -> String {
    let crc = crc32fast::hash(manifest.as_bytes());
    manifest.push_str(&format!("crc32 {crc:08x}\n"));
    manifest
}

/// A deletion file: header, bitmap, and a CRC32 of both.
fn deletion_file(bitmap: &Bitmap) -> Vec<u8> {
    let mut out = DEL_MAGIC.to_vec();
    out.extend_from_slice(&DEL_VERSION.to_le_bytes());
    out.extend_from_slice(&bitmap.to_bytes());
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

/// A segment's deletions from its file: the checksummed form, or the raw bitmap of an older
/// version. None for anything else: a wrong length, header or checksum.
fn read_deletions(bytes: &[u8], docs: usize) -> Option<Bitmap> {
    let raw = docs.div_ceil(64) * 8;
    if bytes.len() == raw {
        return Some(Bitmap::from_bytes(bytes, docs));
    }
    if bytes.len() != raw + 16 || &bytes[..8] != DEL_MAGIC || bytes[8..12] != DEL_VERSION.to_le_bytes() {
        return None;
    }
    let (body, crc) = bytes.split_at(bytes.len() - 4);
    (crc32fast::hash(body).to_le_bytes() == crc).then(|| Bitmap::from_bytes(&body[12..], docs))
}

/// A segment's deletion file of a number (from format 7, each save of its deletions is a new
/// file of the next number); 0 is the one file, without a number, that formats before wrote.
fn del_path(dir: &Path, id: u64, number: u64) -> PathBuf {
    if number == 0 {
        return dir.join(format!("seg-{id:08}.del"));
    }
    dir.join(format!("seg-{id:08}.{number}.del"))
}

pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    crate::faults::check(crate::faults::kind(path))?;
    let mut f = File::create(&tmp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    drop(f);
    crate::faults::check("rename")?;
    std::fs::rename(tmp, path)?;
    // The rename itself, on disk: the directory's entry. (On Unix; elsewhere the rename is it.)
    if let Some(dir) = path.parent() {
        sync_dir(dir)?;
    }
    Ok(())
}

pub(crate) fn sync_dir(path: &Path) -> std::io::Result<()> {
    crate::faults::check("dir sync")?;
    #[cfg(unix)]
    File::open(if path.as_os_str().is_empty() { Path::new(".") } else { path })?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path; // No portable directory synchronization in std on these platforms.
    Ok(())
}

/// Persist every newly created path component, including the entry in its parent directory.
pub(crate) fn create_dir_all_synced(path: &Path) -> std::io::Result<()> {
    create_dir_all_with_sync(path, sync_dir)
}

fn create_dir_all_with_sync(path: &Path, mut sync: impl FnMut(&Path) -> std::io::Result<()>) -> std::io::Result<()> {
    std::fs::create_dir_all(path)?;
    // Include existing ancestors: a previous attempt may have created them but failed to sync
    // their entries. Existence on retry does not prove that those entries reached storage.
    for ancestor in path.canonicalize()?.ancestors() {
        sync(ancestor)?;
    }
    Ok(())
}

/// FNV-1a: catches a torn or garbled log record.
fn fnv(bytes: &[u8]) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Threads for building segments, apart from rayon's global pool (the application's, and where
/// they'd compete with queries): half the cores each for flushes and merges, merges at background
/// priority. A flush's k-means on every core of the global pool had been the slowest queries
/// while writing.
fn build_pool(merge: bool) -> &'static rayon::ThreadPool {
    static FLUSH: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    static MERGE: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    (if merge { &MERGE } else { &FLUSH }).get_or_init(|| {
        let n = (std::thread::available_parallelism().map_or(2, |n| n.get()) / 2).max(1);
        let b = rayon::ThreadPoolBuilder::new().num_threads(n).thread_name(move |i| format!("hinny-{}-{i}", if merge { "merge" } else { "flush" }));
        let b = if merge { b.start_handler(|_| background_priority()) } else { b };
        b.build().expect("build threads")
    })
}

/// Merges don't compete with the user: utility priority (efficiency cores on Apple silicon).
#[cfg(target_os = "macos")]
fn background_priority() {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(class: u32, priority: i32) -> i32;
    }
    const QOS_CLASS_UTILITY: u32 = 0x11;
    unsafe { pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0) };
}

/// On Linux a thread has its own nice value.
#[cfg(target_os = "linux")]
fn background_priority() {
    unsafe extern "C" {
        fn nice(inc: i32) -> i32;
    }
    unsafe { nice(10) };
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn background_priority() {}

#[cfg(test)]
mod durability_tests {
    use super::*;

    #[test]
    fn wal_write_failure_requires_reopen_even_if_io_recovers() {
        let path = std::env::temp_dir().join(format!("hinny-wal-failed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let mut ix = Index::open(&path, 1).unwrap();
        ix.put(1, 1, &[vec!["first".into()]]).unwrap();
        ix.sync().unwrap();
        let log = path.join("wal-1");
        // Simulate a partially appended length, then make the next real write fail by replacing
        // its handle with a read-only one. Capacity 1 forces the error during the append.
        OpenOptions::new().append(true).open(&log).unwrap().write_all(&[99, 0]).unwrap();
        ix.wal = Some(BufWriter::with_capacity(1, File::open(&log).unwrap()));
        assert!(ix.put(2, 1, &[vec!["failed".into()]]).is_err());
        ix.wal = Some(BufWriter::new(OpenOptions::new().append(true).open(&log).unwrap()));
        assert!(ix.put(3, 1, &[vec!["must not acknowledge".into()]]).is_err());
        assert!(ix.sync().is_err());
        assert!(ix.checkpoint().is_err());
        assert!(ix.get(2).is_none());
        drop(ix);
        let mut ix = Index::open(&path, 1).unwrap();
        assert_eq!(ix.versions().len(), 1);
        ix.put(3, 1, &[vec!["after reopen".into()]]).unwrap();
        ix.sync().unwrap();
        ix.simulate_crash();
        let ix = Index::open(&path, 1).unwrap();
        assert!(ix.get(1).is_some());
        assert!(ix.get(3).is_some());
        drop(ix);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn creating_relative_directories_syncs_the_current_directory() {
        // Avoid changing the process-wide working directory while other tests run.
        let path = PathBuf::from(format!("target/hinny-relative-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        create_dir_all_synced(&path.join("nested")).unwrap();
        sync_dir(Path::new("")).unwrap();
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn directory_sync_failure_is_retried_even_when_directories_exist() {
        let root = std::env::temp_dir().join(format!("hinny-directory-retry-{}", std::process::id()));
        let path = root.join("nested/database");
        let _ = std::fs::remove_dir_all(&root);
        let mut visited = Vec::new();
        let result = create_dir_all_with_sync(&path, |p| {
            visited.push(p.to_path_buf());
            if visited.len() == 2 {
                return Err(std::io::Error::other("injected directory sync failure"));
            }
            sync_dir(p)
        });
        assert!(result.is_err());
        assert!(path.exists());
        visited.clear();
        create_dir_all_with_sync(&path, |p| {
            visited.push(p.to_path_buf());
            sync_dir(p)
        }).unwrap();
        assert!(visited.contains(&root.canonicalize().unwrap()));
        assert!(visited.contains(&root.parent().unwrap().canonicalize().unwrap()));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejected_validation_never_replays_or_checkpoints_with_stale_fields() {
        let path = std::env::temp_dir().join(format!("hinny-validate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let mut ix = Index::open(&path, 2).unwrap();
        ix.put(1, 1, &[vec!["first field".into(), "second field".into()]]).unwrap();
        ix.simulate_crash();
        let before = std::fs::read(path.join(MANIFEST)).unwrap();
        let logs = log_files(&path).unwrap();
        // Models an opener whose pre-lock schema became stale while another writer grew it.
        assert!(Index::open_validated(&path, 1, || Err(std::io::Error::other("schema changed"))).is_err());
        assert_eq!(std::fs::read(path.join(MANIFEST)).unwrap(), before);
        assert_eq!(log_files(&path).unwrap(), logs);
        let ix = Index::open(&path, 2).unwrap();
        assert!(ix.search(&Request { text: "second", fields: &[(1, 1.0)], k: 10, all: false, expand: 0, prefix_last: 0, filter: None, sparse: &[], stop: None }).iter().any(|h| h.key == 1));
        drop(ix);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn sync_covers_a_frozen_wal_without_installing_its_segment() {
        let path = std::env::temp_dir().join(format!("hinny-sync-frozen-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let mut ix = Index::open(&path, 1).unwrap();
        ix.put(1, 1, &[vec!["frozen".into()]]).unwrap();
        ix.freeze().unwrap();
        std::fs::rename(path.join("wal-1"), path.join("saved-wal")).unwrap();
        assert!(ix.sync().is_err());
        std::fs::rename(path.join("saved-wal"), path.join("wal-1")).unwrap();
        ix.sync().unwrap();
        assert_eq!(ix.mems.len(), 2);
        assert_eq!(ix.stats().segments, 0);
        drop(ix);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn interrupted_wal_retirement_preserves_a_recoverable_suffix() {
        for failed_at in 0..2 {
            for after_unlink in [false, true] {
                let path = std::env::temp_dir().join(format!("hinny-retire-{failed_at}-{after_unlink}-{}", std::process::id()));
                let _ = std::fs::remove_dir_all(&path);
                let mut ix = Index::open(&path, 1).unwrap();
                ix.put(1, 1, &[vec!["old".into()]]).unwrap();
                ix.simulate_crash();
                let mut ix = Index::open(&path, 1).unwrap();
                ix.auto_merge = false;
                ix.put(1, 2, &[vec!["new".into()]]).unwrap();
                let logs = log_files(&path).unwrap();
                let bytes: Vec<_> = logs.iter().map(|(_, p)| std::fs::read(p).unwrap()).collect();
                ix.checkpoint().unwrap();
                // Restore the WALs to model the moment after manifest publication and before
                // retirement. The latest value is already committed in the segment.
                for ((_, p), bytes) in logs.iter().zip(bytes) {
                    write_atomic(p, &bytes).unwrap();
                }
                let mut attempted = 0;
                assert!(ix.retire_logs(logs.iter().map(|(_, p)| p.clone()).collect(), |p| {
                    let fail = attempted == failed_at;
                    attempted += 1;
                    if !fail || after_unlink {
                        std::fs::remove_file(p)?;
                    }
                    if fail {
                        return Err(std::io::Error::other("injected unlink or directory-sync failure"));
                    }
                    sync_dir(&path)
                }).is_err());
                assert_eq!(attempted, failed_at + 1, "never unlink newer WALs after an error");
                assert!(ix.sync().is_err());
                assert!(ix.put(2, 1, &[vec!["refused".into()]]).is_err());
                drop(ix);
                let ix = Index::open(&path, 1).unwrap();
                assert_eq!(ix.versions().get(&1), Some(&2));
                assert!(ix.get(2).is_none());
                drop(ix);
                std::fs::remove_dir_all(path).unwrap();
            }
        }
    }
}
