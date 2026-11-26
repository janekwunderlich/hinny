//! One immutable, memory-mapped segment of the text index: posting lists per (field, term), a term
//! dictionary, and per passage its field lengths, key, version and passage number.
//!
//! Posting lists are cut into blocks of 128 passages. A full block is bit-packed (`bitpack`):
//! document gaps, then term frequencies. The last, partial block uses variable-byte numbers, as in
//! Lucene and Tantivy (most terms occur in a few passages). A list of 128 passages or more starts
//! with a 16-byte header per block (its last document, where its data and impacts are, the bit
//! widths, its size), so skipping reads 16 bytes per 128 documents and never touches the data it
//! skips.
//!
//! Each block also has its *impacts*: the (term frequency, field length) pairs no other pair in the
//! block beats on both (a Pareto front, at most 8; Lucene's impacts). BM25 grows with frequency and
//! shrinks with length, so the best score in a block is at one of them, under any collection
//! statistics. Stored this way, block bounds stay exact when idf and average lengths are global
//! across segments and change as segments come and go; a score precomputed at build time would
//! not. Shorter lists have no headers: the dictionary holds their last document and impacts.
//!
//! The dictionary is front-coded in blocks of 32 terms (each term as the bytes it shares with the
//! one before plus the rest, numbers as variable-byte), with every block's offset in an index that
//! lookups binary-search by the blocks' first terms.
//!
//! Layout (little-endian): a 256-byte header; posting lists (written as a stream, so building and
//! merging never hold them all); the dictionary index and dictionary; then per passage: keys,
//! versions, passage numbers, the passages ordered by key, exact field lengths (u16), and length
//! codes (a byte, what scoring reads); then the attribute columns (`attrs`) and the vectors
//! (`vector`: clusters, 1-bit codes, full-precision rows), each array 16-byte aligned.
//!
//! The vectors are stored as a search reads them: the codes in FastScan's blocks of 32, what an
//! estimate needs of each vector beside its code, and the full-precision rows in the codes' order,
//! so a cluster's rows are together. A search maps them and builds nothing; before version 8 the
//! first search of a segment read every code to lay them out in memory of its own, and a rerank
//! read rows scattered over the file.
//!
//! The file ends with a CRC-32 of every mebibyte after the header (`verify` checks them), and the
//! header carries its own, checked when the segment is opened.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use memmap2::Mmap;
use rayon::prelude::*;

use crate::attrs::{Bound, Column, Kind, NO_INT, Row, Value};
use crate::bitmap::Bitmap;
use crate::bitpack::{self, BLOCK};
use crate::blob::{self, Blob, TextRef};
use crate::vector;
use crate::norms;
use crate::analysis::Analyzer;

const MAGIC: u32 = 0x5854_5753; // "SWTX"
/// 9: vectors compared by another metric than cosine (`vector::Metric`), which the vector header
/// then names; for dot product with what its search keeps per vector and per list (`along`,
/// `longest`). Only such a segment is written as 9: one whose vectors are cosine's, or that
/// has none, is version 8 to the byte, and a build from before metrics reads it as it always
/// did and refuses the others by number.
/// 8: the vectors as a search reads them (blocked codes, estimate factors, rows in code order),
/// in lists by the centroids of a generation that segments share (`vector::Generation`), and
/// checksums (of the header, and of every mebibyte after it).
/// 7: full-precision vectors in half precision (a flag in the vector header, 0 before).
/// 6: 64-bit string offsets, and stored text in blob files (references in a section of its own).
/// 5: vectors. 4: without them (its vector slot in the header is always 0).
/// How many conditions' matches a segment keeps (`Segment::matching_kept`): a bit a document
/// each, so at most 4 bytes a document.
pub const KEPT_FILTERS: usize = 32;
/// The newest version this build reads, and the one it writes for vectors of another metric
/// than cosine; every other segment is written as `COSINE_VERSION`.
const VERSION: u32 = 9;
const COSINE_VERSION: u32 = 8;
/// The bytes before a version 8 segment's centroids: what earlier versions had (32), the
/// generation, what it was trained on and measured on (16), and a probe budget per depth.
const VECTOR_HEADER: usize = 48 + 4 * vector::DEPTHS;
/// What version 9 adds to them: the metric (u32), and 12 bytes of zeros that keep the arrays
/// after it where they are aligned.
const METRIC_HEADER: usize = 16;
/// The bytes one checksum covers.
const SUM_CHUNK: usize = 1 << 20;
const SUMS_MAGIC: u32 = 0x534d_5553; // "SUMS"
/// How hinny before 0.2 stored text: string attributes `_fugu.text.0`, `_fugu.text.1`, … (one per
/// text field). Still read; a merge moves them into blob files.
pub const LEGACY_TEXT: &str = "_fugu.text.";
pub const MAX_FIELDS: usize = 8;
const HEADER: usize = 256;
pub const BLOCK_HEADER: usize = 16;
const DICT_BLOCK: usize = 32;
pub const MAX_IMPACTS: usize = 8;

// MARK: Impacts

/// The (term frequency, length code) pairs a block's best score can come from.
#[derive(Clone, Copy, Debug, Default)]
pub struct Impacts {
    pub n: u8,
    pub pairs: [(u32, u8); MAX_IMPACTS],
}

impl Impacts {
    /// The Pareto front of `pairs` (higher frequency and shorter field both win). Past
    /// `MAX_IMPACTS`, neighbours are merged into one pair with the higher frequency and the
    /// shorter length of the two, which still bounds both.
    pub fn front(pairs: &mut [(u32, u8)]) -> Impacts {
        // Shortest first; for one length, the highest frequency.
        pairs.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)));
        let mut front: Vec<(u32, u8)> = Vec::new();
        for &(tf, norm) in pairs.iter() {
            // Longer than everything kept so far: only worth keeping with a higher frequency.
            if front.last().is_none_or(|&(best, _)| tf > best) {
                front.push((tf, norm));
            }
        }
        while front.len() > MAX_IMPACTS {
            // Merge the closest neighbours in frequency.
            let i = (0..front.len() - 1).min_by_key(|&i| front[i + 1].0 - front[i].0).unwrap();
            front[i] = (front[i + 1].0, front[i].1);
            front.remove(i + 1);
        }
        let mut out = Impacts { n: front.len() as u8, ..Default::default() };
        out.pairs[..front.len()].copy_from_slice(&front);
        out
    }

    /// The best term-frequency factor, tf / (tf + k1 (1 − b + b len / avg)), for these statistics.
    #[inline]
    pub fn bound(&self, cache: &[f32; 256]) -> f32 {
        let mut best = 0f32;
        for &(tf, norm) in &self.pairs[..self.n as usize] {
            let tf = tf as f32;
            best = best.max(tf / (tf + cache[norm as usize]));
        }
        best
    }

    /// The highest term frequency (for a sparse field: the highest weight, in hundredths).
    pub fn max_tf(&self) -> f32 {
        self.pairs[..self.n as usize].iter().map(|&(tf, _)| tf).max().unwrap_or(0) as f32
    }

    fn write(&self, out: &mut Vec<u8>) {
        for &(tf, norm) in &self.pairs[..self.n as usize] {
            put_varint(out, tf as u64);
            out.push(norm);
        }
    }

    fn read(b: &[u8], at: &mut usize, n: u8) -> Impacts {
        let mut out = Impacts { n, ..Default::default() };
        for p in &mut out.pairs[..n as usize] {
            let tf = varint(b, at) as u32;
            *p = (tf, b[*at]);
            *at += 1;
        }
        out
    }

    /// Both together (for a whole list from its blocks).
    pub fn union(&self, other: &Impacts) -> Impacts {
        let mut pairs: Vec<(u32, u8)> = self.pairs[..self.n as usize].iter().chain(&other.pairs[..other.n as usize]).copied().collect();
        Impacts::front(&mut pairs)
    }
}

/// What the dictionary knows about one (field, term).
#[derive(Clone, Copy, Debug)]
pub struct TermInfo {
    pub field: u8,
    pub df: u32,
    /// File offset of the list: its block headers if it has them, else its data.
    pub postings: u64,
    /// The list's last document.
    pub last: u32,
    /// Over the whole list (an upper bound for every block).
    pub impacts: Impacts,
}

impl TermInfo {
    pub fn blocks(&self) -> u32 {
        self.df.div_ceil(BLOCK as u32)
    }

    /// Lists of a full block or more have block headers.
    pub fn has_headers(&self) -> bool {
        self.df as usize >= BLOCK
    }
}

/// One posting list in the file, block by block.
#[derive(Clone, Copy)]
pub(crate) struct ListView<'a> {
    pub bytes: &'a [u8],
    pub info: TermInfo,
    start: usize,
    headers: bool,
}

impl<'a> ListView<'a> {
    pub fn new(bytes: &'a [u8], info: TermInfo) -> ListView<'a> {
        ListView { bytes, info, start: info.postings as usize, headers: info.has_headers() }
    }

    #[inline]
    pub fn blocks(&self) -> usize {
        self.info.blocks() as usize
    }

    #[inline]
    pub fn last(&self, b: usize) -> u32 {
        if self.headers { u32_at(self.bytes, self.start + b * BLOCK_HEADER) } else { self.info.last }
    }

    /// Where block `b`'s data is, its size, and its bit widths (0, 0 for variable-byte).
    #[inline]
    pub fn block(&self, b: usize) -> (usize, usize, u8, u8) {
        if self.headers {
            let h = self.start + b * BLOCK_HEADER;
            (self.start + u32_at(self.bytes, h + 4) as usize, self.bytes[h + 10] as usize + 1, self.bytes[h + 8], self.bytes[h + 9])
        } else {
            (self.start, self.info.df as usize, 0, 0)
        }
    }

    #[inline]
    pub fn impacts(&self, b: usize) -> Impacts {
        if self.headers {
            let h = self.start + b * BLOCK_HEADER;
            let mut at = self.start + u32_at(self.bytes, h + 12) as usize;
            Impacts::read(self.bytes, &mut at, self.bytes[h + 11])
        } else {
            self.info.impacts
        }
    }

    /// Decodes block `b`: documents (absolute) and term frequencies; returns how many.
    pub fn decode(&self, b: usize, docs: &mut [u32; BLOCK], tfs: &mut [u32; BLOCK], with_tfs: bool) -> usize {
        let (at, n, gw, fw) = self.block(b);
        if n == BLOCK {
            bitpack::unpack(&self.bytes[at..], gw, docs);
            if with_tfs {
                bitpack::unpack(&self.bytes[at + 16 * gw as usize..], fw, tfs);
            }
        } else {
            let mut p = at;
            for d in &mut docs[..n] {
                *d = varint(self.bytes, &mut p) as u32;
            }
            for f in &mut tfs[..n] {
                *f = varint(self.bytes, &mut p) as u32;
            }
        }
        let mut prev = if b == 0 { -1i64 } else { self.last(b - 1) as i64 };
        for d in &mut docs[..n] {
            prev += *d as i64 + 1;
            *d = prev as u32;
        }
        n
    }

    /// Frequencies of a bit-packed block (decoded later than its documents).
    pub fn decode_tfs(&self, b: usize, tfs: &mut [u32; BLOCK]) {
        let (at, _, gw, fw) = self.block(b);
        bitpack::unpack(&self.bytes[at + 16 * gw as usize..], fw, tfs);
    }
}

// MARK: Reading

pub struct Segment {
    map: Mmap,
    pub docs: u32,
    pub fields: usize,
    pub total_tokens: [u64; MAX_FIELDS],
    terms: usize,
    dict_blocks: usize,
    dict_index: usize,
    norms: usize,
    lengths: usize,
    keys: usize,
    versions: usize,
    passages: usize,
    key_order: usize,
    /// Attribute columns: name, kind, where the data starts.
    attrs: Vec<(String, Kind, usize)>,
    /// Each column's codes and dictionary (version 8 on; none: integers and offsets as before).
    layouts: Vec<Option<crate::attrs::Layout>>,
    /// The vector section, if any documents have vectors.
    vectors: Option<VectorSection>,
    /// Its rotation, made on first use.
    rotation: std::sync::OnceLock<vector::Rotation>,
    /// Per vector, what estimates need (`vector::Pre`), made when the segment is first searched.
    pre: std::sync::OnceLock<Vec<vector::Pre>>,
    /// Its codes in FastScan's order (`vector::Blocked`), made when the segment is first searched.
    blocked: std::sync::OnceLock<vector::Blocked>,
    /// Some key has more than one passage, found on first use.
    repeats: std::sync::OnceLock<bool>,
    /// The documents the conditions asked about most recently match (`matching_kept`).
    kept: std::sync::Mutex<Vec<(Bound, std::sync::Arc<Bitmap>)>>,
    /// String offsets are 64-bit (version 6 on) instead of 32.
    wide: bool,
    /// Stored text: the blob files this segment points into, and where its references start.
    blobs: Vec<(u64, std::sync::Arc<Blob>)>,
    refs: usize,
    /// Where the checksums start and how many there are (version 8 on).
    sums: Option<(usize, usize)>,
}

/// Where each array of the vector section starts in the file.
#[derive(Clone, Copy, Debug)]
struct VectorSection {
    dims: usize,
    padded: usize,
    clusters: usize,
    count: usize,
    seed: u64,
    nprobe: usize,
    centroids: usize,
    radius: usize,
    offsets: usize,
    docs: usize,
    /// Before version 8: the codes one after another, and each vector's two factors; a search
    /// lays them out in memory.
    codes: usize,
    norms: usize,
    ips: usize,
    /// Since version 8: the codes in blocks (`vector::Blocked`: each cluster's first block, the
    /// blocks, how many) and `vector::Pre` per vector, as a search reads them.
    starts: usize,
    blocks: usize,
    block_count: usize,
    pre: usize,
    /// Rows are in the codes' order (version 8 on).
    by_code: bool,
    /// The generation its lists are of (version 8 on; 0: one cluster, or clusters of its own
    /// from before), and how many vectors that was trained on.
    generation: u64,
    trained_on: u32,
    /// Probe budgets by depth, measured when the generation was trained (`vector::Budgets`).
    budgets: vector::Budgets,
    /// How the vectors are compared (version 9 on; before, and in a version 8 segment, cosine),
    /// and for dot product (0 otherwise) where `vector::Built`'s `along` (per vector, in the
    /// codes' order) and `longest` (per list) are.
    metric: vector::Metric,
    along: usize,
    longest: usize,
    rows: usize,
    full: usize,
    /// Full-precision rows as f16 (else f32).
    half: bool,
}

/// A bit per little-endian i64 of `col`: whether it's in [lo, hi]. One comparison per value, a
/// word of bits at a time (on x86-64 compiled for AVX2 where the processor has it: SSE2, all a
/// plain build may use, has no 64-bit comparison to vectorize this with).
fn in_range(col: &[u8], lo: i64, hi: i64) -> Vec<u64> {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2")]
        unsafe fn wide(col: &[u8], lo: i64, hi: i64) -> Vec<u64> {
            in_range_portable(col, lo, hi)
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the processor has AVX2.
            return unsafe { wide(col, lo, hi) };
        }
    }
    in_range_portable(col, lo, hi)
}

#[inline(always)]
fn in_range_portable(col: &[u8], lo: i64, hi: i64) -> Vec<u64> {
    // Unsigned after subtracting lo: one comparison for both ends (NO_INT is below any lo).
    let width = hi.wrapping_sub(lo) as u64;
    col.chunks(512)
        .map(|word| {
            let mut bits = 0u64;
            for (i, x) in word.chunks_exact(8).enumerate() {
                let v = i64::from_le_bytes(x.try_into().unwrap());
                bits |= ((v.wrapping_sub(lo) as u64 <= width && lo <= hi) as u64) << i;
            }
            bits
        })
        .collect()
}

/// Which strings of a column a condition accepts.
enum Wanted<'a> {
    /// Any: the document has one.
    Any,
    /// One of these, exactly.
    Exactly(&'a [&'a str]),
    /// Those that start with this, without case (the condition already lowercased).
    Starting(&'a str),
    /// Those this is in, without case (the condition already lowercased).
    Containing(&'a str),
    /// Those the whole of which matches this pattern (`attrs::glob`); with `true`, without case
    /// (the pattern already lowercased).
    Matching(&'a str, bool),
}

impl Segment {
    pub fn open(path: &Path) -> std::io::Result<Segment> {
        let file = File::open(path)?;
        let map = unsafe { Mmap::map(&file)? };
        let bad = || std::io::Error::new(std::io::ErrorKind::InvalidData, "not a text segment");
        // Version 4 is version 5 without vectors (its vector slot in the header is always 0).
        if map.len() < HEADER || u32_at(&map, 0) != MAGIC || u32_at(&map, 4) < 4 {
            return Err(bad());
        }
        if u32_at(&map, 4) > VERSION {
            return Err(crate::index::too_new(&format!("segment {}", path.display()), u32_at(&map, 4), VERSION));
        }
        let damaged = |what: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("segment {} is damaged: {what}", path.display()));
        let summed = u32_at(&map, 4) >= 8;
        if summed && crc32fast::hash(&map[..HEADER - 4]) != u32_at(&map, HEADER - 4) {
            return Err(damaged("its header doesn't match its checksum"));
        }
        let docs = u32_at(&map, 8);
        let fields = u32_at(&map, 12) as usize;
        let at = |i: usize| u64_at(&map, 96 + i * 8) as usize;
        let s = Segment {
            docs,
            fields,
            terms: u32_at(&map, 16) as usize,
            dict_blocks: u32_at(&map, 20) as usize,
            total_tokens: std::array::from_fn(|f| u64_at(&map, 24 + f * 8)),
            dict_index: at(0),
            norms: at(2),
            lengths: at(3),
            keys: at(4),
            versions: at(5),
            passages: at(6),
            key_order: at(7),
            attrs: Vec::new(),
            layouts: Vec::new(),
            vectors: None,
            rotation: std::sync::OnceLock::new(),
            pre: std::sync::OnceLock::new(),
            blocked: std::sync::OnceLock::new(),
            repeats: std::sync::OnceLock::new(),
            kept: std::sync::Mutex::new(Vec::new()),
            wide: u32_at(&map, 4) >= 6,
            blobs: Vec::new(),
            refs: 0,
            sums: None,
            map,
        };
        if fields > MAX_FIELDS || s.norms + fields * docs as usize > s.map.len() || s.dict_index + s.dict_blocks * 8 > s.map.len() || s.key_order + docs as usize * 4 > s.map.len() {
            return Err(bad());
        }
        let mut s = s;
        let mut at = u64_at(&s.map, 160) as usize;
        let count = u32_at(&s.map, at) as usize;
        at += 4;
        for _ in 0..count {
            let len = u16::from_le_bytes([s.map[at], s.map[at + 1]]) as usize;
            let name = String::from_utf8_lossy(&s.map[at + 2..at + 2 + len]).into_owned();
            at += 2 + len;
            let kind = Kind::from_u8(s.map[at]).ok_or_else(bad)?;
            let bytes = u64_at(&s.map, at + 1) as usize;
            at += 9;
            s.attrs.push((name, kind, at));
            let layout = if summed { Some(crate::attrs::Layout::parse(kind, &s.map, at, bytes, docs as usize).ok_or_else(bad)?) } else { None };
            s.layouts.push(layout);
            at += bytes;
        }
        let vat = u64_at(&s.map, 168) as usize;
        if vat != 0 {
            let (dims, padded, clusters, count) = (u32_at(&s.map, vat) as usize, u32_at(&s.map, vat + 4) as usize, u32_at(&s.map, vat + 8) as usize, u32_at(&s.map, vat + 12) as usize);
            let seed = u64_at(&s.map, vat + 16);
            let nprobe = u32_at(&s.map, vat + 24) as usize;
            let half = u32_at(&s.map, vat + 28) == 1;
            // Version 8's header is 80 bytes longer: the generation, what it was trained on,
            // how many vectors its probe budgets were measured on, and the budgets.
            let (generation, trained_on) = if summed { (u64_at(&s.map, vat + 32), u32_at(&s.map, vat + 40)) } else { (0, 0) };
            let mut budgets = vector::Budgets::default();
            // Version 9's is longer again, by the metric; without one the vectors are cosine's.
            let with_metric = u32_at(&s.map, 4) >= 9;
            let mut metric = vector::Metric::Cosine;
            if summed {
                if vat + VECTOR_HEADER + if with_metric { METRIC_HEADER } else { 0 } > s.map.len() {
                    return Err(bad());
                }
                budgets.measured_on = u32_at(&s.map, vat + 44);
                for (i, b) in budgets.by_depth.iter_mut().enumerate() {
                    *b = u32_at(&s.map, vat + 48 + i * 4);
                }
                if with_metric {
                    metric = vector::Metric::from_code(u32_at(&s.map, vat + VECTOR_HEADER)).ok_or_else(|| damaged("its vectors are of a metric this hinny doesn't know"))?;
                }
            }
            let mut at = vat + if with_metric { VECTOR_HEADER + METRIC_HEADER } else if summed { VECTOR_HEADER } else { 32 };
            let mut take = |bytes: usize| {
                let start = at;
                at = (at + bytes).next_multiple_of(16);
                start
            };
            let centroids = take(clusters * padded * 4);
            let radius = take(clusters * 4);
            let offsets = take((clusters + 1) * 4);
            let docs_at = take(count * 4);
            let (mut codes, mut norms_at, mut ips, mut starts, mut blocks, mut block_count, mut pre) = (0, 0, 0, 0, 0, 0, 0);
            if summed {
                starts = take((clusters + 1) * 4);
                // How many blocks there are is the last of their starts.
                if starts + (clusters + 1) * 4 > s.map.len() {
                    return Err(bad());
                }
                block_count = u32_at(&s.map, starts + clusters * 4) as usize;
                blocks = take(block_count.checked_mul(padded * 4).ok_or_else(bad)?);
                pre = take(count * std::mem::size_of::<vector::Pre>());
            } else {
                codes = take(count * padded / 8);
                norms_at = take(count * 4);
                ips = take(count * 4);
            }
            let (along, longest) = if metric == vector::Metric::Dot { (take(count * 4), take(clusters * 4)) } else { (0, 0) };
            let rows = take(docs as usize * 4);
            let full = take(count * dims * if half { 2 } else { 4 });
            if at > s.map.len() + 16 {
                return Err(bad());
            }
            s.vectors = Some(VectorSection { dims, padded, clusters, count, seed, nprobe, centroids, radius, offsets, docs: docs_at, codes, norms: norms_at, ips, starts, blocks, block_count, pre, by_code: summed, generation, trained_on, budgets, metric, along, longest, rows, full, half });
        }
        let sat = if s.wide { u64_at(&s.map, 176) as usize } else { 0 };
        if sat != 0 {
            let n = u32_at(&s.map, sat) as usize;
            let dir = path.parent().unwrap_or(Path::new("."));
            for i in 0..n {
                let id = u64_at(&s.map, sat + 8 + i * 8);
                s.blobs.push((id, std::sync::Arc::new(Blob::open(&blob::path(dir, id))?)));
            }
            s.refs = sat + 8 + n * 8;
            if s.refs + s.docs as usize * 16 > s.map.len() {
                return Err(bad());
            }
        }
        if summed {
            // The checksums: where they start (after everything they cover), how many, and their own.
            let at = u64_at(&s.map, 184) as usize;
            let short = || damaged("its checksums are cut short");
            if at < HEADER || at.checked_add(16).is_none_or(|end| end > s.map.len()) || u32_at(&s.map, at) != SUMS_MAGIC {
                return Err(short());
            }
            let count = u32_at(&s.map, at + 8) as usize;
            let end = at + 12 + count * 4;
            if u32_at(&s.map, at + 4) as usize != SUM_CHUNK || count != (at - HEADER).div_ceil(SUM_CHUNK) || end + 4 > s.map.len() {
                return Err(short());
            }
            if crc32fast::hash(&s.map[at..end]) != u32_at(&s.map, end) {
                return Err(damaged("its checksums don't match their own"));
            }
            s.sums = Some((at, count));
        }
        // Queries jump around the posting lists.
        unsafe { madvise_random(s.map.as_ptr(), s.map.len()) };
        Ok(s)
    }

    /// Reads the whole file against the checksums written with it: an error names the first
    /// mebibyte that differs. Segments from before version 8 have none, and pass.
    pub fn verify(&self) -> std::io::Result<()> {
        let Some((at, _)) = self.sums else { return Ok(()) };
        self.map[HEADER..at].par_chunks(SUM_CHUNK).enumerate().try_for_each(|(i, chunk)| {
            if crc32fast::hash(chunk) == u32_at(&self.map, at + 12 + i * 4) {
                return Ok(());
            }
            let from = HEADER + i * SUM_CHUNK;
            Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bytes {from} to {} don't match their checksum", from + chunk.len())))
        })
    }

    /// Whether the file carries checksums for `verify` to check (version 8 on).
    pub fn has_checksums(&self) -> bool {
        self.sums.is_some()
    }

    /// The blob files this segment points into, to read (`Blob::verify`).
    pub fn blob_files(&self) -> impl Iterator<Item = (u64, &Blob)> + '_ {
        self.blobs.iter().map(|(id, b)| (*id, &**b))
    }

    fn slice<T>(&self, at: usize, len: usize) -> &[T] {
        // Arrays in the vector section are 16-byte aligned in the file, and the map is page aligned.
        debug_assert!(at % std::mem::align_of::<T>() == 0);
        unsafe { std::slice::from_raw_parts(self.map.as_ptr().add(at) as *const T, len) }
    }

    /// The vector dimensions, if this segment has vectors.
    pub fn vector_dims(&self) -> Option<usize> {
        self.vectors.map(|v| v.dims)
    }

    /// This segment's vector index, for searching.
    /// How many vectors this segment holds.
    pub fn vector_count(&self) -> usize {
        self.vectors.map_or(0, |v| v.count)
    }

    /// How this segment's vectors are compared, if it has vectors (cosine for every segment
    /// from before there were other metrics).
    pub fn vector_metric(&self) -> Option<vector::Metric> {
        self.vectors.map(|v| v.metric)
    }

    /// Its rotation alone, without the scan caches `vector_view` builds (for a query that may
    /// compare exactly instead).
    pub fn rotation(&self) -> Option<&vector::Rotation> {
        let v = self.vectors?;
        Some(self.rotation.get_or_init(|| vector::Rotation::new(v.dims, v.seed)))
    }

    /// What a search's plan needs to know of this segment's vector index, without opening it:
    /// its measured probe budgets (none measured: `by_depth[0]` is 0), the budget for the ten
    /// nearest, its clusters and its vectors.
    pub fn vector_budgets(&self) -> Option<(vector::Budgets, usize, usize, usize)> {
        self.vectors.map(|v| (v.budgets, v.nprobe, v.clusters, v.count))
    }

    /// The generation this segment's lists are of, with its centroids, for building another
    /// segment into it or searching it together with others of it. None for a segment of one
    /// cluster and for segments from before version 8 (clusters of their own).
    pub fn vector_generation(&self) -> Option<vector::Generation<'_>> {
        let v = self.vectors.filter(|v| v.generation != 0 && v.clusters > 1)?;
        Some(vector::Generation { id: v.generation, trained_on: v.trained_on, nprobe: v.nprobe as u32, budgets: v.budgets, centroids: self.slice(v.centroids, v.clusters * v.padded) })
    }

    /// A document's vector as this segment's lists hold it: its list, its code (into `code`),
    /// its factors, and for dot product how many times its centroid it holds (0 otherwise).
    /// For a merge into the same generation, which copies them.
    pub fn vector_coded(&self, doc: u32, code: &mut [u64]) -> Option<(u32, vector::Pre, f32)> {
        let v = self.vectors.filter(|v| v.by_code)?;
        let j = *self.slice::<u32>(v.rows, self.docs as usize).get(doc as usize)? as usize;
        if j == u32::MAX as usize {
            return None;
        }
        let offsets = self.slice::<u32>(v.offsets, v.clusters + 1);
        let list = offsets.partition_point(|&o| o as usize <= j) - 1;
        let starts = self.slice::<u32>(v.starts, v.clusters + 1);
        let block = v.padded * 4;
        let blocks = self.slice::<u8>(v.blocks, v.block_count * block);
        vector::unblock(&blocks[starts[list] as usize * block..], block, j - offsets[list] as usize, code);
        let along = if v.along != 0 { self.slice::<f32>(v.along, v.count)[j] } else { 0.0 };
        Some((list as u32, self.slice::<vector::Pre>(v.pre, v.count)[j], along))
    }

    /// Memory its vector scan caches take, once a search built them (FastScan's blocks and the
    /// per-vector factors): about 16 + dimensions / 8 bytes a vector for a segment from before
    /// version 8, and nothing since (a search reads them from the file).
    pub fn vector_cache_bytes(&self) -> usize {
        self.blocked.get().map_or(0, |b| b.bytes.len() + b.starts.len() * 4) + self.pre.get().map_or(0, |p| p.len() * std::mem::size_of::<vector::Pre>())
    }

    pub fn vector_view(&self) -> Option<vector::View<'_>> {
        let v = self.vectors?;
        // As stored (version 8 on), or laid out in memory the first time (before).
        let (pre, blocks, starts): (&[vector::Pre], &[u8], &[u32]) = if v.by_code {
            (self.slice(v.pre, v.count), self.slice(v.blocks, v.block_count * v.padded * 4), self.slice(v.starts, v.clusters + 1))
        } else {
            let blocked = self.blocked.get_or_init(|| vector::Blocked::new(self.slice(v.codes, v.count * v.padded / 64), self.slice(v.offsets, v.clusters + 1), v.padded / 64));
            let pre = self.pre.get_or_init(|| vector::Pre::all(self.slice(v.norms, v.count), self.slice(v.ips, v.count), self.slice(v.codes, v.count * v.padded / 64), v.padded / 64));
            (&pre[..], &blocked.bytes[..], &blocked.starts[..])
        };
        Some(vector::View {
            rotation: self.rotation.get_or_init(|| vector::Rotation::new(v.dims, v.seed)),
            centroids: self.slice(v.centroids, v.clusters * v.padded),
            radius: self.slice(v.radius, v.clusters),
            offsets: self.slice(v.offsets, v.clusters + 1),
            docs: self.slice(v.docs, v.count),
            pre,
            blocks,
            starts,
            block: v.padded * 4,
            rows: self.slice(v.rows, self.docs as usize),
            by_code: v.by_code,
            full: self.rows_of(v),
            nprobe: v.nprobe,
            budgets: v.budgets,
            metric: v.metric,
            along: if v.along != 0 { self.slice(v.along, v.count) } else { &[] },
            longest: if v.longest != 0 { self.slice(v.longest, v.clusters) } else { &[] },
        })
    }

    fn rows_of(&self, v: VectorSection) -> vector::Rows<'_> {
        if v.half { vector::Rows::F16(self.slice(v.full, v.count * v.dims)) } else { vector::Rows::F32(self.slice(v.full, v.count * v.dims)) }
    }

    /// The cluster a document's vector is in (for explaining misses: a scan of every vector).
    pub fn vector_cluster(&self, doc: u32) -> Option<u32> {
        let v = self.vectors?;
        let j = self.slice::<u32>(v.docs, v.count).iter().position(|&d| d == doc)?;
        let offsets = self.slice::<u32>(v.offsets, v.clusters + 1);
        Some((offsets.partition_point(|&o| o as usize <= j) - 1) as u32)
    }

    /// A document's vector at the precision it is kept (as its metric stored it: normalized
    /// for cosine).
    pub fn vector(&self, doc: u32) -> Option<Vec<f32>> {
        let v = self.vectors?;
        let row = *self.slice::<u32>(v.rows, self.docs as usize).get(doc as usize)?;
        (row != u32::MAX).then(|| self.rows_of(v).row(row as usize, v.dims))
    }

    /// A document's vector where the segment keeps it (f16 rows, or f32 before version 7).
    pub fn vector_ref(&self, doc: u32) -> Option<vector::VectorRef<'_>> {
        let v = self.vectors?;
        let row = *self.slice::<u32>(v.rows, self.docs as usize).get(doc as usize)? as usize;
        (row != u32::MAX as usize).then(|| match self.rows_of(v) {
            vector::Rows::F32(f) => vector::VectorRef::F32(&f[row * v.dims..(row + 1) * v.dims]),
            vector::Rows::F16(h) => vector::VectorRef::F16(&h[row * v.dims..(row + 1) * v.dims]),
            vector::Rows::Refs(r) => r[row].1,
        })
    }

    /// Whether a document has a vector.
    pub fn has_vector(&self, doc: u32) -> bool {
        self.vectors.is_some_and(|v| self.slice::<u32>(v.rows, self.docs as usize).get(doc as usize).is_some_and(|&r| r != u32::MAX))
    }

    /// What the segment's metric orders by (`vector::Metric`: the squared distance from `q`,
    /// or for dot product the product, negated) for every vector `keep` keeps: `each(document,
    /// key)`, in the order the rows are stored, so that scoring everything reads the file front
    /// to back (by cluster since version 8, by document before).
    pub fn vector_keys(&self, q: &[f32], mut keep: impl FnMut(u32) -> bool, mut each: impl FnMut(u32, f32)) {
        let Some(v) = self.vectors else { return };
        let rows = self.rows_of(v);
        // (The metric is asked once, not for every row: with the question in the loop, scoring
        // a million rows by distance took 1% longer than before there was another metric.)
        if v.metric == vector::Metric::Dot {
            for (row, &doc) in self.slice::<u32>(v.docs, v.count).iter().enumerate() {
                if keep(doc) {
                    each(doc, -rows.dot(q, row, v.dims));
                }
            }
        } else if v.by_code {
            for (row, &doc) in self.slice::<u32>(v.docs, v.count).iter().enumerate() {
                if keep(doc) {
                    each(doc, rows.dist2(q, row, v.dims));
                }
            }
        } else {
            for (doc, &row) in self.slice::<u32>(v.rows, self.docs as usize).iter().enumerate() {
                if row != u32::MAX && keep(doc as u32) {
                    each(doc as u32, rows.dist2(q, row as usize, v.dims));
                }
            }
        }
    }

    /// What the segment's metric orders a document's vector by against `q`, if it has one.
    pub fn vector_key(&self, doc: u32, q: &[f32]) -> Option<f32> {
        let v = self.vectors?;
        let row = *self.slice::<u32>(v.rows, self.docs as usize).get(doc as usize)?;
        (row != u32::MAX).then(|| self.rows_of(v).key(v.metric, q, row as usize, v.dims))
    }

    /// The attribute columns (name, kind), in column order.
    /// Where a passage's stored text is, if it has some.
    pub fn text_ref(&self, doc: u32) -> Option<TextRef> {
        if self.blobs.is_empty() {
            return None;
        }
        let at = self.refs + doc as usize * 16;
        let i = u32_at(&self.map, at);
        let (id, _) = self.blobs.get(i as usize)?;
        Some(TextRef { blob: *id, inner: u32_at(&self.map, at + 4), block: u64_at(&self.map, at + 8) })
    }

    /// A passage's stored text, one per field: from its blob file, or from the attributes of
    /// segments written before blob files.
    pub fn stored(&self, doc: u32) -> Option<Vec<String>> {
        if let Some(r) = self.text_ref(doc) {
            return self.blobs.iter().find(|(id, _)| *id == r.blob)?.1.read(r);
        }
        let legacy: Vec<(usize, String)> = self
            .attributes(doc)
            .into_iter()
            .filter_map(|(n, v)| match (n.strip_prefix(LEGACY_TEXT).and_then(|i| i.parse::<usize>().ok()), v) {
                (Some(i), Value::Str(t)) => Some((i, t)),
                _ => None,
            })
            .collect();
        if legacy.is_empty() {
            return None;
        }
        let mut out = vec![String::new(); self.fields.max(legacy.iter().map(|(i, _)| i + 1).max().unwrap_or(0))];
        for (i, t) in legacy {
            out[i] = t;
        }
        Some(out)
    }

    /// The blob files this segment points into, with their record counts.
    pub fn blobs(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.blobs.iter().map(|(id, b)| (*id, b.records))
    }

    /// Which of `blobs()` holds a passage's stored text (its place in that order), if it has some.
    pub fn text_blob(&self, doc: u32) -> Option<usize> {
        if self.blobs.is_empty() {
            return None;
        }
        let i = u32_at(&self.map, self.refs + doc as usize * 16) as usize;
        (i < self.blobs.len()).then_some(i)
    }

    pub fn attr_columns(&self) -> Vec<(String, Kind)> {
        self.attrs.iter().map(|(n, k, _)| (n.clone(), *k)).collect()
    }

    /// One document's attributes, for filters.
    pub fn row(&self, doc: u32) -> SegmentRow<'_> {
        SegmentRow { seg: self, doc }
    }

    /// The documents whose integer in column `c` is in [lo, hi] (none if it isn't an integer
    /// column). One comparison per document over the mapped column, a word of bits at a time.
    fn ints(&self, c: usize, lo: i64, hi: i64) -> Option<Bitmap> {
        let &(_, kind, at) = self.attrs.get(c)?;
        // (A column of floats is one of integers here: `attrs::float_code`.)
        if matches!(kind, Kind::Str | Kind::Tags) {
            return None;
        }
        let n = self.docs as usize;
        let words = match &self.layouts[c] {
            Some(layout) => match layout.codes_between(lo, hi) {
                Some((first, last)) => layout.in_range(&self.map, first, last),
                None => return Some(Bitmap::new(n)),
            },
            None => in_range(&self.map[at..at + n * 8], lo, hi),
        };
        Some(Bitmap::of_words(n, words))
    }

    /// The documents whose integer in column `c` is one of `values` (none if it isn't an integer
    /// column stored as codes: version 8 on).
    fn ints_among(&self, c: usize, values: &[Value]) -> Option<Bitmap> {
        let layout = self.layouts.get(c)?.as_ref().filter(|l| l.kind == Kind::Int)?;
        let mut codes: Vec<u64> = values.iter().filter_map(|v| match v {
            Value::Int(x) if *x != NO_INT => layout.codes_between(*x, *x).map(|(code, _)| code),
            _ => None,
        }).collect();
        codes.sort_unstable();
        codes.dedup();
        let words = layout.in_runs(&self.map, &crate::attrs::runs_of(&codes));
        Some(Bitmap::of_words(self.docs as usize, words))
    }

    /// The documents whose string in column `c` is one `wanted` accepts, or one of whose tags
    /// is (none if it isn't a column of strings stored with a dictionary: version 8 on). The
    /// strings are found in the dictionary, where each is once and in order (one by binary
    /// search, a prefix as the ranges that start with it, a part by one pass over the
    /// dictionary's bytes), and the documents by their codes (tags: from the documents kept
    /// with each).
    fn strings(&self, c: usize, wanted: Wanted<'_>) -> Option<Bitmap> {
        use crate::attrs::{INTO_ASCII, lower, runs_of};
        let layout = self.layouts.get(c)?.as_ref().filter(|l| matches!(l.kind, Kind::Str | Kind::Tags))?;
        if layout.kind == Kind::Tags && matches!(wanted, Wanted::Any) {
            return Some(Bitmap::of_words(self.docs as usize, layout.tagged(&self.map)));
        }
        // The codes of the strings `accepts` accepts, one string at a time: for what the
        // dictionary's bytes don't answer as they are.
        let each = |accepts: &dyn Fn(&str) -> bool| {
            let mut codes = Vec::new();
            layout.each_string(&self.map, |code, s| {
                if accepts(s) {
                    codes.push(code);
                }
            });
            codes
        };
        let runs = match wanted {
            Wanted::Any if layout.top == 0 => Vec::new(),
            Wanted::Any => vec![(1, layout.top)],
            Wanted::Exactly(values) => {
                let mut codes: Vec<u64> = values.iter().filter_map(|v| layout.find(&self.map, v)).collect();
                codes.sort_unstable();
                codes.dedup();
                runs_of(&codes)
            }
            Wanted::Starting(p) if p.is_ascii() => layout.starting_with(&self.map, p),
            // A prefix or a part with a character outside ASCII: only strings with one can have
            // it, lowercased as `Bound::matches` does.
            Wanted::Starting(p) => runs_of(&each(&|s| !s.is_ascii() && lower(s).starts_with(p))),
            Wanted::Containing("") => vec![(1, layout.top)].into_iter().filter(|_| layout.top > 0).collect(),
            Wanted::Containing(p) if p.is_ascii() => {
                let mut codes = layout.containing(&self.map, p);
                if layout.into_ascii {
                    // Strings whose lowercase has ASCII their bytes don't.
                    codes.extend(each(&|s| !s.is_ascii() && s.contains(INTO_ASCII) && lower(s).contains(p)));
                    codes.sort_unstable();
                    codes.dedup();
                }
                runs_of(&codes)
            }
            Wanted::Containing(p) => runs_of(&each(&|s| !s.is_ascii() && lower(s).contains(p))),
            // With case: only among the strings that start as the pattern does. Without: every
            // string, lowercased as `Bound::matches` does.
            Wanted::Matching(p, false) => {
                let (first, end) = layout.with_prefix(&self.map, crate::attrs::glob_prefix(p));
                runs_of(&(first..end).filter(|&code| layout.string(&self.map, code).is_some_and(|s| crate::attrs::glob(s, p))).collect::<Vec<_>>())
            }
            Wanted::Matching(p, true) => runs_of(&each(&|s| crate::attrs::glob(&lower(s), p))),
        };
        let words = if layout.kind == Kind::Tags { layout.having(&self.map, &runs) } else { layout.in_runs(&self.map, &runs) };
        Some(Bitmap::of_words(self.docs as usize, words))
    }

    /// The documents `bound` matches, if a sample of 256 says there are at most about `most`.
    pub fn matching_at_most(&self, bound: &Bound, most: usize) -> Option<std::sync::Arc<Bitmap>> {
        (self.sample_share(bound) * self.docs as f64 <= 2.0 * most as f64).then(|| self.matching_kept(bound))
    }

    /// `matching`, remembered. A segment doesn't change, so the documents a condition matches
    /// don't either, and a filter that is asked again (a tenant's, an owner's, a kind's: most
    /// are) costs finding it among those kept, not a pass over its column. Kept are the
    /// `KEPT_FILTERS` conditions asked about most recently, a bit a document each: conditions
    /// on one column, so "this owner and since yesterday" keeps the owner's however the date
    /// moves; ranges aren't kept (a pass over a column of integers is the cheapest there is,
    /// and their ends are rarely the same twice). What's kept goes when the segment does.
    pub fn matching_kept(&self, bound: &Bound) -> std::sync::Arc<Bitmap> {
        use std::sync::Arc;
        let n = self.docs as usize;
        match bound {
            Bound::And(bs) if !bs.is_empty() => bs[1..].iter().fold(self.matching_kept(&bs[0]), |m, b| Arc::new(m.and(&self.matching_kept(b)))),
            Bound::Or(bs) if !bs.is_empty() => bs[1..].iter().fold(self.matching_kept(&bs[0]), |m, b| Arc::new(m.or(&self.matching_kept(b)))),
            Bound::Not(b) => Arc::new(self.matching_kept(b).not(n)),
            Bound::Range(..) | Bound::And(_) | Bound::Or(_) => Arc::new(self.matching(bound)),
            _ => {
                {
                    let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(i) = kept.iter().position(|(b, _)| b == bound) {
                        // To the front: the last asked about is the last to go.
                        kept[..=i].rotate_right(1);
                        return kept[0].1.clone();
                    }
                }
                // Not while others wait: two that ask at once both look, and one is kept.
                let bits = Arc::new(self.matching(bound));
                let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
                if !kept.iter().any(|(b, _)| b == bound) {
                    kept.insert(0, (bound.clone(), bits.clone()));
                    kept.truncate(KEPT_FILTERS);
                }
                bits
            }
        }
    }

    /// How many conditions' documents are kept (`matching_kept`), and the bytes they take.
    pub fn kept_filters(&self) -> (usize, usize) {
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        (kept.len(), kept.len() * (self.docs as usize).div_ceil(64) * 8)
    }

    /// The share of documents `bound` matches, in a sample of 256 spread over the segment.
    pub fn sample_share(&self, bound: &Bound) -> f64 {
        let n = self.docs as usize;
        let step = (n / 256).max(1);
        let sampled = n.div_ceil(step);
        let hits = (0..n).step_by(step).filter(|&d| bound.matches(&self.row(d as u32))).count();
        hits as f64 / sampled.max(1) as f64
    }

    /// Whether `matching` answers `bound` from columns alone, no document read one at a time:
    /// everything but conditions on a string column from before version 8. Then it costs a
    /// fraction of a millisecond per million documents, whatever share of them passes.
    pub fn by_column(&self, bound: &Bound) -> bool {
        let coded = |c: &usize| self.layouts.get(*c).is_some_and(|l| l.is_some());
        match bound {
            Bound::Eq(Some((c, Kind::Str | Kind::Tags)), Value::Str(_)) | Bound::Exists(Some((c, Kind::Str | Kind::Tags))) | Bound::Prefix(Some(c), _) | Bound::Contains(Some(c), _) | Bound::Glob(Some(c), ..) => coded(c),
            Bound::In(Some((c, Kind::Str | Kind::Tags)), values) => values.is_empty() || coded(c),
            Bound::And(bs) | Bound::Or(bs) => bs.iter().all(|b| self.by_column(b)),
            Bound::Not(b) => self.by_column(b),
            _ => true,
        }
    }

    /// The documents `bound` matches. Conditions on a column are checked a column at a time, on
    /// its codes (integers: a compare per document, which vectorizes; strings: looked up in the
    /// dictionary, then a compare per document with their codes); what's left, a document at a
    /// time through `Row` (five times as long for integers; for strings, every document's
    /// string instead of the dictionary): string columns from before version 8, and values of a
    /// kind the column isn't.
    pub fn matching(&self, bound: &Bound) -> Bitmap {
        let n = self.docs as usize;
        let fast = match bound {
            Bound::Eq(Some((c, Kind::Int)), Value::Int(x)) if *x != NO_INT => self.ints(*c, *x, *x),
            Bound::Range(Some(c), lo, hi) => self.ints(*c, lo.unwrap_or(NO_INT + 1).max(NO_INT + 1), hi.unwrap_or(i64::MAX)),
            Bound::Exists(Some((c, Kind::Int))) => self.ints(*c, NO_INT + 1, i64::MAX),
            Bound::In(Some((c, Kind::Int)), values) if !values.is_empty() => self.ints_among(*c, values).or_else(|| {
                let each: Option<Vec<Bitmap>> = values.iter().map(|v| match v {
                    Value::Int(x) if *x != NO_INT => self.ints(*c, *x, *x),
                    _ => Some(Bitmap::new(n)),
                }).collect();
                each.map(|each| each[1..].iter().fold(each[0].clone(), |m, b| m.or(b)))
            }),
            Bound::Eq(Some((c, Kind::Str | Kind::Tags)), Value::Str(x)) => self.strings(*c, Wanted::Exactly(&[x.as_str()])),
            Bound::In(Some((c, Kind::Str | Kind::Tags)), values) => {
                let strings: Vec<&str> = values.iter().filter_map(|v| match v {
                    Value::Str(x) => Some(x.as_str()),
                    Value::Int(_) | Value::Float(_) | Value::Tags(_) => None,
                }).collect();
                self.strings(*c, Wanted::Exactly(&strings))
            }
            Bound::Exists(Some((c, Kind::Str | Kind::Tags))) => self.strings(*c, Wanted::Any),
            // As `Bound::matches` compares: without case, the condition already lowercased.
            Bound::Prefix(Some(c), p) => self.strings(*c, Wanted::Starting(p)),
            Bound::Contains(Some(c), p) => self.strings(*c, Wanted::Containing(p)),
            Bound::Glob(Some(c), p, without_case) => self.strings(*c, Wanted::Matching(p, *without_case)),
            Bound::And(bs) if !bs.is_empty() => Some(bs[1..].iter().fold(self.matching(&bs[0]), |m, b| m.and(&self.matching(b)))),
            Bound::Or(bs) if !bs.is_empty() => Some(bs[1..].iter().fold(self.matching(&bs[0]), |m, b| m.or(&self.matching(b)))),
            Bound::And(_) => Some(Bitmap::new(n).not(n)),
            Bound::Or(_) => Some(Bitmap::new(n)),
            Bound::Not(b) => Some(self.matching(b).not(n)),
            // What no document can match: an attribute this segment doesn't have, a value of
            // another kind than the column's, one of no values.
            Bound::Eq(None, _) | Bound::In(None, _) | Bound::Range(None, ..) | Bound::Prefix(None, _) | Bound::Contains(None, _) | Bound::Glob(None, ..) | Bound::Exists(None) => Some(Bitmap::new(n)),
            Bound::Eq(Some((_, Kind::Int | Kind::Float)), _) | Bound::Eq(Some((_, Kind::Str | Kind::Tags)), Value::Int(_) | Value::Float(_) | Value::Tags(_)) => Some(Bitmap::new(n)),
            Bound::In(_, values) if values.is_empty() => Some(Bitmap::new(n)),
            _ => None,
        };
        fast.unwrap_or_else(|| Bitmap::from_words(n, |w| (0..64.min(n - w * 64)).fold(0u64, |bits, i| bits | (bound.matches(&self.row((w * 64 + i) as u32)) as u64) << i)))
    }

    /// One document's attribute values (those it has).
    pub fn attributes(&self, doc: u32) -> Vec<(String, Value)> {
        let row = self.row(doc);
        self.attrs
            .iter()
            .enumerate()
            .filter_map(|(c, (n, k, _))| {
                let v = match k {
                    Kind::Int => row.int(c).map(Value::Int),
                    Kind::Float => row.int(c).map(|x| Value::Float(crate::attrs::code_float(x))),
                    Kind::Str => row.str(c).map(|s| Value::Str(s.to_string())),
                    Kind::Tags => {
                        let mut tags = Vec::new();
                        row.some_tag(c, &mut |s| {
                            tags.push(s.to_string());
                            false
                        });
                        (!tags.is_empty()).then_some(Value::Tags(tags))
                    }
                };
                v.map(|v| (n.clone(), v))
            })
            .collect()
    }

    pub fn bytes(&self) -> &[u8] {
        &self.map
    }

    pub fn average_length(&self, field: usize) -> f32 {
        self.total_tokens[field] as f32 / self.docs.max(1) as f32
    }

    pub fn norms(&self, field: usize) -> &[u8] {
        let start = self.norms + field * self.docs as usize;
        &self.map[start..start + self.docs as usize]
    }

    pub fn length(&self, doc: u32, field: usize) -> u16 {
        let at = self.lengths + (field * self.docs as usize + doc as usize) * 2;
        u16::from_le_bytes([self.map[at], self.map[at + 1]])
    }

    pub fn key(&self, doc: u32) -> u64 {
        u64_at(&self.map, self.keys + doc as usize * 8)
    }

    pub fn version(&self, doc: u32) -> u64 {
        u64_at(&self.map, self.versions + doc as usize * 8)
    }

    pub fn passage(&self, doc: u32) -> u32 {
        u32_at(&self.map, self.passages + doc as usize * 4)
    }

    /// The `i`-th passage in the order of keys (a key's passages one after another).
    pub fn by_key(&self, i: usize) -> u32 {
        u32_at(&self.map, self.key_order + i * 4)
    }

    /// A column's codes, zones and dictionary (version 8 on), with the bytes they are in.
    pub fn codes(&self, c: usize) -> Option<(&crate::attrs::Layout, &[u8])> {
        Some((self.layouts.get(c)?.as_ref()?, &self.map[..]))
    }

    /// An integer column's (version 8 on).
    pub fn int_codes(&self, c: usize) -> Option<(&crate::attrs::Layout, &[u8])> {
        self.codes(c).filter(|(l, _)| l.kind == Kind::Int)
    }

    /// Whether some key has more than one passage here (live or deleted): neighbours in key order.
    pub fn repeats_keys(&self) -> bool {
        *self.repeats.get_or_init(|| {
            let order = |i: usize| u32_at(&self.map, self.key_order + i * 4);
            (1..self.docs as usize).any(|i| self.key(order(i - 1)) == self.key(order(i)))
        })
    }

    /// The passages of one key (an email, a file).
    pub fn docs_with_key(&self, key: u64) -> Vec<u32> {
        let order = |i: usize| u32_at(&self.map, self.key_order + i * 4);
        let (mut lo, mut hi) = (0, self.docs as usize);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.key(order(mid)) < key { lo = mid + 1 } else { hi = mid }
        }
        let mut out = Vec::new();
        while lo < self.docs as usize && self.key(order(lo)) == key {
            out.push(order(lo));
            lo += 1;
        }
        out
    }

    pub fn term_count(&self) -> usize {
        self.terms
    }

    fn block_at(&self, b: usize) -> usize {
        u64_at(&self.map, self.dict_index + b * 8) as usize
    }

    /// A dictionary block's first key (field byte, then the term).
    fn first_key(&self, b: usize) -> &[u8] {
        let mut at = self.block_at(b);
        let len = varint(&self.map, &mut at) as usize;
        &self.map[at..at + len]
    }

    /// The last dictionary block whose first key is ≤ `key` (where `key` would be).
    fn block_for(&self, key: &[u8]) -> usize {
        let (mut lo, mut hi) = (0, self.dict_blocks);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.first_key(mid) <= key { lo = mid } else { hi = mid }
        }
        lo
    }

    /// Every dictionary entry in key order.
    pub fn entries(&self) -> DictIter<'_> {
        DictIter { seg: self, block: 0, i: 0, at: 0, key: Vec::with_capacity(48), postings: 0 }
    }

    /// Calls `f` with each (key, info) from `key`'s position on, in key order, until it returns false.
    fn scan(&self, key: &[u8], mut f: impl FnMut(&[u8], TermInfo) -> bool) {
        if self.dict_blocks == 0 {
            return;
        }
        let mut it = DictIter { seg: self, block: self.block_for(key), i: 0, at: 0, key: Vec::with_capacity(48), postings: 0 };
        while let Some(info) = it.next_entry() {
            if it.key.as_slice() < key {
                continue;
            }
            if !f(&it.key, info) {
                return;
            }
        }
    }

    pub fn lookup(&self, field: u8, term: &str) -> Option<TermInfo> {
        let mut key = Vec::with_capacity(term.len() + 1);
        key.push(field);
        key.extend_from_slice(term.as_bytes());
        let mut found = None;
        self.scan(&key, |k, info| {
            if k == key.as_slice() {
                found = Some(info);
            }
            false
        });
        found
    }

    /// Every term of `field` that starts with `prefix`, in term order.
    pub fn with_prefix(&self, field: u8, prefix: &str) -> Vec<(String, TermInfo)> {
        let mut key = Vec::with_capacity(prefix.len() + 1);
        key.push(field);
        key.extend_from_slice(prefix.as_bytes());
        let mut out = Vec::new();
        self.scan(&key, |k, info| {
            if !k.starts_with(&key) {
                return false;
            }
            out.push((String::from_utf8_lossy(&k[1..]).into_owned(), info));
            true
        });
        out
    }

    /// A whole posting list: documents and term frequencies.
    pub fn read_list(&self, info: TermInfo) -> (Vec<u32>, Vec<u32>) {
        let list = ListView::new(&self.map, info);
        let (mut docs, mut tfs) = (Vec::with_capacity(info.df as usize), Vec::with_capacity(info.df as usize));
        let (mut d, mut t) = ([0u32; BLOCK], [0u32; BLOCK]);
        for b in 0..list.blocks() {
            let n = list.decode(b, &mut d, &mut t, true);
            docs.extend_from_slice(&d[..n]);
            tfs.extend(t[..n].iter().map(|f| f + 1));
        }
        (docs, tfs)
    }
}

pub struct SegmentRow<'a> {
    seg: &'a Segment,
    doc: u32,
}

impl Row for SegmentRow<'_> {
    #[inline]
    fn int(&self, column: usize) -> Option<i64> {
        let (_, kind, at) = self.seg.attrs.get(column)?;
        if matches!(kind, Kind::Str | Kind::Tags) {
            return None;
        }
        if let Some(layout) = &self.seg.layouts[column] {
            return layout.int(&self.seg.map, self.doc as usize);
        }
        let x = u64_at(&self.seg.map, at + self.doc as usize * 8) as i64;
        (x != NO_INT).then_some(x)
    }

    #[inline]
    fn str(&self, column: usize) -> Option<&str> {
        let (_, kind, at) = self.seg.attrs.get(column)?;
        if *kind != Kind::Str {
            return None;
        }
        if let Some(layout) = &self.seg.layouts[column] {
            return layout.string(&self.seg.map, layout.code(&self.seg.map, self.doc as usize));
        }
        let w = if self.seg.wide { 8 } else { 4 };
        let o = at + self.doc as usize * w;
        let (a, b) = if self.seg.wide { (u64_at(&self.seg.map, o) as usize, u64_at(&self.seg.map, o + 8) as usize) } else { (u32_at(&self.seg.map, o) as usize, u32_at(&self.seg.map, o + 4) as usize) };
        if a == b {
            return None;
        }
        let arena = at + (self.seg.docs as usize + 1) * w;
        std::str::from_utf8(&self.seg.map[arena + a..arena + b]).ok()
    }

    fn some_tag(&self, column: usize, is: &mut dyn FnMut(&str) -> bool) -> bool {
        let Some(layout) = self.seg.layouts.get(column).and_then(|l| l.as_ref()).filter(|l| l.kind == Kind::Tags) else { return false };
        layout.tags(&self.seg.map, self.doc as usize).any(|code| layout.string(&self.seg.map, code).is_some_and(&mut *is))
    }
}

/// Walks the dictionary in key order; `key` is the current entry's key.
pub struct DictIter<'a> {
    seg: &'a Segment,
    block: usize,
    i: usize,
    at: usize,
    pub key: Vec<u8>,
    postings: u64,
}

impl DictIter<'_> {
    pub fn next_entry(&mut self) -> Option<TermInfo> {
        let seg = self.seg;
        if self.block >= seg.dict_blocks {
            return None;
        }
        let map = &seg.map[..];
        if self.i == 0 {
            self.at = seg.block_at(self.block);
            let len = varint(map, &mut self.at) as usize;
            self.key.clear();
            self.key.extend_from_slice(&map[self.at..self.at + len]);
            self.at += len;
            self.postings = varint(map, &mut self.at);
        } else {
            let shared = varint(map, &mut self.at) as usize;
            let len = varint(map, &mut self.at) as usize;
            self.key.truncate(shared);
            self.key.extend_from_slice(&map[self.at..self.at + len]);
            self.at += len;
            self.postings += varint(map, &mut self.at);
        }
        let df = varint(map, &mut self.at) as u32;
        let last = varint(map, &mut self.at) as u32;
        let n = map[self.at];
        self.at += 1;
        let impacts = Impacts::read(map, &mut self.at, n);
        self.i += 1;
        if self.i == DICT_BLOCK.min(seg.terms - self.block * DICT_BLOCK) {
            self.block += 1;
            self.i = 0;
        }
        Some(TermInfo { field: self.key[0], df, postings: HEADER as u64 + self.postings, last, impacts })
    }
}

#[cfg(not(unix))]
unsafe fn madvise_random(_ptr: *const u8, _len: usize) {}

#[cfg(unix)]
unsafe fn madvise_random(ptr: *const u8, len: usize) {
    unsafe extern "C" {
        fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
    }
    const MADV_RANDOM: i32 = 1;
    unsafe { madvise(ptr as *mut _, len, MADV_RANDOM) };
}

#[inline]
pub(crate) fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

#[inline]
fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// A variable-byte number: 7 bits per byte, low bits first, high bit set on all but the last.
#[inline]
pub(crate) fn varint(b: &[u8], at: &mut usize) -> u64 {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let byte = b[*at];
        *at += 1;
        v |= ((byte & 0x7f) as u64) << shift;
        if byte < 0x80 {
            return v;
        }
        shift += 7;
    }
}

pub(crate) fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

// MARK: Passages in memory

#[derive(Default, Clone)]
pub struct Postings {
    pub docs: Vec<u32>,
    pub tfs: Vec<u32>,
}

/// Passages indexed in memory: the newest passages of a live index (searched as they are, then
/// written out as a segment), and the chunks a build indexes in parallel.
pub struct MemIndex {
    pub fields: usize,
    pub postings: HashMap<Box<[u8]>, Postings, crate::hash::FxBuild>,
    /// Per field, each passage's length in words.
    pub lengths: Vec<Vec<u32>>,
    pub totals: [u64; MAX_FIELDS],
    pub keys: Vec<u64>,
    pub versions: Vec<u64>,
    pub passages: Vec<u32>,
    /// Attribute columns, one value (or none) per passage.
    pub attrs: Vec<(String, Column)>,
    /// Vectors as their metric stores them (normalized for cosine), per passage (the first
    /// passage of a document carries its vector).
    pub vectors: Vec<Option<std::sync::Arc<[u16]>>>,
    /// Stored text, per passage (one per field), when the index stores text.
    pub texts: Vec<Option<std::sync::Arc<[String]>>>,
    /// How each field's text becomes terms.
    pub analyzers: Vec<Analyzer>,
    /// A term's key (field byte + term), reused between words.
    key: Vec<u8>,
}

impl MemIndex {
    pub fn new(fields: usize) -> MemIndex {
        assert!(fields <= MAX_FIELDS);
        MemIndex { fields, postings: HashMap::default(), lengths: vec![Vec::new(); fields], totals: [0; MAX_FIELDS], keys: Vec::new(), versions: Vec::new(), passages: Vec::new(), attrs: Vec::new(), vectors: Vec::new(), texts: Vec::new(), analyzers: vec![Analyzer::PLAIN; fields], key: Vec::new() }
    }

    /// `self` followed by `other`, `other`'s passages numbered after these: two memtable chunks as
    /// one (for binomial merging of a memtable's chunks).
    pub fn concat(&self, other: &MemIndex) -> MemIndex {
        let base = self.len() as u32;
        let mut m = MemIndex::with_analyzers(self.fields, &self.analyzers);
        m.postings = self.postings.clone();
        for (k, p) in &other.postings {
            let e = m.postings.entry(k.clone()).or_default();
            e.docs.extend(p.docs.iter().map(|d| d + base));
            e.tfs.extend_from_slice(&p.tfs);
        }
        for f in 0..self.fields {
            m.lengths[f] = self.lengths[f].clone();
            m.lengths[f].extend_from_slice(&other.lengths[f]);
            m.totals[f] = self.totals[f] + other.totals[f];
        }
        m.keys = [&self.keys[..], &other.keys[..]].concat();
        m.versions = [&self.versions[..], &other.versions[..]].concat();
        m.passages = [&self.passages[..], &other.passages[..]].concat();
        m.vectors = self.vectors.iter().chain(&other.vectors).cloned().collect();
        m.texts = self.texts.iter().chain(&other.texts).cloned().collect();
        let names: Vec<(String, Kind)> = self.attrs.iter().chain(&other.attrs).map(|(n, c)| (n.clone(), c.kind())).fold(Vec::new(), |mut v, x| {
            if !v.iter().any(|(n, _)| *n == x.0) {
                v.push(x);
            }
            v
        });
        // Columns appended whole (value by value, every string was copied again at each merge).
        for (name, kind) in names {
            let mut c = Column::new(kind);
            for (src, n) in [(self, self.len()), (other, other.len())] {
                let start = c.len();
                if let Some((_, col)) = src.attrs.iter().find(|(cn, _)| *cn == name) {
                    c.extend_from(col);
                }
                c.pad(start + n);
            }
            m.attrs.push((name, c));
        }
        m
    }

    /// An empty one whose fields are analyzed so.
    pub fn with_analyzers(fields: usize, analyzers: &[Analyzer]) -> MemIndex {
        let mut m = MemIndex::new(fields);
        for (f, a) in analyzers.iter().enumerate().take(fields) {
            m.analyzers[f] = *a;
        }
        m
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Indexes one passage (`texts`: one per field; missing fields are empty); its number here.
    pub fn add(&mut self, key: u64, version: u64, passage: u32, texts: &[&str]) -> u32 {
        self.add_with(key, version, passage, texts, &[])
    }

    /// Like `add`, with attributes. An attribute first seen here gets a column (none for the
    /// passages before); one of another kind than its column's is left out.
    pub fn add_with(&mut self, key: u64, version: u64, passage: u32, texts: &[&str], attrs: &[(String, Value)]) -> u32 {
        self.add_full(key, version, passage, texts, attrs, None)
    }

    /// Like `add_with`, with a vector (as its metric stores it: normalized for cosine).
    pub fn add_full(&mut self, key: u64, version: u64, passage: u32, texts: &[&str], attrs: &[(String, Value)], vector: Option<Vec<f32>>) -> u32 {
        let doc = self.keys.len() as u32;
        // In half precision, as segments keep them: half the memory, and half to read when
        // unflushed vectors are scanned.
        self.vectors.push(vector.map(|v| vector::to_f16(&v).into()));
        self.texts.push(None);
        for (name, value) in attrs {
            if !self.attrs.iter().any(|(n, _)| n == name) {
                let mut c = Column::new(value.kind());
                c.pad(doc as usize);
                self.attrs.push((name.clone(), c));
            }
        }
        for (name, column) in &mut self.attrs {
            column.push(attrs.iter().find(|(n, _)| n == name).map(|(_, v)| v));
        }
        self.keys.push(key);
        self.versions.push(version);
        self.passages.push(passage);
        for f in 0..self.fields {
            let analyzer = self.analyzers[f];
            if analyzer.sparse {
                // Weighted terms: a term's weight (in hundredths) is its posting's frequency, a
                // term given twice the sum of its weights (as a word's repeats add up).
                let (postings, key) = (&mut self.postings, &mut self.key);
                let mut terms = 0u32;
                for (t, q) in crate::analysis::decode_sparse(texts.get(f).copied().unwrap_or("")) {
                    key.clear();
                    key.push(f as u8);
                    key.extend_from_slice(t.as_bytes());
                    let p = match postings.get_mut(&key[..]) {
                        Some(p) => p,
                        None => postings.entry(key[..].into()).or_default(),
                    };
                    if p.docs.last() == Some(&doc) {
                        *p.tfs.last_mut().unwrap() += q;
                    } else {
                        p.docs.push(doc);
                        p.tfs.push(q);
                        terms += 1;
                    }
                }
                self.lengths[f].push(terms);
                self.totals[f] += terms as u64;
                continue;
            }
            // A term's posting list ends at this document once the document has had it: counting
            // its frequency there costs a lookup per word (sorting each document's words to count
            // them, with a string comparison per step, had been a fifth of indexing).
            let (postings, key) = (&mut self.postings, &mut self.key);
            let mut words = 0u32;
            analyzer.for_each(texts.get(f).copied().unwrap_or(""), |t| {
                words += 1;
                key.clear();
                key.push(f as u8);
                key.extend_from_slice(t.as_bytes());
                match postings.get_mut(&key[..]) {
                    Some(p) if p.docs.last() == Some(&doc) => *p.tfs.last_mut().unwrap() += 1,
                    Some(p) => {
                        p.docs.push(doc);
                        p.tfs.push(1);
                    }
                    None => {
                        let p = postings.entry(key[..].into()).or_default();
                        p.docs.push(doc);
                        p.tfs.push(1);
                    }
                }
            });
            self.lengths[f].push(words);
            self.totals[f] += words as u64;
        }
        doc
    }

    pub fn get(&self, field: u8, term: &str) -> Option<&Postings> {
        let mut key = Vec::with_capacity(term.len() + 1);
        key.push(field);
        key.extend_from_slice(term.as_bytes());
        self.postings.get(key.as_slice())
    }

    /// Terms of `field` starting with `prefix`, with their document counts.
    pub fn with_prefix(&self, field: u8, prefix: &str) -> Vec<(String, u32)> {
        self.postings
            .iter()
            .filter(|(k, _)| k[0] == field && k[1..].starts_with(prefix.as_bytes()))
            .map(|(k, p)| (String::from_utf8_lossy(&k[1..]).into_owned(), p.docs.len() as u32))
            .collect()
    }

    pub fn norm_id(&self, field: usize, doc: u32) -> u8 {
        norms::id(self.lengths[field][doc as usize])
    }

    pub fn attr_columns(&self) -> Vec<(String, Kind)> {
        self.attrs.iter().map(|(n, c)| (n.clone(), c.kind())).collect()
    }

    pub fn row(&self, doc: u32) -> MemRow<'_> {
        MemRow { mem: self, doc }
    }

    pub fn attributes(&self, doc: u32) -> Vec<(String, Value)> {
        self.attrs.iter().filter_map(|(n, c)| c.get(doc as usize).map(|v| (n.clone(), v))).collect()
    }
}

pub struct MemRow<'a> {
    mem: &'a MemIndex,
    doc: u32,
}

impl Row for MemRow<'_> {
    fn int(&self, column: usize) -> Option<i64> {
        match &self.mem.attrs.get(column)?.1 {
            Column::Int(v) | Column::Float(v) => v.get(self.doc as usize).copied().filter(|&x| x != NO_INT),
            _ => None,
        }
    }

    fn str(&self, column: usize) -> Option<&str> {
        match &self.mem.attrs.get(column)?.1 {
            Column::Str(v) => v.get(self.doc as usize).and_then(|s| s.as_deref()),
            _ => None,
        }
    }

    fn some_tag(&self, column: usize, is: &mut dyn FnMut(&str) -> bool) -> bool {
        match self.mem.attrs.get(column) {
            Some((_, Column::Tags(t))) => t.of(self.doc as usize).any(is),
            _ => false,
        }
    }
}

// MARK: Writing

/// Per passage, for the file's columns.
struct Columns<'a> {
    keys: Vec<u64>,
    versions: Vec<u64>,
    passages: Vec<u32>,
    /// Per field.
    lengths: Vec<Vec<u16>>,
    attrs: Vec<(String, Column)>,
    /// Per passage: its vector, if any, where it's kept (a memtable's, a segment's).
    vectors: Vec<Option<vector::VectorRef<'a>>>,
    /// Per passage: where its stored text is, if it has some.
    stored: Vec<Option<TextRef>>,
}

impl Columns<'_> {
    fn new(fields: usize) -> Self {
        Self { keys: Vec::new(), versions: Vec::new(), passages: Vec::new(), lengths: vec![Vec::new(); fields], attrs: Vec::new(), vectors: Vec::new(), stored: Vec::new() }
    }

    fn norm_ids(&self) -> Vec<Vec<u8>> {
        self.lengths.iter().map(|l| l.iter().map(|&n| norms::id(n as u32)).collect()).collect()
    }
}

/// One encoded posting list.
struct Encoded {
    bytes: Vec<u8>,
    df: u32,
    last: u32,
    impacts: Impacts,
    /// Has bit-packed blocks, read as 32-bit words: must start 4-byte aligned.
    aligned: bool,
}

/// Writes a segment term by term (lists in key order), then the dictionary and the columns.
/// A segment file as it's written: with a CRC-32 of every `SUM_CHUNK` bytes after the header.
struct Summed {
    out: BufWriter<File>,
    crc: crc32fast::Hasher,
    filled: usize,
    sums: Vec<u32>,
}

impl Summed {
    fn write_all(&mut self, mut bytes: &[u8]) -> std::io::Result<()> {
        self.out.write_all(bytes)?;
        while !bytes.is_empty() {
            let take = (SUM_CHUNK - self.filled).min(bytes.len());
            self.crc.update(&bytes[..take]);
            self.filled += take;
            bytes = &bytes[take..];
            if self.filled == SUM_CHUNK {
                self.sums.push(std::mem::take(&mut self.crc).finalize());
                self.filled = 0;
            }
        }
        Ok(())
    }

    /// The file, with the checksums of what was written after them (`at`: where they start).
    fn finish(mut self, at: u64) -> std::io::Result<BufWriter<File>> {
        if self.filled > 0 {
            self.sums.push(self.crc.finalize());
        }
        let mut tail = Vec::with_capacity(16 + self.sums.len() * 4);
        tail.extend_from_slice(&SUMS_MAGIC.to_le_bytes());
        tail.extend_from_slice(&(SUM_CHUNK as u32).to_le_bytes());
        tail.extend_from_slice(&(self.sums.len() as u32).to_le_bytes());
        for sum in &self.sums {
            tail.extend_from_slice(&sum.to_le_bytes());
        }
        tail.extend_from_slice(&crc32fast::hash(&tail).to_le_bytes());
        debug_assert_eq!(self.sums.len(), (at as usize - HEADER).div_ceil(SUM_CHUNK));
        self.out.write_all(&tail)?;
        Ok(self.out)
    }
}

struct Writer {
    out: Summed,
    pos: u64,
    dict: Vec<u8>,
    block_offsets: Vec<u64>,
    prev_key: Vec<u8>,
    prev_off: u64,
    terms: usize,
}

impl Writer {
    fn create(tmp: &Path) -> std::io::Result<Writer> {
        let mut out = BufWriter::with_capacity(8 << 20, File::create(tmp)?);
        out.write_all(&[0u8; HEADER])?;
        let out = Summed { out, crc: crc32fast::Hasher::new(), filled: 0, sums: Vec::new() };
        Ok(Writer { out, pos: HEADER as u64, dict: Vec::new(), block_offsets: Vec::new(), prev_key: Vec::new(), prev_off: 0, terms: 0 })
    }

    fn add(&mut self, key: &[u8], list: &Encoded) -> std::io::Result<()> {
        if list.aligned && self.pos % 4 != 0 {
            let pad = 4 - self.pos % 4;
            self.out.write_all(&[0u8; 4][..pad as usize])?;
            self.pos += pad;
        }
        // Offsets in the dictionary are relative to the end of the header.
        let off = self.pos - HEADER as u64;
        if self.terms % DICT_BLOCK == 0 {
            self.block_offsets.push(self.dict.len() as u64);
            put_varint(&mut self.dict, key.len() as u64);
            self.dict.extend_from_slice(key);
            put_varint(&mut self.dict, off);
        } else {
            let shared = key.iter().zip(&self.prev_key).take_while(|(a, b)| a == b).count();
            put_varint(&mut self.dict, shared as u64);
            put_varint(&mut self.dict, (key.len() - shared) as u64);
            self.dict.extend_from_slice(&key[shared..]);
            put_varint(&mut self.dict, off - self.prev_off);
        }
        put_varint(&mut self.dict, list.df as u64);
        put_varint(&mut self.dict, list.last as u64);
        self.dict.push(list.impacts.n);
        list.impacts.write(&mut self.dict);
        self.prev_key.clear();
        self.prev_key.extend_from_slice(key);
        self.prev_off = off;
        self.terms += 1;
        self.out.write_all(&list.bytes)?;
        self.pos += list.bytes.len() as u64;
        Ok(())
    }

    fn finish(mut self, tmp: &Path, path: &Path, fields: usize, columns: &Columns<'_>, norm_ids: &[Vec<u8>], built: Option<vector::Built>) -> std::io::Result<()> {
        let docs = columns.keys.len();
        let put = |out: &mut Summed, pos: &mut u64, bytes: &[u8]| -> std::io::Result<u64> {
            let at = *pos;
            out.write_all(bytes)?;
            *pos += bytes.len() as u64;
            Ok(at)
        };
        let dict_index_at = self.pos;
        let index: Vec<u8> = self.block_offsets.iter().flat_map(|b| (dict_index_at + self.block_offsets.len() as u64 * 8 + b).to_le_bytes()).collect();
        let (out, pos) = (&mut self.out, &mut self.pos);
        put(out, pos, &index)?;
        put(out, pos, &self.dict)?;
        let keys_at = put(out, pos, &columns.keys.iter().flat_map(|k| k.to_le_bytes()).collect::<Vec<u8>>())?;
        let versions_at = put(out, pos, &columns.versions.iter().flat_map(|k| k.to_le_bytes()).collect::<Vec<u8>>())?;
        let passages_at = put(out, pos, &columns.passages.iter().flat_map(|k| k.to_le_bytes()).collect::<Vec<u8>>())?;
        let mut order: Vec<u32> = (0..docs as u32).collect();
        order.sort_by_key(|&d| (columns.keys[d as usize], d));
        let key_order_at = put(out, pos, &order.iter().flat_map(|k| k.to_le_bytes()).collect::<Vec<u8>>())?;
        let lengths_at = *pos;
        for l in &columns.lengths {
            put(out, pos, &l.iter().flat_map(|k| k.to_le_bytes()).collect::<Vec<u8>>())?;
        }
        let norms_at = *pos;
        for n in norm_ids {
            put(out, pos, n)?;
        }
        let mut section = (columns.attrs.len() as u32).to_le_bytes().to_vec();
        for (name, column) in &columns.attrs {
            let mut data = Vec::new();
            column.encode(&mut data);
            section.extend_from_slice(&(name.len() as u16).to_le_bytes());
            section.extend_from_slice(name.as_bytes());
            section.push(column.kind() as u8);
            section.extend_from_slice(&(data.len() as u64).to_le_bytes());
            section.extend_from_slice(&data);
        }
        let attrs_at = put(out, pos, &section)?;
        let with_vectors: Vec<(u32, vector::VectorRef)> = columns.vectors.iter().enumerate().filter_map(|(d, v)| v.map(|v| (d as u32, v))).collect();
        let mut vectors_at = 0u64;
        let mut version = COSINE_VERSION;
        if let (Some(&(_, first)), Some(built)) = (with_vectors.first(), built) {
            let dims = first.len();
            let padded = dims.next_power_of_two().max(64);
            let align = |out: &mut Summed, pos: &mut u64| -> std::io::Result<()> {
                let pad = pos.next_multiple_of(16) - *pos;
                out.write_all(&vec![0u8; pad as usize])?;
                *pos += pad;
                Ok(())
            };
            align(out, pos)?;
            let mut head = Vec::with_capacity(32);
            for v in [dims as u32, padded as u32, built.radius.len() as u32, with_vectors.len() as u32] {
                head.extend_from_slice(&v.to_le_bytes());
            }
            head.extend_from_slice(&built.seed.to_le_bytes());
            head.extend_from_slice(&built.nprobe.to_le_bytes());
            // Full-precision rows in half precision.
            head.extend_from_slice(&1u32.to_le_bytes());
            head.extend_from_slice(&built.generation.to_le_bytes());
            head.extend_from_slice(&built.trained_on.to_le_bytes());
            head.extend_from_slice(&built.budgets.measured_on.to_le_bytes());
            for b in built.budgets.by_depth {
                head.extend_from_slice(&b.to_le_bytes());
            }
            debug_assert_eq!(head.len(), VECTOR_HEADER);
            // Another metric than cosine: named here, in a segment of the version that has it.
            if built.metric != vector::Metric::Cosine {
                version = VERSION;
                head.extend_from_slice(&built.metric.code().to_le_bytes());
                head.resize(VECTOR_HEADER + METRIC_HEADER, 0);
            }
            vectors_at = put(out, pos, &head)?;
            // Rows go in the codes' order: a document's row is its code's place.
            let mut rows = vec![u32::MAX; docs];
            for (row, &d) in built.docs.iter().enumerate() {
                rows[d as usize] = row as u32;
            }
            // The codes and factors as a search reads them (`vector::View`).
            let words = padded / 64;
            let blocked = vector::Blocked::new(&built.codes, &built.offsets, words);
            let pre = &built.pre;
            let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let mut arrays: Vec<Vec<u8>> = vec![
                f32s(&built.centroids),
                f32s(&built.radius),
                u32s(&built.offsets),
                u32s(&built.docs),
                u32s(&blocked.starts),
                blocked.bytes,
                pre.iter().flat_map(|p| [p.inv_ip, p.err, p.norm, p.ones]).flat_map(f32::to_le_bytes).collect(),
            ];
            // Dot product: each vector's share of its centroid, beside its factors, and the
            // most of that in each list.
            if built.metric == vector::Metric::Dot {
                debug_assert!(built.along.len() == built.docs.len() && built.longest.len() == built.radius.len());
                arrays.push(f32s(&built.along));
                arrays.push(f32s(&built.longest));
            }
            arrays.push(u32s(&rows));
            for a in &arrays {
                put(out, pos, a)?;
                align(out, pos)?;
            }
            // The rows in half precision, streamed (at a million vectors, half a gigabyte).
            for &d in &built.docs {
                let v = columns.vectors[d as usize].expect("a coded document has a vector");
                put(out, pos, &v.to_f16().iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            }
            align(out, pos)?;
        }

        // Stored text: the blob files, then a reference per passage (index into them; u32::MAX for
        // none), 16 bytes each.
        let mut stored_at = 0u64;
        if columns.stored.iter().any(Option::is_some) {
            let mut ids: Vec<u64> = columns.stored.iter().flatten().map(|r| r.blob).collect();
            ids.sort_unstable();
            ids.dedup();
            let pad = pos.next_multiple_of(8) - *pos;
            put(out, pos, &vec![0u8; pad as usize])?;
            let mut section = (ids.len() as u32).to_le_bytes().to_vec();
            section.extend_from_slice(&0u32.to_le_bytes());
            for id in &ids {
                section.extend_from_slice(&id.to_le_bytes());
            }
            for r in &columns.stored {
                match r {
                    Some(r) => {
                        section.extend_from_slice(&(ids.binary_search(&r.blob).unwrap() as u32).to_le_bytes());
                        section.extend_from_slice(&r.inner.to_le_bytes());
                        section.extend_from_slice(&r.block.to_le_bytes());
                    }
                    None => section.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                }
            }
            stored_at = put(out, pos, &section)?;
        }

        let mut header = vec![0u8; HEADER];
        header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        header[4..8].copy_from_slice(&version.to_le_bytes());
        header[8..12].copy_from_slice(&(docs as u32).to_le_bytes());
        header[12..16].copy_from_slice(&(fields as u32).to_le_bytes());
        header[16..20].copy_from_slice(&(self.terms as u32).to_le_bytes());
        header[20..24].copy_from_slice(&(self.block_offsets.len() as u32).to_le_bytes());
        for (f, l) in columns.lengths.iter().enumerate() {
            let total: u64 = l.iter().map(|&n| n as u64).sum();
            header[24 + f * 8..32 + f * 8].copy_from_slice(&total.to_le_bytes());
        }
        for (i, v) in [dict_index_at, dict_index_at + index.len() as u64, norms_at, lengths_at, keys_at, versions_at, passages_at, key_order_at].iter().enumerate() {
            header[96 + i * 8..104 + i * 8].copy_from_slice(&v.to_le_bytes());
        }
        header[160..168].copy_from_slice(&attrs_at.to_le_bytes());
        header[168..176].copy_from_slice(&vectors_at.to_le_bytes());
        header[176..184].copy_from_slice(&stored_at.to_le_bytes());
        // The checksums of everything after the header, then the header's own.
        let pad = pos.next_multiple_of(8) - *pos;
        let sums_at = put(out, pos, &vec![0u8; pad as usize])? + pad;
        header[184..192].copy_from_slice(&sums_at.to_le_bytes());
        let sum = crc32fast::hash(&header[..HEADER - 4]);
        header[HEADER - 4..].copy_from_slice(&sum.to_le_bytes());
        let mut file = self.out.finish(sums_at)?.into_inner().map_err(|e| e.into_error())?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;
        // On disk before the manifest names it.
        file.sync_all()?;
        drop(file);
        std::fs::rename(tmp, path)
    }
}

/// Encodes one posting list (documents ascending, frequencies ≥ 1).
fn encode(docs: &[u32], tfs: &[u32], norm_ids: &[u8]) -> Encoded {
    let headers = docs.len() >= BLOCK;
    let blocks = docs.len().div_ceil(BLOCK);
    let mut fronts: Vec<Impacts> = Vec::with_capacity(blocks);
    let mut pairs: Vec<(u32, u8)> = Vec::with_capacity(BLOCK);
    for b in 0..blocks {
        pairs.clear();
        pairs.extend((b * BLOCK..((b + 1) * BLOCK).min(docs.len())).map(|d| (tfs[d], norm_ids[docs[d] as usize])));
        fronts.push(Impacts::front(&mut pairs));
    }
    let impacts = fronts.iter().skip(1).fold(fronts[0], |a, b| a.union(b));
    let mut out = Vec::new();
    // Headers, then the blocks' impacts (padded to 4), then the data.
    out.resize(if headers { blocks * BLOCK_HEADER } else { 0 }, 0);
    let mut impact_at = Vec::with_capacity(blocks);
    if headers {
        for f in &fronts {
            impact_at.push(out.len() as u32);
            f.write(&mut out);
        }
        out.resize(out.len().next_multiple_of(4), 0);
    }
    let mut prev: i64 = -1;
    let mut gaps = [0u32; BLOCK];
    let mut freqs = [0u32; BLOCK];
    for b in 0..blocks {
        let range = b * BLOCK..((b + 1) * BLOCK).min(docs.len());
        let n = range.len();
        for (i, d) in range.clone().enumerate() {
            gaps[i] = (docs[d] as i64 - prev - 1) as u32;
            prev = docs[d] as i64;
            freqs[i] = tfs[d] - 1;
        }
        let rel = out.len() as u32;
        let (mut gw, mut fw) = (0, 0);
        if n == BLOCK {
            (gw, fw) = (bitpack::width(&gaps), bitpack::width(&freqs));
            bitpack::pack(&gaps, gw, &mut out);
            bitpack::pack(&freqs, fw, &mut out);
        } else {
            for &g in &gaps[..n] {
                put_varint(&mut out, g as u64);
            }
            for &f in &freqs[..n] {
                put_varint(&mut out, f as u64);
            }
        }
        if headers {
            let h = b * BLOCK_HEADER;
            out[h..h + 4].copy_from_slice(&docs[range.end - 1].to_le_bytes());
            out[h + 4..h + 8].copy_from_slice(&rel.to_le_bytes());
            out[h + 8] = gw;
            out[h + 9] = fw;
            out[h + 10] = (n - 1) as u8;
            out[h + 11] = fronts[b].n;
            out[h + 12..h + 16].copy_from_slice(&impact_at[b].to_le_bytes());
        }
    }
    Encoded { bytes: out, df: docs.len() as u32, last: *docs.last().unwrap(), impacts, aligned: headers }
}

fn tmp_for(path: &Path) -> std::path::PathBuf {
    path.with_extension("tmp")
}

/// Writes passages indexed in memory as a segment: `chunks` in order (their passages numbered
/// one after the other). Terms are encoded in parallel. Passages' stored text goes into a new
/// blob file of id `blob_id` (next to the segment), written first.
pub fn write_chunks(path: &Path, chunks: &[&MemIndex], blob_id: u64, generation: Option<&Segment>, total: usize) -> std::io::Result<()> {
    write_chunks_with(path, chunks, blob_id, generation, total, vector::Metric::Cosine)
}

/// `write_chunks`, for vectors compared by `metric` (and stored as it stores them).
pub fn write_chunks_with(path: &Path, chunks: &[&MemIndex], blob_id: u64, generation: Option<&Segment>, total: usize, metric: vector::Metric) -> std::io::Result<()> {
    crate::faults::check("segment write")?;
    let fields = chunks.first().map_or(0, |c| c.fields);
    let mut offsets = Vec::with_capacity(chunks.len());
    let mut columns = Columns::new(fields);
    for c in chunks {
        for (name, column) in &c.attrs {
            if !columns.attrs.iter().any(|(n, _)| n == name) {
                columns.attrs.push((name.clone(), Column::new(column.kind())));
            }
        }
    }
    for c in chunks {
        for (name, column) in &mut columns.attrs {
            let mine = c.attrs.iter().find(|(n, _)| n == name).map(|(_, col)| col);
            for d in 0..c.len() {
                column.push(mine.and_then(|col| col.get(d)).as_ref());
            }
        }
        offsets.push(columns.keys.len() as u32);
        columns.vectors.extend(c.vectors.iter().map(|v| v.as_deref().map(vector::VectorRef::F16)));
        columns.stored.resize(columns.stored.len() + c.len(), None);
        columns.keys.extend_from_slice(&c.keys);
        columns.versions.extend_from_slice(&c.versions);
        columns.passages.extend_from_slice(&c.passages);
        for f in 0..fields {
            columns.lengths[f].extend(c.lengths[f].iter().map(|&n| n.min(u16::MAX as u32) as u16));
        }
    }
    let norm_ids = columns.norm_ids();
    // Stored text, posting lists and the vector index at once: none needs another.
    let text = || -> std::io::Result<Vec<(usize, TextRef)>> {
        let stored: Vec<(usize, &[String])> = chunks.iter().flat_map(|c| &c.texts).enumerate().filter_map(|(d, t)| t.as_deref().map(|t| (d, t))).collect();
        if stored.is_empty() {
            return Ok(Vec::new());
        }
        let refs = blob::write(path.parent().unwrap_or(Path::new(".")), blob_id, &stored.iter().map(|(_, t)| *t).collect::<Vec<_>>())?;
        Ok(stored.iter().map(|(d, _)| *d).zip(refs).collect())
    };
    let postings = || {
        let mut keys: Vec<&[u8]> = chunks.iter().flat_map(|c| c.postings.keys().map(|k| &k[..])).collect();
        keys.par_sort_unstable();
        keys.dedup();
        let lists: Vec<Encoded> = keys
            .par_iter()
            .map(|key| {
                let (mut docs, mut tfs) = (Vec::new(), Vec::new());
                for (c, &base) in chunks.iter().zip(&offsets) {
                    if let Some(p) = c.postings.get(*key) {
                        docs.extend(p.docs.iter().map(|d| d + base));
                        tfs.extend_from_slice(&p.tfs);
                    }
                }
                encode(&docs, &tfs, &norm_ids[key[0] as usize])
            })
            .collect();
        (keys, lists)
    };
    let generation = generation.filter(|g| g.vector_metric() == Some(metric)).and_then(Segment::vector_generation);
    let ((refs, (keys, lists)), built) = rayon::join(|| rayon::join(text, postings), || build_vectors(&columns.vectors, blob_id, generation, None, total, metric));
    for (d, r) in refs? {
        columns.stored[d] = Some(r);
    }
    let tmp = tmp_for(path);
    let mut writer = Writer::create(&tmp)?;
    for (key, list) in keys.iter().zip(&lists) {
        writer.add(key, list)?;
    }
    writer.finish(&tmp, path, fields, &columns, &norm_ids, built)
}

/// The vector index of a segment's documents (`vectors` per document), if any have vectors.
/// The vector index of a segment being written as `id`: into `generation` (the namespace's, if
/// it has one), copying what `known` holds, or its own (see `vector::build`).
fn build_vectors(vectors: &[Option<vector::VectorRef<'_>>], id: u64, generation: Option<vector::Generation<'_>>, known: Option<vector::Known<'_>>, total: usize, metric: vector::Metric) -> Option<vector::Built> {
    let with: Vec<(u32, vector::VectorRef)> = vectors.iter().enumerate().filter_map(|(d, v)| v.map(|v| (d as u32, v))).collect();
    let dims = with.first()?.1.len();
    // A generation of other dimensions can't be built into (a namespace has one dimension; this
    // is for a file that isn't what it should be).
    let generation = generation.filter(|g| g.centroids.len() % dims.next_power_of_two().max(64) == 0);
    Some(vector::build(&with, dims, 0x5EED_F0F0 ^ dims as u64, id, generation, known, total, metric))
}

/// Builds a segment of `count` documents from text (`text(doc, field)`), on every core: documents
/// are indexed in chunks, then written as one segment. Keys are the document numbers.
pub fn build<'t>(path: &Path, fields: usize, count: usize, text: &(dyn Fn(usize, usize) -> &'t str + Sync)) -> std::io::Result<()> {
    assert!(fields <= MAX_FIELDS && count < u32::MAX as usize);
    let chunks = (rayon::current_num_threads() * 4).max(1);
    let per = count.div_ceil(chunks).max(1);
    let built: Vec<MemIndex> = (0..chunks)
        .into_par_iter()
        .map(|c| {
            let mut m = MemIndex::new(fields);
            let mut texts: Vec<&str> = Vec::with_capacity(fields);
            for doc in c * per..((c + 1) * per).min(count) {
                texts.clear();
                texts.extend((0..fields).map(|f| text(doc, f)));
                m.add(doc as u64, 0, 0, &texts);
            }
            m
        })
        .collect();
    write_chunks(path, &built.iter().collect::<Vec<_>>(), 0, None, 0)
}

/// Merges segments into one, leaving out their deleted passages: posting lists are decoded,
/// renumbered and re-encoded term by term (never all in memory), lengths recomputed from the
/// exact ones kept per passage. Stored text stays where it is (the references are copied) unless
/// its blob file is in `rewrite`, or it's text of a segment from before blob files: that goes
/// into a new blob file of id `blob_id`. Returns the passages kept.
pub fn merge(path: &Path, inputs: &[(&Segment, &Bitmap)], rewrite: &std::collections::HashSet<u64>, blob_id: u64, generation: Option<&Segment>, total: usize) -> std::io::Result<usize> {
    crate::faults::check("merge write")?;
    let fields = inputs.first().map_or(0, |(s, _)| s.fields);
    let mut columns = Columns::new(fields);
    for (seg, _) in inputs {
        for (name, kind) in seg.attr_columns() {
            if !name.starts_with(LEGACY_TEXT) && !columns.attrs.iter().any(|(n, _)| *n == name) {
                columns.attrs.push((name, Column::new(kind)));
            }
        }
    }
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut out_blob: Option<blob::Writer> = None;
    // The inputs' vectors are of one metric, which the result's are of too: vectors stored for
    // one (normalized, or not) mean something else under another.
    let mut metrics = inputs.iter().filter_map(|(s, _)| s.vector_metric());
    let metric = metrics.next().unwrap_or_default();
    if let Some(other) = metrics.find(|m| *m != metric) {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("segments whose vectors are compared by {} and by {} can't be merged", metric.name(), other.name())));
    }
    // What the inputs' lists already hold of the generation this merges into (if the result
    // doesn't outgrow it): lists, codes and factors by new document, copied instead of worked out.
    let generation = generation.filter(|g| g.vector_metric() == Some(metric)).and_then(Segment::vector_generation);
    let vectors: usize = inputs.iter().map(|(s, d)| s.vector_count().saturating_sub(d.count())).sum();
    let words = inputs.iter().find_map(|(s, _)| s.vector_dims()).map_or(0, |d| d.next_power_of_two().max(64) / 64);
    let reuse = generation.filter(|g| vector::builds_into(Some(g), words * 64, vectors, total));
    let (mut lists, mut codes, mut pre, mut along): (Vec<u32>, Vec<u64>, Vec<vector::Pre>, Vec<f32>) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    // Old document → new, per input (u32::MAX: deleted).
    let mut remap: Vec<Vec<u32>> = Vec::with_capacity(inputs.len());
    for (seg, deleted) in inputs {
        let same = reuse.is_some_and(|g| seg.vector_generation().is_some_and(|mine| mine.id == g.id && mine.centroids.len() == g.centroids.len()));
        let mut m = Vec::with_capacity(seg.docs as usize);
        for d in 0..seg.docs {
            if deleted.get(d) {
                m.push(u32::MAX);
                continue;
            }
            m.push(columns.keys.len() as u32);
            columns.keys.push(seg.key(d));
            columns.versions.push(seg.version(d));
            columns.passages.push(seg.passage(d));
            for f in 0..fields {
                columns.lengths[f].push(seg.length(d, f));
            }
            let values = seg.attributes(d);
            for (name, column) in &mut columns.attrs {
                column.push(values.iter().find(|(n, _)| n == name).map(|(_, v)| v));
            }
            columns.vectors.push(seg.vector_ref(d));
            if reuse.is_some() {
                let at = codes.len();
                codes.resize(at + words, 0);
                let coded = if same { seg.vector_coded(d, &mut codes[at..]) } else { None };
                lists.push(coded.map_or(u32::MAX, |c| c.0));
                pre.push(coded.map_or(vector::Pre::default(), |c| c.1));
                if metric == vector::Metric::Dot {
                    along.push(coded.map_or(0.0, |c| c.2));
                }
            }
            let moved = match seg.text_ref(d) {
                Some(r) if !rewrite.contains(&r.blob) => {
                    columns.stored.push(Some(r));
                    continue;
                }
                _ => seg.stored(d),
            };
            columns.stored.push(match moved {
                Some(texts) => {
                    if out_blob.is_none() {
                        out_blob = Some(blob::Writer::create(dir, blob_id)?);
                    }
                    Some(out_blob.as_mut().unwrap().add(&texts.iter().map(String::as_str).collect::<Vec<_>>())?)
                }
                None => None,
            });
        }
        remap.push(m);
    }
    if let Some(w) = out_blob {
        w.finish()?;
    }
    let norm_ids = columns.norm_ids();
    let tmp = tmp_for(path);
    let mut writer = Writer::create(&tmp)?;
    let mut iters: Vec<_> = inputs.iter().map(|(s, _)| s.entries()).collect();
    let mut current: Vec<Option<TermInfo>> = iters.iter_mut().map(|it| it.next_entry()).collect();
    let (mut docs, mut tfs) = (Vec::new(), Vec::new());
    let mut key = Vec::with_capacity(48);
    // The vector index meanwhile.
    let (postings, built) = rayon::join(|| -> std::io::Result<()> { loop {
        // The smallest key any input is at.
        let Some(first) = (0..inputs.len()).filter(|&i| current[i].is_some()).min_by(|&a, &b| iters[a].key.cmp(&iters[b].key)) else { break };
        key.clear();
        key.extend_from_slice(&iters[first].key);
        docs.clear();
        tfs.clear();
        for i in 0..inputs.len() {
            let Some(info) = current[i] else { continue };
            if iters[i].key != key {
                continue;
            }
            let (d, t) = inputs[i].0.read_list(info);
            for (doc, tf) in d.into_iter().zip(t) {
                let new = remap[i][doc as usize];
                if new != u32::MAX {
                    docs.push(new);
                    tfs.push(tf);
                }
            }
            current[i] = iters[i].next_entry();
        }
        if !docs.is_empty() {
            writer.add(&key, &encode(&docs, &tfs, &norm_ids[key[0] as usize]))?;
        }
    }
    Ok(()) }, || build_vectors(&columns.vectors, blob_id, generation, reuse.map(|_| vector::Known { lists: &lists, codes: &codes, pre: &pre, along: &along }), total, metric));
    postings?;
    let kept = columns.keys.len();
    writer.finish(&tmp, path, fields, &columns, &norm_ids, built)?;
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attrs::Filter;

    /// Columns of every width, with a step, the integers' extremes, sorted values (zones wholly
    /// inside or outside a condition), a column where every document has the same value, and
    /// strings by the thousand: what was written comes back, and every condition checked on a
    /// column's codes matches the documents it matches one document at a time.
    #[test]
    fn columns_of_every_shape_agree_with_rows() {
        let dir = std::env::temp_dir().join(format!("hinny-columns-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let docs = 5000u64;
        let mut m = MemIndex::new(1);
        let mut written: Vec<Vec<(String, Value)>> = Vec::new();
        for d in 0..docs {
            let r = next();
            let mut attrs = vec![("constant".to_string(), Value::Int(7)), ("one".to_string(), Value::Str("same".into()))];
            // A byte a code, with documents that have none.
            if d % 3 != 0 {
                attrs.push(("small".to_string(), Value::Int((r % 200) as i64 - 20)));
            }
            // Sorted, in steps of 60: two bytes a code, and zones a range takes whole or not at all.
            attrs.push(("sorted".to_string(), Value::Int(1_700_000_000 + d as i64 * 60)));
            attrs.push(("wide".to_string(), Value::Int((r % 3_000_000_000) as i64 - 1_500_000_000)));
            attrs.push(("huge".to_string(), Value::Int([NO_INT + 1, i64::MAX, 0, -1, r as i64][(r >> 40) as usize % 5].max(NO_INT + 1))));
            // An empty string is no value.
            attrs.push(("kind".to_string(), Value::Str(["pdf", "Mail", "jpg", "", "PDF"][(r >> 20) as usize % 5].into())));
            attrs.push(("path".to_string(), Value::Str(format!("/home/user{}/Doc{d}.txt", d % 40))));
            // Tags: one half the documents have (kept as a bit a document), ones a few have (as
            // their numbers), documents with several and with none; a column where every
            // document has the one tag, and one where none has any.
            let mut tags: Vec<String> = Vec::new();
            if d % 2 == 0 {
                tags.push("common".into());
            }
            if d % 5 == 0 {
                tags.push(format!("t{}", d % 700));
            }
            if (r >> 12) % 7 == 0 {
                tags.push("Mixed".into());
            }
            if (r >> 30) % 11 == 0 {
                tags.push("Über".into());
            }
            tags.sort();
            attrs.push(("tags".to_string(), Value::Tags(tags)));
            attrs.push(("all".to_string(), Value::Tags(vec!["every".into()])));
            attrs.push(("empty".to_string(), Value::Tags(Vec::new())));
            m.add_full(d, 1, 0, &["x"], &attrs, None);
            written.push(attrs.into_iter().filter(|(_, v)| *v != Value::Str(String::new()) && *v != Value::Tags(Vec::new())).collect());
        }
        let path = dir.join("seg");
        write_chunks(&path, &[&m], 0, None, 0).unwrap();
        let seg = Segment::open(&path).unwrap();
        for d in (0..docs).step_by(37).chain([docs - 1]) {
            let (mut got, mut want) = (seg.attributes(d as u32), written[d as usize].clone());
            got.sort_by(|a, b| a.0.cmp(&b.0));
            want.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(got, want, "document {d}");
        }
        let widths: Vec<(String, usize)> = seg.attrs.iter().zip(&seg.layouts).map(|((n, _, _), l)| (n.clone(), l.unwrap().width)).collect();
        let width = |name: &str| widths.iter().find(|(n, _)| n == name).unwrap().1;
        assert_eq!([width("constant"), width("one"), width("small"), width("sorted"), width("wide"), width("huge"), width("kind"), width("path")], [0, 0, 1, 2, 4, 8, 1, 2]);

        let n = |s: &str| s.to_string();
        let int = |x: i64| Value::Int(x);
        let text = |s: &str| Value::Str(s.into());
        let mut filters = vec![
            Filter::Eq(n("constant"), int(7)),
            Filter::Eq(n("constant"), int(8)),
            Filter::Range(n("constant"), Some(8), None),
            Filter::Exists(n("constant")),
            Filter::Eq(n("one"), text("same")),
            Filter::Eq(n("one"), text("other")),
            Filter::Prefix(n("one"), n("SA")),
            Filter::Exists(n("small")),
            Filter::Range(n("small"), Some(-20), Some(179)),
            Filter::Range(n("small"), Some(-21), Some(-21)),
            Filter::Range(n("small"), Some(180), None),
            Filter::In(n("small"), vec![int(-20), int(0), int(179), int(500), text("pdf")]),
            // Between two steps, exactly on one, and the ends.
            Filter::Eq(n("sorted"), int(1_700_000_000 + 60 * 2000)),
            Filter::Eq(n("sorted"), int(1_700_000_000 + 60 * 2000 + 1)),
            Filter::Range(n("sorted"), Some(1_700_000_000 + 60 * 1000 + 1), Some(1_700_000_000 + 60 * 3100 - 1)),
            Filter::Range(n("sorted"), Some(1_700_000_000 + 59), Some(1_700_000_000 + 59)),
            Filter::Range(n("sorted"), None, Some(1_700_000_000)),
            Filter::Range(n("sorted"), Some(1_700_000_000 + 60 * 4999), None),
            Filter::Range(n("sorted"), Some(i64::MIN + 5), Some(i64::MAX - 5)),
            Filter::Range(n("wide"), Some(-10_000_000), Some(250_000_000)),
            Filter::Range(n("wide"), Some(5), Some(4)),
            Filter::Eq(n("huge"), int(i64::MAX)),
            Filter::Eq(n("huge"), int(NO_INT + 1)),
            Filter::Range(n("huge"), Some(-1), Some(0)),
            Filter::Range(n("huge"), None, Some(-2)),
            Filter::Range(n("huge"), Some(1), None),
            Filter::Exists(n("kind")),
            Filter::Eq(n("kind"), text("pdf")),
            Filter::Eq(n("kind"), text("")),
            Filter::Eq(n("kind"), int(3)),
            Filter::In(n("kind"), vec![text("jpg"), text("PDF"), text("none"), int(1)]),
            Filter::In(n("kind"), vec![]),
            // Without case: two strings that aren't next to each other in the dictionary.
            Filter::Prefix(n("kind"), n("pd")),
            Filter::Contains(n("kind"), n("A")),
            Filter::Contains(n("kind"), n("zzz")),
            Filter::Prefix(n("kind"), n("")),
            Filter::Eq(n("path"), text("/home/user7/Doc47.txt")),
            Filter::Eq(n("path"), text("/home/user7/Doc48.txt")),
            Filter::Prefix(n("path"), n("/home/user3")),
            Filter::Prefix(n("path"), n("/HOME/User39/")),
            Filter::Contains(n("path"), n("doc49")),
            Filter::In(n("path"), vec![text("/home/user0/Doc0.txt"), text("/home/user39/Doc4999.txt"), text("/nowhere")]),
            Filter::Exists(n("nothing")),
            Filter::Eq(n("nothing"), text("x")),
            Filter::Not(Box::new(Filter::Prefix(n("path"), n("/home/user1")))),
            Filter::And(vec![Filter::Eq(n("kind"), text("Mail")), Filter::Range(n("sorted"), Some(1_700_000_000 + 60 * 1024), Some(1_700_000_000 + 60 * 2047)), Filter::Not(Box::new(Filter::Exists(n("small"))))]),
            Filter::Or(vec![Filter::Eq(n("kind"), text("jpg")), Filter::Range(n("small"), Some(100), None), Filter::Contains(n("path"), n("user12/"))]),
            Filter::Eq(n("tags"), text("common")),
            Filter::Eq(n("tags"), text("COMMON")),
            Filter::Eq(n("tags"), text("t35")),
            Filter::Eq(n("tags"), text("t36")),
            Filter::Eq(n("tags"), int(1)),
            Filter::Eq(n("tags"), Value::Tags(vec!["common".into()])),
            Filter::In(n("tags"), vec![text("t0"), text("t695"), text("Mixed"), text("none"), int(3)]),
            Filter::In(n("tags"), vec![]),
            Filter::Exists(n("tags")),
            Filter::Not(Box::new(Filter::Exists(n("tags")))),
            Filter::Not(Box::new(Filter::Eq(n("tags"), text("common")))),
            Filter::Prefix(n("tags"), n("T1")),
            Filter::Prefix(n("tags"), n("ü")),
            Filter::Prefix(n("tags"), n("")),
            Filter::Contains(n("tags"), n("OMM")),
            Filter::Contains(n("tags"), n("")),
            Filter::Contains(n("tags"), n("zzz")),
            Filter::Range(n("tags"), Some(0), Some(10)),
            Filter::And(vec![Filter::Eq(n("tags"), text("common")), Filter::Eq(n("tags"), text("Mixed")), Filter::Not(Box::new(Filter::Eq(n("tags"), text("Über"))))]),
            Filter::Or(vec![Filter::Eq(n("tags"), text("t5")), Filter::Eq(n("kind"), text("pdf"))]),
            Filter::Glob(n("path"), n("/home/user7/*"), false),
            Filter::Glob(n("path"), n("/home/user?/Doc4?.txt"), false),
            Filter::Glob(n("path"), n("*/doc49.TXT"), true),
            Filter::Glob(n("path"), n("*/doc49.TXT"), false),
            Filter::Glob(n("path"), n("*"), false),
            Filter::Glob(n("path"), n("/home/user1*/Doc*7.txt"), false),
            Filter::Glob(n("kind"), n("?df"), true),
            Filter::Glob(n("kind"), n("pdf"), false),
            Filter::Glob(n("kind"), n(""), false),
            Filter::Glob(n("tags"), n("t?"), false),
            Filter::Glob(n("tags"), n("*ber"), false),
            Filter::Glob(n("tags"), n("ü*"), true),
            Filter::Glob(n("tags"), n("t1*5"), false),
            Filter::Not(Box::new(Filter::Glob(n("tags"), n("*"), false))),
            Filter::Glob(n("nothing"), n("*"), false),
            Filter::Glob(n("small"), n("*"), false),
            Filter::Eq(n("all"), text("every")),
            Filter::Eq(n("all"), text("none")),
            Filter::Exists(n("all")),
            Filter::Prefix(n("all"), n("EV")),
            Filter::Exists(n("empty")),
            Filter::Eq(n("empty"), text("every")),
            Filter::Not(Box::new(Filter::Contains(n("empty"), n("")))),
        ];
        for t in (0..700).step_by(9) {
            filters.push(Filter::Eq(n("tags"), text(&format!("t{t}"))));
            filters.push(Filter::Prefix(n("tags"), format!("t{}", t % 70)));
        }
        // And a few hundred made up: ranges and values around what the columns hold.
        for _ in 0..300 {
            let r = next();
            let (a, b) = ((r % 400) as i64 - 60, ((r >> 16) % 400) as i64 - 60);
            filters.push(match r >> 60 & 3 {
                0 => Filter::Range(n("small"), Some(a.min(b)), Some(a.max(b))),
                1 => Filter::Range(n("sorted"), Some(1_700_000_000 + a * 733), Some(1_700_000_000 + b * 1777)),
                2 => Filter::Range(n("wide"), Some(a * 7_000_000), Some(b * 9_000_000)),
                _ => Filter::Eq(n("path"), text(&format!("/home/user{}/Doc{}.txt", (r >> 8) % 41, (r >> 24) % 5100))),
            });
        }
        for f in &filters {
            let b = f.bind(&seg.attr_columns());
            let got = seg.matching(&b);
            let want: Vec<u32> = (0..seg.docs).filter(|&d| b.matches(&seg.row(d))).collect();
            assert_eq!(got.count(), want.len(), "{f:?}");
            assert_eq!(got.ones().collect::<Vec<_>>(), want, "{f:?}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Prefixes and parts without case, on a dictionary: the same documents as lowercasing every
    /// document's string, with capitals, letters outside ASCII, and the two characters outside
    /// ASCII whose lowercase is inside it (İ and the Kelvin sign), in a column that has them and
    /// in one that doesn't.
    #[test]
    fn strings_without_case_agree_with_rows() {
        let dir = std::env::temp_dir().join(format!("hinny-case-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut rng = 0xD1B5_4A32_D192_ED03u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let plain = ["report", "Report", "REPORT", "rep", "kelvin", "Kiosk", "invoice", "Index", "is", "IS", "müller", "MÜLLER", "Ärger", "straße", "ΣΟΦΟΣ", "σοφός", "a.b", "A_B", "@home", "`home", "ix", "I", "k", "K"];
        let odd = ["\u{130}stanbul", "\u{130}", "x\u{130}", "\u{212A}elvin", "0\u{212A}", "\u{212A}", "i\u{307}stanbul", "D\u{130}K\u{130}"];
        let mut m = MemIndex::new(1);
        for d in 0..3000u64 {
            let word = |all: bool, r: u64| {
                let r = r as usize;
                if all && r % 5 == 0 { odd[(r >> 8) % odd.len()] } else { plain[(r >> 8) % plain.len()] }
            };
            // One to three words, some run together.
            let join = ["", " ", "/"][(next() % 3) as usize];
            let ascii_and_more = [word(false, next()), word(false, next()), word(false, next())][..1 + (next() % 3) as usize].join(join);
            let with_odd = [word(true, next()), word(true, next()), word(true, next())][..1 + (next() % 3) as usize].join(join);
            let mut attrs = vec![("usual".to_string(), Value::Str(ascii_and_more)), ("odd".to_string(), Value::Str(with_odd))];
            if d % 7 == 0 {
                attrs.pop();
            }
            m.add_full(d, 1, 0, &["x"], &attrs, None);
        }
        let path = dir.join("seg");
        write_chunks(&path, &[&m], 0, None, 0).unwrap();
        let seg = Segment::open(&path).unwrap();
        let flags: Vec<(String, bool)> = seg.attrs.iter().zip(&seg.layouts).map(|((n, _, _), l)| (n.clone(), l.unwrap().into_ascii)).collect();
        assert!(flags.contains(&("usual".to_string(), false)) && flags.contains(&("odd".to_string(), true)), "{flags:?}");

        // Conditions cut out of the strings there are, as written and lowercased, and a few that
        // match nothing or everything.
        let mut parts: Vec<String> = ["", "i", "I", "k", "K", "is", "ki", "ik", "i\u{307}", "\u{130}", "\u{212A}", "ς", "σ", "Σ", "ss", "ß", "nothing", "t r", "t/r", "@", "`", "."].iter().map(|s| s.to_string()).collect();
        for word in plain.iter().chain(&odd) {
            for form in [word.to_string(), word.to_lowercase()] {
                let at: Vec<usize> = form.char_indices().map(|(i, _)| i).chain([form.len()]).collect();
                for (i, &a) in at.iter().enumerate() {
                    for &b in &at[i + 1..] {
                        parts.push(form[a..b].to_string());
                    }
                }
            }
        }
        parts.sort_unstable();
        parts.dedup();
        let mut some = 0;
        for column in ["usual", "odd"] {
            for part in &parts {
                for f in [Filter::Prefix(column.to_string(), part.clone()), Filter::Contains(column.to_string(), part.clone())] {
                    let b = f.bind(&seg.attr_columns());
                    let got = seg.matching(&b);
                    let want: Vec<u32> = (0..seg.docs).filter(|&d| b.matches(&seg.row(d))).collect();
                    some += !want.is_empty() as usize;
                    assert_eq!(got.ones().collect::<Vec<_>>(), want, "{f:?}");
                }
            }
        }
        assert!(some > 400, "{some} conditions matched something, of {}", parts.len() * 4);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn matching_agrees_with_rows() {
        let dir = std::env::temp_dir().join(format!("hinny-matching-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut m = MemIndex::new(1);
        for d in 0..1000u64 {
            let mut attrs = vec![("kind".to_string(), Value::Str(["pdf", "Mail", "jpg"][d as usize % 3].into()))];
            // Missing in some documents, negative in others.
            if d % 7 != 0 {
                attrs.push(("n".to_string(), Value::Int(d as i64 % 50 - 10)));
            }
            m.add_full(d, 1, 0, &["x"], &attrs, None);
        }
        let path = dir.join("seg");
        write_chunks(&path, &[&m], 0, None, 0).unwrap();
        let seg = Segment::open(&path).unwrap();
        let n = |s: &str| s.to_string();
        let filters = [
            Filter::Eq(n("n"), Value::Int(3)),
            Filter::Eq(n("n"), Value::Int(NO_INT)),
            Filter::Range(n("n"), Some(-5), Some(20)),
            Filter::Range(n("n"), None, Some(0)),
            Filter::Exists(n("n")),
            Filter::Exists(n("nothing")),
            Filter::In(n("n"), vec![Value::Int(1), Value::Int(30)]),
            Filter::Eq(n("kind"), Value::Str("Mail".into())),
            Filter::Prefix(n("kind"), "p".into()),
            Filter::Not(Box::new(Filter::Range(n("n"), Some(0), None))),
            Filter::And(vec![Filter::Range(n("n"), Some(0), None), Filter::Or(vec![Filter::Eq(n("kind"), Value::Str("pdf".into())), Filter::Not(Box::new(Filter::Exists(n("n"))))])]),
            Filter::And(vec![]),
            Filter::Or(vec![]),
        ];
        for f in &filters {
            let b = f.bind(&seg.attr_columns());
            let got = seg.matching(&b);
            let want: Vec<u32> = (0..seg.docs).filter(|&d| b.matches(&seg.row(d))).collect();
            assert_eq!(got.ones().collect::<Vec<_>>(), want, "{f:?}");
            assert_eq!(got.count(), want.len(), "{f:?}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
