//! Answers for one- and two-letter queries, worked out in advance.
//!
//! A first keystroke matches millions of names, so scanning for it reads all ~232 MB of names to
//! keep ten. Instead, for every first letter and every two-letter word start ("in" in "Invoice",
//! "my_inbox"), this keeps the best `PER_KEY` entries, ranked with the same score a scan would give
//! them. A one- or two-letter query then reads a short list: microseconds instead of milliseconds.
//!
//! Stored next to the index (`names.idx.starts`, ~10 MB), tied to the build it was made from;
//! entries added since are checked separately, deleted ones skipped on read.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use rayon::prelude::*;

use crate::search::{Query, rank};
use crate::store::Index;

const MAGIC: u32 = 0x5357_5753; // "SWWS"
const VERSION: u32 = 1;
/// Kept per key: far more than a result list shows, so deletions and live additions don't empty it.
pub const PER_KEY: usize = 1024;
/// One-letter keys 0..256, two-letter keys 256 + (first << 8 | second).
const KEYS: usize = 256 + 65_536;

fn key(bytes: &[u8]) -> Option<usize> {
    match bytes {
        [a] => Some(*a as usize),
        [a, b] => Some(256 + ((*a as usize) << 8 | *b as usize)),
        _ => None,
    }
}

pub struct WordStarts {
    map: Mmap,
    pub build_id: u64,
}

impl WordStarts {
    pub fn path_for(index: &Path) -> PathBuf {
        let mut p = index.as_os_str().to_owned();
        p.push(".starts");
        PathBuf::from(p)
    }

    /// Ranks every entry under each of its keys (all cores) and writes the best `PER_KEY` per key.
    pub fn build(index: &Index, path: &Path, now: u32) -> std::io::Result<usize> {
        // One chunk per thread: each holds a list per key (65 k of them), so more chunks only cost memory.
        let chunks = rayon::current_num_threads();
        let per = index.count.div_ceil(chunks).max(1);
        // Per chunk: (key, score, entry) for every key an entry has; trimmed per key as it grows.
        let lists: Vec<Vec<Vec<(f32, u32)>>> = (0..chunks)
            .into_par_iter()
            .map(|c| {
                let mut by_key: Vec<Vec<(f32, u32)>> = vec![Vec::new(); KEYS];
                let mut seen: Vec<usize> = Vec::with_capacity(8);
                for e in c * per..((c + 1) * per).min(index.count) {
                    if index.is_deleted(e) {
                        continue;
                    }
                    let name = index.folded_name(e);
                    if name.is_empty() {
                        continue;
                    }
                    // One letter: the start of the name (as the scan matches it).
                    push(index, e, name, 0, &name[..1], now, &mut by_key);
                    // Two letters: each word start, once per key (the first, best place).
                    seen.clear();
                    for p in 0..name.len().saturating_sub(1) {
                        if p > 0 && name[p - 1].is_ascii_alphanumeric() {
                            continue;
                        }
                        let k = key(&name[p..p + 2]).unwrap();
                        if seen.contains(&k) {
                            continue;
                        }
                        seen.push(k);
                        push(index, e, name, p, &name[p..p + 2], now, &mut by_key);
                    }
                }
                by_key
            })
            .collect();
        // Merge the chunks: best PER_KEY per key, best first.
        let merged: Vec<Vec<u32>> = (0..KEYS)
            .into_par_iter()
            .map(|k| {
                let mut all: Vec<(f32, u32)> = lists.iter().flat_map(|l| l[k].iter().copied()).collect();
                trim(&mut all);
                all.into_iter().map(|(_, e)| e).collect()
            })
            .collect();
        write(path, index.build_id, &merged)
    }

    pub fn open(path: &Path, index: &Index) -> Option<WordStarts> {
        let file = File::open(path).ok()?;
        let map = unsafe { Mmap::map(&file).ok()? };
        if map.len() < 24 || u32::from_le_bytes(map[0..4].try_into().ok()?) != MAGIC || u32::from_le_bytes(map[4..8].try_into().ok()?) != VERSION {
            return None;
        }
        let build_id = u64::from_le_bytes(map[8..16].try_into().ok()?);
        // Made from another build: its entry numbers don't match this index.
        if build_id != index.build_id || map.len() < 16 + KEYS * 8 {
            return None;
        }
        Some(WordStarts { map, build_id })
    }

    /// The best entries for a one- or two-byte (folded) query, best first.
    pub fn get(&self, bytes: &[u8]) -> &[u32] {
        let Some(k) = key(bytes) else { return &[] };
        let table = 16 + k * 8;
        let start = u32::from_le_bytes(self.map[table..table + 4].try_into().unwrap()) as usize;
        let len = u32::from_le_bytes(self.map[table + 4..table + 8].try_into().unwrap()) as usize;
        let base = 16 + KEYS * 8 + start * 4;
        let bytes = &self.map[base..base + len * 4];
        // The file is written 4-byte aligned after the table, so this cast is exact.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, len) }
    }
}

#[inline]
fn push(index: &Index, e: usize, name: &[u8], at: usize, query: &[u8], now: u32, by_key: &mut [Vec<(f32, u32)>]) {
    let q = Query { main: query, others: &[], whole: true, now, limit: PER_KEY };
    let Some(score) = rank(index, e, name, at, &q) else { return };
    let list = &mut by_key[key(query).unwrap()];
    list.push((score, e as u32));
    if list.len() >= PER_KEY * 4 {
        trim(list);
    }
}

fn trim(list: &mut Vec<(f32, u32)>) {
    if list.len() > PER_KEY {
        list.select_nth_unstable_by(PER_KEY, |a, b| b.0.total_cmp(&a.0));
        list.truncate(PER_KEY);
    }
    list.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
}

fn write(path: &Path, build_id: u64, lists: &[Vec<u32>]) -> std::io::Result<usize> {
    let tmp = path.with_extension("starts-tmp");
    let mut out = BufWriter::with_capacity(4 << 20, File::create(&tmp)?);
    out.write_all(&MAGIC.to_le_bytes())?;
    out.write_all(&VERSION.to_le_bytes())?;
    out.write_all(&build_id.to_le_bytes())?;
    let mut start = 0u32;
    for list in lists {
        out.write_all(&start.to_le_bytes())?;
        out.write_all(&(list.len() as u32).to_le_bytes())?;
        start += list.len() as u32;
    }
    for list in lists {
        for e in list {
            out.write_all(&e.to_le_bytes())?;
        }
    }
    out.flush()?;
    drop(out);
    std::fs::rename(tmp, path)?;
    Ok(start as usize)
}
