//! Stored text, kept apart from the index: blob files of zstd-compressed blocks that segments
//! point into. A merge copies a segment's 16-byte references instead of rewriting its text, the
//! key–value separation of WiscKey (FAST '16) and RocksDB's BlobDB: in a log-structured index that
//! stores documents, rewriting their text on every merge is most of the merge's work. A blob file
//! is rewritten only when most of it is dead (or by `optimize`), and deleted once no segment
//! points into it.
//!
//! A blob file: a 24-byte header ("fugublob", the format, a reserved word, the record count),
//! then blocks: `[compressed length u32][length u32][zstd bytes]`. A block holds whole records,
//! about 4 KiB of them, so a hit's text costs one small decompression. A record: `[fields
//! u32][length u32 per field][the texts]`.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

/// The blob file format this version writes (and the newest it reads).
pub const VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"fugublob";
const HEADER: usize = 24;
/// Records are packed into blocks of about this many bytes before compression.
const BLOCK: usize = 4 * 1024;
const LEVEL: i32 = 3;

/// A block compressed, with zstd's checksum of its content in the frame: reading it back checks
/// the text against it (blocks written before carry none, and are read as they are).
fn pack(block: &[u8]) -> std::io::Result<Vec<u8>> {
    thread_local! {
        static PACK: std::cell::RefCell<Option<zstd::bulk::Compressor<'static>>> = const { std::cell::RefCell::new(None) };
    }
    PACK.with(|p| {
        let mut p = p.borrow_mut();
        if p.is_none() {
            let mut c = zstd::bulk::Compressor::new(LEVEL)?;
            c.include_checksum(true)?;
            *p = Some(c);
        }
        p.as_mut().unwrap().compress(block)
    })
}

/// Where a record is: which blob file, the block's offset in it, and the record's offset in the
/// decompressed block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextRef {
    pub blob: u64,
    pub block: u64,
    pub inner: u32,
}

pub fn path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("blob-{id:08}.fzt"))
}

/// The id of a blob file, from its name.
pub fn id_of(name: &str) -> Option<u64> {
    name.strip_prefix("blob-")?.strip_suffix(".fzt")?.parse().ok()
}

pub struct Writer {
    id: u64,
    tmp: PathBuf,
    path: PathBuf,
    out: BufWriter<File>,
    pos: u64,
    block: Vec<u8>,
    records: u64,
}

impl Writer {
    pub fn create(dir: &Path, id: u64) -> std::io::Result<Writer> {
        let path = path(dir, id);
        let tmp = path.with_extension("tmp");
        let mut out = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
        out.write_all(&[0u8; HEADER])?;
        Ok(Writer { id, tmp, path, out, pos: HEADER as u64, block: Vec::with_capacity(BLOCK * 2), records: 0 })
    }

    /// Adds a record (one text per field) and says where it is.
    pub fn add(&mut self, texts: &[&str]) -> std::io::Result<TextRef> {
        let size = 4 + 4 * texts.len() + texts.iter().map(|t| t.len()).sum::<usize>();
        if !self.block.is_empty() && self.block.len() + size > BLOCK {
            self.flush_block()?;
        }
        let r = TextRef { blob: self.id, block: self.pos, inner: self.block.len() as u32 };
        self.block.extend_from_slice(&(texts.len() as u32).to_le_bytes());
        for t in texts {
            self.block.extend_from_slice(&(t.len() as u32).to_le_bytes());
        }
        for t in texts {
            self.block.extend_from_slice(t.as_bytes());
        }
        self.records += 1;
        Ok(r)
    }

    fn flush_block(&mut self) -> std::io::Result<()> {
        if self.block.is_empty() {
            return Ok(());
        }
        let packed = pack(&self.block)?;
        let len = self.block.len();
        self.put_block(&packed, len)?;
        self.block.clear();
        Ok(())
    }

    fn put_block(&mut self, packed: &[u8], len: usize) -> std::io::Result<()> {
        self.out.write_all(&(packed.len() as u32).to_le_bytes())?;
        self.out.write_all(&(len as u32).to_le_bytes())?;
        self.out.write_all(packed)?;
        self.pos += 8 + packed.len() as u64;
        Ok(())
    }

    /// Writes the rest, syncs, and puts the file in place. Before any segment points into it.
    pub fn finish(mut self) -> std::io::Result<u64> {
        self.flush_block()?;
        let mut file = self.out.into_inner().map_err(|e| e.into_error())?;
        let mut header = Vec::with_capacity(HEADER);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&self.records.to_le_bytes());
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&self.tmp, &self.path)?;
        Ok(self.records)
    }
}

/// Writes `records` (each one text per field) as blob file `id`, its blocks compressed at once on
/// the current thread pool, and returns their references in order. The same file `Writer` makes;
/// a flush's text compressed one block after another had been a third of the flush.
pub fn write(dir: &Path, id: u64, records: &[&[String]]) -> std::io::Result<Vec<TextRef>> {
    use rayon::prelude::*;
    // Packed as `Writer::add` does: (block, offset in it) per record.
    let mut blocks: Vec<Vec<u8>> = vec![Vec::with_capacity(BLOCK * 2)];
    let mut at: Vec<(usize, u32)> = Vec::with_capacity(records.len());
    for texts in records {
        let size = 4 + 4 * texts.len() + texts.iter().map(|t| t.len()).sum::<usize>();
        let last = blocks.last().unwrap();
        if !last.is_empty() && last.len() + size > BLOCK {
            blocks.push(Vec::with_capacity(BLOCK * 2));
        }
        let n = blocks.len() - 1;
        let b = &mut blocks[n];
        at.push((n, b.len() as u32));
        b.extend_from_slice(&(texts.len() as u32).to_le_bytes());
        for t in texts.iter() {
            b.extend_from_slice(&(t.len() as u32).to_le_bytes());
        }
        for t in texts.iter() {
            b.extend_from_slice(t.as_bytes());
        }
    }
    let packed: Vec<Vec<u8>> = blocks.par_iter().filter(|b| !b.is_empty()).map(|b| pack(b)).collect::<std::io::Result<_>>()?;
    let mut w = Writer::create(dir, id)?;
    let mut offsets = Vec::with_capacity(packed.len());
    for (p, b) in packed.iter().zip(&blocks) {
        offsets.push(w.pos);
        w.put_block(p, b.len())?;
    }
    w.records = records.len() as u64;
    w.finish()?;
    Ok(at.into_iter().map(|(b, inner)| TextRef { blob: id, block: offsets[b], inner }).collect())
}

/// A blob file, mapped.
pub struct Blob {
    map: Mmap,
    pub records: u64,
}

impl Blob {
    pub fn open(path: &Path) -> std::io::Result<Blob> {
        let file = File::open(path)?;
        let map = unsafe { Mmap::map(&file)? };
        let bad = || std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{} isn't a blob file", path.display()));
        if map.len() < HEADER || &map[..8] != MAGIC {
            return Err(bad());
        }
        let version = u32::from_le_bytes(map[8..12].try_into().unwrap());
        if version > VERSION {
            return Err(crate::index::too_new(&format!("blob file {}", path.display()), version, VERSION));
        }
        let records = u64::from_le_bytes(map[16..24].try_into().unwrap());
        Ok(Blob { map, records })
    }

    /// The block at `at`, decompressed (and checked against its checksum, if it was written with
    /// one), and where the next one starts; none if it's damaged or cut short.
    fn block(&self, at: usize) -> Option<(Vec<u8>, usize)> {
        let head = self.map.get(at..at + 8)?;
        let (clen, len) = (u32::from_le_bytes(head[..4].try_into().unwrap()) as usize, u32::from_le_bytes(head[4..].try_into().unwrap()) as usize);
        thread_local! {
            // A decompression context is worth keeping: making one per read cost more than the
            // decompression of a small block.
            static UNPACK: std::cell::RefCell<Option<zstd::bulk::Decompressor<'static>>> = const { std::cell::RefCell::new(None) };
        }
        let block = UNPACK.with(|u| {
            let mut u = u.borrow_mut();
            if u.is_none() {
                *u = zstd::bulk::Decompressor::new().ok();
            }
            u.as_mut()?.decompress(self.map.get(at + 8..at + 8 + clen)?, len).ok()
        })?;
        (block.len() == len).then_some((block, at + 8 + clen))
    }

    /// Reads every block: an error says where the first damaged one starts. (Each is checked
    /// against the checksum written with it; blocks from before there were any are only read.)
    pub fn verify(&self) -> std::io::Result<()> {
        let mut at = HEADER;
        while at < self.map.len() {
            let Some((_, next)) = self.block(at) else {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("the block at byte {at} is damaged")));
            };
            at = next;
        }
        Ok(())
    }

    /// The texts of the record at `r` (its block decompressed), or none if the file is damaged.
    pub fn read(&self, r: TextRef) -> Option<Vec<String>> {
        let (block, _) = self.block(r.block as usize)?;
        let mut p = r.inner as usize;
        let n = u32::from_le_bytes(block.get(p..p + 4)?.try_into().unwrap()) as usize;
        p += 4;
        let lens: Vec<usize> = (0..n).map(|i| block.get(p + i * 4..p + i * 4 + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)).collect::<Option<_>>()?;
        p += 4 * n;
        let mut out = Vec::with_capacity(n);
        for l in lens {
            out.push(String::from_utf8_lossy(block.get(p..p + l)?).into_owned());
            p += l;
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip_across_blocks() {
        let dir = std::env::temp_dir().join(format!("hinny-blob-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut w = Writer::create(&dir, 7).unwrap();
        let big = "Größe ".repeat(10_000); // a record bigger than a block
        let texts: Vec<(String, String)> = (0..2000).map(|i| (format!("title {i}"), if i == 1000 { big.clone() } else { format!("body {i} ").repeat(i % 50) })).collect();
        let refs: Vec<TextRef> = texts.iter().map(|(a, b)| w.add(&[a, b]).unwrap()).collect();
        assert_eq!(w.finish().unwrap(), 2000);
        let blob = Blob::open(&path(&dir, 7)).unwrap();
        assert_eq!(blob.records, 2000);
        assert!(refs.windows(2).any(|p| p[0].block != p[1].block), "more than one block");
        for (r, (a, b)) in refs.iter().zip(&texts) {
            assert_eq!(blob.read(*r).unwrap(), vec![a.clone(), b.clone()]);
        }
        // Text compresses: well under the raw size.
        let raw: usize = texts.iter().map(|(a, b)| a.len() + b.len()).sum();
        assert!(std::fs::metadata(path(&dir, 7)).unwrap().len() < raw as u64 / 3);
        assert_eq!(id_of("blob-00000007.fzt"), Some(7));
        // Written at once: the same file.
        let owned: Vec<Vec<String>> = texts.iter().map(|(a, b)| vec![a.clone(), b.clone()]).collect();
        let records: Vec<&[String]> = owned.iter().map(Vec::as_slice).collect();
        let batch = write(&dir, 8, &records).unwrap();
        assert_eq!(batch.iter().map(|r| (r.block, r.inner)).collect::<Vec<_>>(), refs.iter().map(|r| (r.block, r.inner)).collect::<Vec<_>>());
        assert_eq!(std::fs::read(path(&dir, 8)).unwrap(), std::fs::read(path(&dir, 7)).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
