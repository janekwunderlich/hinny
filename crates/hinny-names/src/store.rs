//! The column store: every entry's name and metadata, column by column, in one file.
//!
//! - Children of a folder are contiguous, so a path lookup walks from the root and paths are never
//!   stored (they're rebuilt from parent links, only for results shown).
//! - The matching copy of all names is one folded byte array with a 0 after each name: a search is
//!   one scan over it (the 0 keeps matches from spanning two names).
//! - The file is memory-mapped read-only: opening is instant whatever the size, pages load when
//!   touched, and they stay clean (macOS can drop and re-read them, and they don't count toward the
//!   app's memory footprint). Live updates go to small overlays: a deleted-entries bitmap and the
//!   changed dates. (A copy-on-write mapping counted every page read as dirty: 604 MB of footprint.)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use memmap2::Mmap;

use crate::fold::fold;
use crate::walk::{self, Job, RawEntry};

const MAGIC: u32 = 0x5857_5753; // "SWWX"
const VERSION: u32 = 2;
const JOURNAL_MAGIC: u32 = 0x4A57_5753; // "SWWJ"
pub const NONE: u32 = u32::MAX;
/// Depth's top bit marks a deleted entry (until the next save compacts it away).
const DELETED: u8 = 0x80;

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct Traits(pub u8);

impl Traits {
    pub const DIRECTORY: u8 = 1 << 0;
    /// Its name or a folder above it starts with ".". Inherited.
    pub const HIDDEN: u8 = 1 << 1;
    /// Inside ~/Library. Inherited.
    pub const LIBRARY: u8 = 1 << 2;
    /// Inside a code project (a folder with package.json, .git, Cargo.toml…). Inherited.
    pub const CODE: u8 = 1 << 3;
    /// Inside a bundle (an app's or a library's contents). Inherited.
    pub const IN_PACKAGE: u8 = 1 << 4;
    pub const APPLICATION: u8 = 1 << 5;
    /// A kind people open: PDF, Office, iWork, images, media, text.
    pub const DOCUMENT: u8 = 1 << 6;
    /// Source code, build output, data files.
    pub const CODE_FILE: u8 = 1 << 7;
    const INHERITED: u8 = Self::HIDDEN | Self::LIBRARY | Self::CODE | Self::IN_PACKAGE;

    pub fn has(self, bit: u8) -> bool {
        self.0 & bit != 0
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Header {
    magic: u32,
    version: u32,
    count: u64,
    root_bytes: u64,
    name_bytes: u64,
    folded_bytes: u64,
    last_event_id: u64,
    /// Changes with every write; the journal names the build it applies to.
    build_id: u64,
}

// MARK: Classifying names

const PROJECT_MARKERS: &[&[u8]] = &[b"package.json", b"Cargo.toml", b"pyproject.toml", b"Package.swift", b"go.mod", b".git", b"Gemfile", b"pom.xml", b"build.gradle", b"CMakeLists.txt"];
const PACKAGES: &[&str] = &["app", "bundle", "framework", "photoslibrary", "musiclibrary", "tvlibrary", "xcodeproj", "xcworkspace", "plugin", "kext", "appex", "xpc", "lproj", "dsym", "xcarchive", "pkg", "mpkg", "rtfd", "pages", "key", "numbers", "sparsebundle"];
const DOCUMENTS: &[&str] = &["pdf", "doc", "docx", "pages", "rtf", "rtfd", "odt", "txt", "md", "markdown", "tex", "epub", "xls", "xlsx", "numbers", "csv", "tsv", "ods", "ppt", "pptx", "key", "odp", "html", "htm", "eml", "ics", "vcf", "png", "jpg", "jpeg", "heic", "gif", "tiff", "tif", "webp", "svg", "psd", "sketch", "fig", "ai", "mp3", "m4a", "wav", "aiff", "flac", "mov", "mp4", "m4v", "mkv", "zip", "dmg"];
const CODE_FILES: &[&str] = &["swift", "ts", "tsx", "js", "jsx", "mjs", "cjs", "py", "pyi", "pyc", "rs", "go", "c", "h", "cc", "cpp", "hpp", "m", "mm", "java", "kt", "rb", "php", "sh", "zsh", "json", "yaml", "yml", "toml", "lock", "xml", "plist", "log", "o", "a", "dylib", "so", "class", "jar", "map", "css", "scss", "d", "ilean", "olean", "ir", "pack", "idx", "db", "sqlite", "wasm", "node", "nib", "car", "strings", "rlib", "rmeta"];

fn extension(name: &[u8]) -> String {
    match name.iter().rposition(|&b| b == b'.') {
        Some(dot) if dot > 0 => String::from_utf8_lossy(&name[dot + 1..]).to_ascii_lowercase(),
        _ => String::new(),
    }
}

/// An entry's own traits, and what it passes on to the entries inside it.
fn classify(name: &[u8], is_dir: bool, inherited: u8) -> (u8, u8) {
    let ext = extension(name);
    let mut own = inherited;
    if name.first() == Some(&b'.') {
        own |= Traits::HIDDEN;
    }
    if is_dir {
        own |= Traits::DIRECTORY;
        if ext == "app" {
            own |= Traits::APPLICATION;
        }
    }
    if DOCUMENTS.contains(&ext.as_str()) {
        own |= Traits::DOCUMENT;
    }
    if !is_dir && CODE_FILES.contains(&ext.as_str()) {
        own |= Traits::CODE_FILE;
    }
    let mut pass_on = own & Traits::INHERITED;
    if is_dir && PACKAGES.contains(&ext.as_str()) {
        pass_on |= Traits::IN_PACKAGE;
    }
    (own, pass_on)
}

// MARK: Building

/// Columns being filled by the walker (behind a lock: directories are read on all cores).
#[derive(Default)]
pub struct Builder {
    names: Vec<u8>,
    folded: Vec<u8>,
    name_off: Vec<u32>,
    name_len: Vec<u8>,
    fold_off: Vec<u32>,
    fold_len: Vec<u16>,
    parent: Vec<u32>,
    child_start: Vec<u32>,
    child_count: Vec<u32>,
    traits: Vec<u8>,
    depth: Vec<u8>,
    modified: Vec<u32>,
    /// Folded display names that differ from the name on disk (localized folders): entry → alias.
    aliases: Vec<(u32, Vec<u8>)>,
}

/// An entry folded and classified by a worker, ready to append.
struct Prepared {
    name: Vec<u8>,
    folded: Vec<u8>,
    own: u8,
    pass_on: u8,
    modified: u32,
    descend: bool,
}

/// What the walker carries along with each directory to read.
pub struct Place {
    index: u32,
    pass_on: u8,
    depth: u8,
}

impl Builder {
    fn push(&mut self, name: &[u8], parent: u32, traits: u8, depth: u8, modified: u32) -> u32 {
        let index = self.parent.len() as u32;
        let folded = fold(name);
        self.name_off.push(self.names.len() as u32);
        self.name_len.push(name.len().min(255) as u8);
        self.names.extend_from_slice(&name[..name.len().min(255)]);
        self.fold_off.push(self.folded.len() as u32);
        self.fold_len.push(folded.len().min(u16::MAX as usize) as u16);
        self.folded.extend_from_slice(&folded);
        self.folded.push(0);
        self.parent.push(parent);
        self.child_start.push(NONE);
        self.child_count.push(0);
        self.traits.push(traits);
        self.depth.push(depth.min(127));
        self.modified.push(modified);
        index
    }

    /// Appends an entry whose folding and traits were worked out outside the lock.
    fn push_prepared(&mut self, p: &Prepared, parent: u32, depth: u8) -> u32 {
        let index = self.parent.len() as u32;
        self.name_off.push(self.names.len() as u32);
        self.name_len.push(p.name.len().min(255) as u8);
        self.names.extend_from_slice(&p.name[..p.name.len().min(255)]);
        self.fold_off.push(self.folded.len() as u32);
        self.fold_len.push(p.folded.len().min(u16::MAX as usize) as u16);
        self.folded.extend_from_slice(&p.folded);
        self.folded.push(0);
        self.parent.push(parent);
        self.child_start.push(NONE);
        self.child_count.push(0);
        self.traits.push(p.own);
        self.depth.push(depth.min(127));
        self.modified.push(p.modified);
        index
    }

    /// Reads everything under `root` on all cores.
    pub fn walk(root: &Path) -> Builder {
        let mut builder = Builder::default();
        builder.push(b"", NONE, Traits::DIRECTORY, 0, 0);
        let library = root.join("Library");
        // Cloud drives live under ~/Library but are the user's documents: iCloud Drive, and the
        // File Provider folders (Dropbox, Google Drive, OneDrive). They rank like ~/Documents, with
        // depth counted from the drive.
        let cloud_roots = [library.join("Mobile Documents/com~apple~CloudDocs"), library.join("CloudStorage")];
        let shared = Mutex::new(builder);
        let root_job = Job { path: root.to_path_buf(), data: Place { index: 0, pass_on: 0, depth: 0 } };
        walk::walk(root_job, &|job: &Job<Place>, entries: Vec<RawEntry>| {
            // Folding and classifying happen here, on the worker; the lock only covers the appends.
            let dir = &job.data;
            let mut inherited = dir.pass_on;
            if entries.iter().any(|e| PROJECT_MARKERS.contains(&e.name.as_slice())) {
                inherited |= Traits::CODE;
            }
            // A folder holding ".localized" shows a translated name in Finder: keep it as an alias.
            let alias = if entries.iter().any(|e| e.name == b".localized") {
                crate::localized::display_name(&job.path).map(|n| fold(n.as_bytes()))
            } else {
                None
            };
            let prepared: Vec<Prepared> = entries
                .into_iter()
                .map(|e| {
                    let (own, pass_on) = classify(&e.name, e.is_dir, inherited);
                    let descend = e.is_dir && !e.is_symlink && !e.stop;
                    Prepared { folded: fold(&e.name), name: e.name, own, pass_on, modified: e.modified, descend }
                })
                .collect();
            let (start, n) = {
                let mut b = shared.lock().unwrap();
                if let Some(alias) = alias {
                    b.aliases.push((dir.index, alias));
                }
                let start = b.len() as u32;
                for p in &prepared {
                    b.push_prepared(p, dir.index, dir.depth + 1);
                }
                b.child_start[dir.index as usize] = start;
                b.child_count[dir.index as usize] = prepared.len() as u32;
                (start, prepared.len())
            };
            let mut jobs = Vec::new();
            for (i, p) in prepared.into_iter().enumerate().take(n) {
                if !p.descend {
                    continue;
                }
                let path = job.path.join(std::ffi::OsStr::from_bytes_compat(&p.name));
                let mut pass_on = p.pass_on;
                let mut depth = dir.depth + 1;
                if path == library {
                    pass_on |= Traits::LIBRARY;
                }
                if cloud_roots.contains(&path) {
                    pass_on &= !Traits::LIBRARY;
                    depth = 1;
                }
                jobs.push(Job { path, data: Place { index: start + i as u32, pass_on, depth } });
            }
            jobs
        });
        shared.into_inner().unwrap()
    }

    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Writes the columns to `path` (via a temporary file, renamed into place). Returns the new build id.
    pub fn write(&self, path: &Path, root: &Path, last_event_id: u64) -> std::io::Result<u64> {
        let build_id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("tmp");
        let mut out = BufWriter::with_capacity(4 << 20, File::create(&tmp)?);
        let root_bytes = root.as_os_str().as_encoded_bytes();
        let header = Header {
            magic: MAGIC,
            version: VERSION,
            count: self.len() as u64,
            root_bytes: root_bytes.len() as u64,
            name_bytes: self.names.len() as u64,
            folded_bytes: self.folded.len() as u64,
            last_event_id,
            build_id,
        };
        let mut written = 0usize;
        let mut put = |out: &mut BufWriter<File>, bytes: &[u8], align: bool| -> std::io::Result<()> {
            if align {
                let pad = (16 - written % 16) % 16;
                out.write_all(&[0u8; 16][..pad])?;
                written += pad;
            }
            out.write_all(bytes)?;
            written += bytes.len();
            Ok(())
        };
        put(&mut out, as_bytes(std::slice::from_ref(&header)), false)?;
        put(&mut out, root_bytes, true)?;
        put(&mut out, &self.names, true)?;
        // 32 bytes of padding after the folded names: SIMD loads may read a little past the end.
        let mut folded = self.folded.clone();
        folded.extend_from_slice(&[0u8; 32]);
        put(&mut out, &folded, true)?;
        put(&mut out, as_bytes(&self.name_off), true)?;
        put(&mut out, &self.name_len, true)?;
        put(&mut out, as_bytes(&self.fold_off), true)?;
        put(&mut out, as_bytes(&self.fold_len), true)?;
        put(&mut out, as_bytes(&self.parent), true)?;
        put(&mut out, as_bytes(&self.child_start), true)?;
        put(&mut out, as_bytes(&self.child_count), true)?;
        put(&mut out, &self.traits, true)?;
        put(&mut out, &self.depth, true)?;
        put(&mut out, as_bytes(&self.modified), true)?;
        out.flush()?;
        drop(out);
        std::fs::rename(tmp, path)?;
        write_aliases(&aliases_path(path), build_id, &self.aliases)?;
        Ok(build_id)
    }
}

trait FromBytesCompat {
    fn from_bytes_compat(bytes: &[u8]) -> &Self;
}

impl FromBytesCompat for std::ffi::OsStr {
    fn from_bytes_compat(bytes: &[u8]) -> &Self {
        use std::os::unix::ffi::OsStrExt;
        std::ffi::OsStr::from_bytes(bytes)
    }
}

const ALIASES_MAGIC: u32 = 0x4157_5753; // "SWWA"

pub fn aliases_path(index: &Path) -> PathBuf {
    let mut p = index.as_os_str().to_owned();
    p.push(".aliases");
    PathBuf::from(p)
}

fn write_aliases(path: &Path, build_id: u64, aliases: &[(u32, Vec<u8>)]) -> std::io::Result<()> {
    let mut out = Vec::with_capacity(16 + aliases.len() * 24);
    out.extend_from_slice(&ALIASES_MAGIC.to_le_bytes());
    out.extend_from_slice(&(aliases.len() as u32).to_le_bytes());
    out.extend_from_slice(&build_id.to_le_bytes());
    for (e, alias) in aliases {
        let alias = &alias[..alias.len().min(u16::MAX as usize)];
        out.extend_from_slice(&e.to_le_bytes());
        out.extend_from_slice(&(alias.len() as u16).to_le_bytes());
        out.extend_from_slice(alias);
    }
    std::fs::write(path, out)
}

fn read_aliases(path: &Path, build_id: u64) -> Vec<(u32, Vec<u8>)> {
    let Ok(bytes) = std::fs::read(path) else { return Vec::new() };
    if bytes.len() < 16 || u32::from_le_bytes(bytes[0..4].try_into().unwrap()) != ALIASES_MAGIC || u64::from_le_bytes(bytes[8..16].try_into().unwrap()) != build_id {
        return Vec::new();
    }
    let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(n);
    let mut at = 16;
    for _ in 0..n {
        let Some(head) = bytes.get(at..at + 6) else { break };
        let e = u32::from_le_bytes(head[0..4].try_into().unwrap());
        let len = u16::from_le_bytes(head[4..6].try_into().unwrap()) as usize;
        let Some(alias) = bytes.get(at + 6..at + 6 + len) else { break };
        out.push((e, alias.to_vec()));
        at += 6 + len;
    }
    out
}

fn as_bytes<T: Copy>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

// MARK: The mapped index

/// An entry added by a live update since the file was written.
pub struct Added {
    pub name: Vec<u8>,
    pub folded: Vec<u8>,
    pub parent: u32,
    pub traits: u8,
    pub depth: u8,
    pub modified: u32,
    pub deleted: bool,
}

pub struct Index {
    _map: Mmap,
    pub root: PathBuf,
    pub count: usize,
    pub last_event_id: u64,
    pub build_id: u64,
    pub(crate) names: *const u8,
    pub(crate) folded: *const u8,
    pub folded_bytes: usize,
    pub(crate) name_off: *const u32,
    pub(crate) name_len: *const u8,
    pub(crate) fold_off: *const u32,
    pub(crate) fold_len: *const u16,
    pub(crate) parent: *const u32,
    pub(crate) child_start: *const u32,
    pub(crate) child_count: *const u32,
    pub(crate) traits: *const u8,
    pub(crate) depth: *const u8,
    pub(crate) modified: *const u32,
    /// Entries of the file deleted since it was written (one bit each, ~1.2 MB for 9 M).
    deleted: Vec<u64>,
    /// Dates changed since the file was written, and a bit per entry saying whether to look.
    changed: HashMap<u32, u32>,
    changed_bits: Vec<u64>,
    /// Entries added since the file was written (indexes count, count + 1, …).
    pub added: Vec<Added>,
    /// Children added to a folder since the file was written.
    pub added_children: HashMap<u32, Vec<u32>>,
    pub dirty: bool,
    /// Bumped by every change: remembered search results are only reused within one generation.
    pub generation: u64,
    /// Localized display names (entry, folded alias), searched alongside the names.
    pub aliases: Vec<(u32, Vec<u8>)>,
    /// iCloud Drive and ~/Library/CloudStorage: what's added in them ranks as documents, not ~/Library.
    cloud_roots: Vec<usize>,
    /// Where changes are appended (when opened with `open_with_journal`).
    pub journal: Option<Journal>,
}

// The raw pointers point into the mapping the Index owns; mutation goes through &mut self.
unsafe impl Send for Index {}
unsafe impl Sync for Index {}

impl Index {
    pub fn default_path() -> PathBuf {
        let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"));
        // An index from when this was called Fugu is still found, while there's no Hinny one.
        let support = home.join("Library/Application Support");
        let (new, old) = (support.join("Hinny/names.idx"), support.join("Fugu/names.idx"));
        if !new.exists() && old.exists() { old } else { new }
    }

    /// Maps the file read-only: instant, and only touched pages are read.
    pub fn open(path: &Path) -> std::io::Result<Index> {
        let file = File::open(path)?;
        let map = unsafe { Mmap::map(&file)? };
        let base = map.as_ptr() as *mut u8;
        let header = unsafe { (base as *const Header).read_unaligned() };
        if header.magic != MAGIC || header.version != VERSION {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a names.idx file, or an older version"));
        }
        let count = header.count as usize;
        let mut cursor = std::mem::size_of::<Header>();
        let mut column = |bytes: usize| -> *mut u8 {
            cursor = (cursor + 15) & !15;
            let p = unsafe { base.add(cursor) };
            cursor += bytes;
            p
        };
        let root_ptr = column(header.root_bytes as usize);
        let root = PathBuf::from(std::ffi::OsStr::from_bytes_compat(unsafe { std::slice::from_raw_parts(root_ptr, header.root_bytes as usize) }));
        let names = column(header.name_bytes as usize);
        let folded = column(header.folded_bytes as usize + 32);
        let name_off = column(count * 4).cast::<u32>();
        let name_len = column(count);
        let fold_off = column(count * 4).cast::<u32>();
        let fold_len = column(count * 2).cast::<u16>();
        let parent = column(count * 4).cast::<u32>();
        let child_start = column(count * 4).cast::<u32>();
        let child_count = column(count * 4).cast::<u32>();
        let traits = column(count);
        let depth = column(count);
        let modified = column(count * 4).cast::<u32>();
        Ok(Index {
            _map: map,
            root,
            count,
            last_event_id: header.last_event_id,
            build_id: header.build_id,
            names,
            folded,
            folded_bytes: header.folded_bytes as usize,
            name_off,
            name_len,
            fold_off,
            fold_len,
            parent,
            child_start,
            child_count,
            traits,
            depth,
            modified,
            deleted: vec![0; count / 64 + 1],
            changed: HashMap::new(),
            changed_bits: vec![0; count / 64 + 1],
            added: Vec::new(),
            added_children: HashMap::new(),
            dirty: false,
            generation: 0,
            aliases: read_aliases(&aliases_path(path), header.build_id),
            cloud_roots: Vec::new(),
            journal: None,
        })
        .map(|mut index| {
            let library = index.root.join("Library");
            index.cloud_roots = [library.join("Mobile Documents/com~apple~CloudDocs"), library.join("CloudStorage")].iter().filter_map(|p| index.lookup(p)).collect();
            index
        })
    }

    /// Opens the index and replays its journal (the changes since the file was written), then keeps
    /// appending to it. A journal from another build (left over from before a compaction) is reset.
    pub fn open_with_journal(path: &Path) -> std::io::Result<Index> {
        let mut index = Index::open(path)?;
        let journal_path = Journal::path_for(path);
        if let Ok(bytes) = std::fs::read(&journal_path) {
            if let Some(records) = Journal::records(&bytes, index.build_id) {
                for record in records {
                    match record {
                        Record::Add { parent, is_dir, modified, name } => {
                            index.add(parent as usize, name, is_dir, modified);
                        }
                        Record::Remove(e) => index.remove(e as usize),
                        Record::Modified(e, m) => index.set_modified(e as usize, m),
                        Record::EventId(id) => index.last_event_id = index.last_event_id.max(id),
                    }
                }
                index.journal = Some(Journal::append(&journal_path, bytes.len() as u64, index.added.len())?);
            }
        }
        if index.journal.is_none() {
            index.journal = Some(Journal::create(&journal_path, index.build_id)?);
        }
        index.dirty = false;
        Ok(index)
    }

    /// Appends the buffered changes (and the event ID they bring the index up to) to the journal.
    pub fn flush_journal(&mut self) -> std::io::Result<()> {
        let id = self.last_event_id;
        if let Some(j) = self.journal.as_mut() {
            if !j.buffer.is_empty() {
                j.event_id(id);
            }
            j.flush()?;
        }
        Ok(())
    }

    pub fn total(&self) -> usize {
        self.count + self.added.len()
    }

    pub fn name(&self, i: usize) -> &[u8] {
        if i >= self.count {
            return &self.added[i - self.count].name;
        }
        unsafe { std::slice::from_raw_parts(self.names.add(*self.name_off.add(i) as usize), *self.name_len.add(i) as usize) }
    }

    pub fn folded_name(&self, i: usize) -> &[u8] {
        if i >= self.count {
            return &self.added[i - self.count].folded;
        }
        unsafe { std::slice::from_raw_parts(self.folded.add(*self.fold_off.add(i) as usize), *self.fold_len.add(i) as usize) }
    }

    pub fn traits(&self, i: usize) -> Traits {
        Traits(if i >= self.count { self.added[i - self.count].traits } else { unsafe { *self.traits.add(i) } })
    }

    #[inline]
    pub fn modified(&self, i: usize) -> u32 {
        if i >= self.count {
            return self.added[i - self.count].modified;
        }
        // One bit test per entry; the map only for the few whose date changed.
        if self.changed_bits[i / 64] >> (i % 64) & 1 == 1 {
            if let Some(&m) = self.changed.get(&(i as u32)) {
                return m;
            }
        }
        unsafe { *self.modified.add(i) }
    }

    pub fn depth(&self, i: usize) -> u8 {
        if i >= self.count { self.added[i - self.count].depth } else { unsafe { *self.depth.add(i) & !DELETED } }
    }

    pub fn parent(&self, i: usize) -> u32 {
        if i >= self.count { self.added[i - self.count].parent } else { unsafe { *self.parent.add(i) } }
    }

    #[inline]
    pub fn is_deleted(&self, i: usize) -> bool {
        if i >= self.count { self.added[i - self.count].deleted } else { self.deleted[i / 64] >> (i % 64) & 1 == 1 }
    }

    /// The full path, rebuilt from parent links.
    pub fn path(&self, i: usize) -> PathBuf {
        let mut parts = Vec::new();
        let mut e = i;
        while e != 0 {
            parts.push(self.name(e));
            e = self.parent(e) as usize;
        }
        let mut path = self.root.clone();
        for part in parts.iter().rev() {
            path.push(std::ffi::OsStr::from_bytes_compat(part));
        }
        path
    }

    pub fn children(&self, e: usize) -> Vec<usize> {
        let mut out = Vec::new();
        if e < self.count {
            let start = unsafe { *self.child_start.add(e) };
            if start != NONE {
                let n = unsafe { *self.child_count.add(e) } as usize;
                out.extend(start as usize..start as usize + n);
            }
        }
        if let Some(added) = self.added_children.get(&(e as u32)) {
            out.extend(added.iter().map(|&c| c as usize));
        }
        out
    }

    /// The child of `e` named `name` (no allocation: lookups run for every file system event).
    pub fn child(&self, e: usize, name: &[u8]) -> Option<usize> {
        if e < self.count {
            let start = unsafe { *self.child_start.add(e) };
            if start != NONE {
                let n = unsafe { *self.child_count.add(e) } as usize;
                if let Some(c) = (start as usize..start as usize + n).find(|&c| self.name(c) == name && !self.is_deleted(c)) {
                    return Some(c);
                }
            }
        }
        self.added_children.get(&(e as u32))?.iter().map(|&c| c as usize).find(|&c| self.name(c) == name && !self.is_deleted(c))
    }

    /// Reads the columns a search touches into memory (the folded names, their offsets and lengths,
    /// traits, depths, dates, and the names and parent links paths are rebuilt from), so the first
    /// searches after opening don't wait on the disk (a full scan took 70–90 ms cold, path building
    /// several ms). They're clean file pages macOS can drop and re-read; they don't count toward the
    /// app's memory footprint.
    pub fn prefault(&self) {
        let mut at = 0;
        while let Some(next) = self.prefault_part(at, usize::MAX) {
            at = next;
        }
    }

    /// Reads in up to `budget` bytes of the search columns from offset `at` (counted across all of
    /// them); the offset to continue from, or None when everything is in. In steps, a caller can
    /// hold the index lock briefly for each instead of for the whole ~700 MB.
    pub fn prefault_part(&self, at: usize, budget: usize) -> Option<usize> {
        let mut skip = at;
        let mut left = budget;
        for (ptr, len) in self.prefault_ranges() {
            if skip >= len {
                skip -= len;
                continue;
            }
            let end = len.min(skip.saturating_add(left));
            unsafe {
                let from = ptr.add(skip);
                let span = end - skip;
                libc::madvise(from.sub(from as usize % 16384) as *mut libc::c_void, span + from as usize % 16384, libc::MADV_WILLNEED);
                let mut sum = 0u8;
                let mut i = 0;
                while i < span {
                    sum = sum.wrapping_add(std::ptr::read_volatile(from.add(i)));
                    i += 16384;
                }
                std::hint::black_box(sum);
            }
            left -= end - skip;
            if left == 0 {
                return Some(at + budget);
            }
            skip = 0;
        }
        None
    }

    fn prefault_ranges(&self) -> [(*const u8, usize); 9] {
        let names_bytes = if self.count == 0 { 0 } else { unsafe { *self.name_off.add(self.count - 1) as usize + *self.name_len.add(self.count - 1) as usize } };
        [
            (self.folded, self.folded_bytes),
            // For rebuilding the paths of results: names and parent links.
            (self.names, names_bytes),
            (self.name_off as *const u8, self.count * 4),
            (self.parent as *const u8, self.count * 4),
            (self.fold_off as *const u8, self.count * 4),
            (self.fold_len as *const u8, self.count * 2),
            (self.traits, self.count),
            (self.depth as *const u8, self.count),
            (self.modified as *const u8, self.count * 4),
        ]
    }

    /// The entry for a path under the root (walking children from the root), if indexed.
    pub fn lookup(&self, path: &Path) -> Option<usize> {
        let rel = path.strip_prefix(&self.root).ok()?;
        let mut e = 0;
        for component in rel.components() {
            e = self.child(e, component.as_os_str().as_encoded_bytes())?;
        }
        Some(e)
    }

    // MARK: Live updates

    /// Adds an entry under `parent` (a live update); returns its index.
    pub fn add(&mut self, parent: usize, name: &[u8], is_dir: bool, modified: u32) -> usize {
        // What the parent passes on: its inherited traits, plus "inside a package" for bundles.
        let parent_traits = self.traits(parent).0;
        let mut inherited = parent_traits & Traits::INHERITED;
        let (_, parent_pass) = classify(self.name(parent), true, inherited);
        inherited |= parent_pass;
        if parent == 0 && name == b"Library" {
            inherited |= Traits::LIBRARY;
        }
        let in_cloud_root = self.cloud_roots.contains(&parent);
        if in_cloud_root {
            inherited &= !Traits::LIBRARY;
        }
        let (own, _) = classify(name, is_dir, inherited);
        let index = self.total();
        let depth = if in_cloud_root { 2 } else { self.depth(parent).saturating_add(1).min(127) };
        self.added.push(Added { name: name.to_vec(), folded: fold(name), parent: parent as u32, traits: own, depth, modified, deleted: false });
        if let Some(j) = self.journal.as_mut() {
            j.add(parent as u32, is_dir, modified, name);
        }
        self.added_children.entry(parent as u32).or_default().push(index as u32);
        self.dirty = true;
        self.generation += 1;
        index
    }

    /// Marks an entry and everything under it deleted.
    pub fn remove(&mut self, e: usize) {
        if let Some(j) = self.journal.as_mut() {
            j.remove(e as u32);
        }
        let mut stack = vec![e];
        while let Some(i) = stack.pop() {
            if self.is_deleted(i) {
                continue;
            }
            if i >= self.count {
                self.added[i - self.count].deleted = true;
            } else {
                self.deleted[i / 64] |= 1 << (i % 64);
            }
            stack.extend(self.children(i));
        }
        self.dirty = true;
        self.generation += 1;
    }

    pub fn set_modified(&mut self, e: usize, modified: u32) {
        if self.modified(e) == modified {
            return;
        }
        if let Some(j) = self.journal.as_mut() {
            j.modified(e as u32, modified);
        }
        if e >= self.count {
            self.added[e - self.count].modified = modified;
        } else {
            self.changed.insert(e as u32, modified);
            self.changed_bits[e / 64] |= 1 << (e % 64);
        }
        self.dirty = true;
        self.generation += 1;
    }

    /// Rewrites the file without deleted entries, with added ones in place (children contiguous again).
    pub fn compact_to(&self, path: &Path) -> std::io::Result<usize> {
        let mut b = Builder::default();
        b.push(b"", NONE, Traits::DIRECTORY, 0, 0);
        // Aliases follow their entries to the new numbering.
        let aliased: HashMap<usize, &Vec<u8>> = self.aliases.iter().map(|(e, a)| (*e as usize, a)).collect();
        // Breadth first: each folder's children are pushed together.
        let mut queue = std::collections::VecDeque::from([(0usize, 0u32)]);
        while let Some((old, new)) = queue.pop_front() {
            let kids: Vec<usize> = self.children(old).into_iter().filter(|&c| !self.is_deleted(c)).collect();
            if kids.is_empty() {
                continue;
            }
            let start = b.len() as u32;
            for &c in &kids {
                let idx = b.push(self.name(c), new, self.traits(c).0, self.depth(c), self.modified(c));
                if let Some(alias) = aliased.get(&c) {
                    b.aliases.push((idx, (*alias).clone()));
                }
                if self.traits(c).has(Traits::DIRECTORY) {
                    queue.push_back((c, idx));
                }
            }
            b.child_start[new as usize] = start;
            b.child_count[new as usize] = kids.len() as u32;
        }
        b.write(path, &self.root, self.last_event_id)?;
        Ok(b.len())
    }
}

// MARK: The journal

/// Changes since the index file was written, appended as small binary records. Saving is writing a
/// few bytes per change instead of rewriting the whole file (722 MB for 9 M entries): the file is
/// only rewritten (compacted) when the journal gets big, and preferably on power.
pub struct Journal {
    file: File,
    buffer: Vec<u8>,
    /// Bytes on disk.
    pub bytes: u64,
    /// Entries added or removed since the file was written (what makes compaction worthwhile).
    pub changes: usize,
}

pub enum Record<'a> {
    Add { parent: u32, is_dir: bool, modified: u32, name: &'a [u8] },
    Remove(u32),
    Modified(u32, u32),
    EventId(u64),
}

impl Journal {
    pub fn path_for(index: &Path) -> PathBuf {
        let mut p = index.as_os_str().to_owned();
        p.push(".journal");
        PathBuf::from(p)
    }

    fn create(path: &Path, build_id: u64) -> std::io::Result<Journal> {
        let mut file = File::create(path)?;
        let mut header = Vec::with_capacity(16);
        header.extend_from_slice(&JOURNAL_MAGIC.to_le_bytes());
        header.extend_from_slice(&1u32.to_le_bytes());
        header.extend_from_slice(&build_id.to_le_bytes());
        file.write_all(&header)?;
        Ok(Journal { file, buffer: Vec::new(), bytes: 16, changes: 0 })
    }

    fn append(path: &Path, bytes: u64, changes: usize) -> std::io::Result<Journal> {
        let file = std::fs::OpenOptions::new().append(true).open(path)?;
        Ok(Journal { file, buffer: Vec::new(), bytes, changes })
    }

    /// The records in `bytes` if they belong to build `build_id` (a torn last record is dropped).
    fn records(bytes: &[u8], build_id: u64) -> Option<Vec<Record<'_>>> {
        if bytes.len() < 16 || u32::from_le_bytes(bytes[0..4].try_into().ok()?) != JOURNAL_MAGIC || u64::from_le_bytes(bytes[8..16].try_into().ok()?) != build_id {
            return None;
        }
        let mut out = Vec::new();
        let mut at = 16;
        let u32_at = |i: usize| bytes.get(i..i + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
        while at < bytes.len() {
            let kind = bytes[at];
            match kind {
                1 => {
                    let (Some(parent), Some(&is_dir), Some(modified), Some(len)) = (u32_at(at + 1), bytes.get(at + 5), u32_at(at + 6), bytes.get(at + 10..at + 12)) else { break };
                    let len = u16::from_le_bytes(len.try_into().unwrap()) as usize;
                    let Some(name) = bytes.get(at + 12..at + 12 + len) else { break };
                    out.push(Record::Add { parent, is_dir: is_dir == 1, modified, name });
                    at += 12 + len;
                }
                2 => {
                    let Some(e) = u32_at(at + 1) else { break };
                    out.push(Record::Remove(e));
                    at += 5;
                }
                3 => {
                    let (Some(e), Some(m)) = (u32_at(at + 1), u32_at(at + 5)) else { break };
                    out.push(Record::Modified(e, m));
                    at += 9;
                }
                4 => {
                    let Some(id) = bytes.get(at + 1..at + 9) else { break };
                    out.push(Record::EventId(u64::from_le_bytes(id.try_into().unwrap())));
                    at += 9;
                }
                _ => break,
            }
        }
        Some(out)
    }

    fn add(&mut self, parent: u32, is_dir: bool, modified: u32, name: &[u8]) {
        let name = &name[..name.len().min(u16::MAX as usize)];
        self.buffer.push(1);
        self.buffer.extend_from_slice(&parent.to_le_bytes());
        self.buffer.push(is_dir as u8);
        self.buffer.extend_from_slice(&modified.to_le_bytes());
        self.buffer.extend_from_slice(&(name.len() as u16).to_le_bytes());
        self.buffer.extend_from_slice(name);
        self.changes += 1;
    }

    fn remove(&mut self, e: u32) {
        self.buffer.push(2);
        self.buffer.extend_from_slice(&e.to_le_bytes());
        self.changes += 1;
    }

    fn modified(&mut self, e: u32, modified: u32) {
        self.buffer.push(3);
        self.buffer.extend_from_slice(&e.to_le_bytes());
        self.buffer.extend_from_slice(&modified.to_le_bytes());
    }

    fn event_id(&mut self, id: u64) {
        self.buffer.push(4);
        self.buffer.extend_from_slice(&id.to_le_bytes());
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        self.file.write_all(&self.buffer)?;
        self.bytes += self.buffer.len() as u64;
        self.buffer.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> PathBuf {
        let root = std::env::temp_dir().join(format!("hinny-names-test-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(root.join("Docs/Taxes")).unwrap();
        std::fs::write(root.join("Docs/Taxes/Return 2025.pdf"), b"x").unwrap();
        std::fs::write(root.join("Docs/notes.md"), b"x").unwrap();
        root
    }

    #[test]
    fn journal_survives_reopen() {
        let root = tree();
        let file = root.with_extension("idx");
        Builder::walk(&root).write(&file, &root, 1).unwrap();

        let mut ix = Index::open_with_journal(&file).unwrap();
        let taxes = ix.lookup(&root.join("Docs/Taxes")).unwrap();
        let added = ix.add(taxes, b"Receipt.pdf", false, 42);
        let notes = ix.lookup(&root.join("Docs/notes.md")).unwrap();
        ix.remove(notes);
        ix.last_event_id = 99;
        ix.flush_journal().unwrap();
        drop(ix);

        // A torn record at the end (a crash mid-write) is ignored.
        let mut journal = std::fs::OpenOptions::new().append(true).open(Journal::path_for(&file)).unwrap();
        journal.write_all(&[1, 0, 0]).unwrap();
        drop(journal);

        let ix = Index::open_with_journal(&file).unwrap();
        assert_eq!(ix.lookup(&root.join("Docs/Taxes/Receipt.pdf")), Some(added));
        assert_eq!(ix.modified(added), 42);
        assert!(ix.lookup(&root.join("Docs/notes.md")).is_none());
        assert!(ix.lookup(&root.join("Docs/Taxes/Return 2025.pdf")).is_some());
        assert_eq!(ix.last_event_id, 99);

        // Compacting folds the journal into the file; the journal starts over.
        ix.compact_to(&file).unwrap();
        drop(ix);
        let ix = Index::open_with_journal(&file).unwrap();
        assert!(ix.added.is_empty());
        assert!(ix.lookup(&root.join("Docs/Taxes/Receipt.pdf")).is_some());
        assert!(ix.lookup(&root.join("Docs/notes.md")).is_none());

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_file(&file);
        let _ = std::fs::remove_file(Journal::path_for(&file));
    }
}
