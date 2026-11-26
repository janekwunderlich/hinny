//! Reading directories with `getattrlistbulk`: one system call fills a buffer with many entries and
//! their attributes (name, type, modified, flags), instead of `readdir` plus a `stat` per file.
//! Directories are read on all cores (rayon's work stealing over the tree).

use std::ffi::CString;
use std::mem::size_of;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

/// `SF_DATALESS` (sys/stat.h): an iCloud or File Provider item whose contents aren't on disk.
/// Listing a dataless folder makes the provider fetch it, so the walker never descends into one.
const SF_DATALESS: u32 = 0x4000_0000;
const BUFFER: usize = 256 * 1024;

pub struct RawEntry {
    pub name: Vec<u8>,
    pub is_dir: bool,
    pub is_symlink: bool,
    /// Seconds since 1970.
    pub modified: u32,
    /// Don't descend: dataless (not local) or another volume.
    pub stop: bool,
}

/// Reads one directory. Returns false if it can't be opened (permissions, gone).
pub fn read_dir(path: &Path, out: &mut Vec<RawEntry>, buffer: &mut [u8]) -> bool {
    let Ok(c_path) = CString::new(path.as_os_str().as_bytes()) else { return false };
    let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC) };
    if fd < 0 {
        return false;
    }
    let mut attrs: libc::attrlist = unsafe { std::mem::zeroed() };
    attrs.bitmapcount = libc::ATTR_BIT_MAP_COUNT;
    attrs.commonattr = libc::ATTR_CMN_RETURNED_ATTRS | libc::ATTR_CMN_NAME | libc::ATTR_CMN_OBJTYPE | libc::ATTR_CMN_MODTIME | libc::ATTR_CMN_FLAGS;
    attrs.dirattr = libc::ATTR_DIR_MOUNTSTATUS;

    loop {
        let n = unsafe { libc::getattrlistbulk(fd, (&mut attrs as *mut libc::attrlist).cast(), buffer.as_mut_ptr().cast(), buffer.len(), 0) };
        if n <= 0 {
            unsafe { libc::close(fd) };
            return n == 0;
        }
        let mut entry = buffer.as_ptr();
        for _ in 0..n {
            unsafe {
                let length = (entry as *const u32).read_unaligned() as usize;
                let mut field = entry.add(4);
                let returned = (field as *const libc::attribute_set_t).read_unaligned();
                field = field.add(size_of::<libc::attribute_set_t>());

                let mut name = Vec::new();
                if returned.commonattr & libc::ATTR_CMN_NAME != 0 {
                    let r = (field as *const libc::attrreference_t).read_unaligned();
                    let start = field.offset(r.attr_dataoffset as isize);
                    // The length counts the terminating NUL.
                    let len = (r.attr_length as usize).saturating_sub(1);
                    name = std::slice::from_raw_parts(start, len).to_vec();
                    field = field.add(size_of::<libc::attrreference_t>());
                }
                let mut objtype = 0u32;
                if returned.commonattr & libc::ATTR_CMN_OBJTYPE != 0 {
                    objtype = (field as *const u32).read_unaligned();
                    field = field.add(4);
                }
                let mut modified = 0u32;
                if returned.commonattr & libc::ATTR_CMN_MODTIME != 0 {
                    let ts = (field as *const libc::timespec).read_unaligned();
                    modified = ts.tv_sec.clamp(0, u32::MAX as i64) as u32;
                    field = field.add(size_of::<libc::timespec>());
                }
                let mut flags = 0u32;
                if returned.commonattr & libc::ATTR_CMN_FLAGS != 0 {
                    flags = (field as *const u32).read_unaligned();
                    field = field.add(4);
                }
                let mut mount = 0u32;
                if returned.dirattr & libc::ATTR_DIR_MOUNTSTATUS != 0 {
                    mount = (field as *const u32).read_unaligned();
                }
                // vnode types (sys/vnode.h): VDIR = 2, VLNK = 5.
                let is_dir = objtype == 2;
                out.push(RawEntry {
                    name,
                    is_dir,
                    is_symlink: objtype == 5,
                    modified,
                    stop: is_dir && (flags & SF_DATALESS != 0 || mount & libc::DIR_MNTSTATUS_MNTPOINT != 0),
                });
                entry = entry.add(length);
            }
        }
    }
}

/// A directory to read, with what the sink needs to place its entries.
pub struct Job<T: Send> {
    pub path: PathBuf,
    pub data: T,
}

/// Walks the tree under `root` on all cores. For each directory read, `sink(job, entries)` places the
/// entries and returns the subdirectories to read next (it runs concurrently: it must lock what it shares).
pub fn walk<T, F>(root: Job<T>, sink: &F)
where
    T: Send,
    F: Fn(&Job<T>, Vec<RawEntry>) -> Vec<Job<T>> + Sync,
{
    rayon::scope(|scope| visit(root, sink, scope));
}

fn visit<'s, T, F>(job: Job<T>, sink: &'s F, scope: &rayon::Scope<'s>)
where
    T: Send + 's,
    F: Fn(&Job<T>, Vec<RawEntry>) -> Vec<Job<T>> + Sync,
{
    thread_local! {
        static BUFFER_TLS: std::cell::RefCell<Vec<u8>> = std::cell::RefCell::new(vec![0u8; BUFFER]);
    }
    let mut entries = Vec::new();
    // A read that fails partway still places what it got.
    BUFFER_TLS.with(|b| read_dir(&job.path, &mut entries, &mut b.borrow_mut()));
    if entries.is_empty() {
        return;
    }
    let children = sink(&job, entries);
    // The last child stays on this thread; the rest are up for stealing.
    let mut children = children.into_iter();
    let last = children.next_back();
    for child in children {
        scope.spawn(move |s| visit(child, sink, s));
    }
    if let Some(child) = last {
        visit(child, sink, scope);
    }
}
