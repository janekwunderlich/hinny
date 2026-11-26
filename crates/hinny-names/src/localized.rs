//! Finder's localized folder names ("Dokumente" for ~/Documents on a German Mac): a folder holding
//! a `.localized` file shows a translated name that isn't its name on disk. Those names are stored
//! as aliases next to the index (`names.idx.aliases`), so typing the name you see finds the folder.

use std::ffi::{c_char, c_void};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFURLCreateFromFileSystemRepresentation(alloc: *const c_void, buffer: *const u8, length: isize, is_directory: u8) -> *const c_void;
    fn CFURLCopyResourcePropertyForKey(url: *const c_void, key: *const c_void, value: *mut *const c_void, error: *mut *const c_void) -> u8;
    fn CFStringGetCString(s: *const c_void, buffer: *mut c_char, size: isize, encoding: u32) -> u8;
    fn CFRelease(value: *const c_void);
    static kCFURLLocalizedNameKey: *const c_void;
}

/// The name Finder shows for `path`, if it differs from the name on disk.
pub fn display_name(path: &Path) -> Option<String> {
    let bytes = path.as_os_str().as_bytes();
    unsafe {
        let url = CFURLCreateFromFileSystemRepresentation(std::ptr::null(), bytes.as_ptr(), bytes.len() as isize, 1);
        if url.is_null() {
            return None;
        }
        let mut value: *const c_void = std::ptr::null();
        let ok = CFURLCopyResourcePropertyForKey(url, kCFURLLocalizedNameKey, &mut value, std::ptr::null_mut()) != 0;
        CFRelease(url);
        if !ok || value.is_null() {
            return None;
        }
        let mut buf = [0 as c_char; 1024];
        let got = CFStringGetCString(value, buf.as_mut_ptr(), buf.len() as isize, 0x0800_0100) != 0;
        CFRelease(value);
        let name = got.then(|| std::ffi::CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned())?;
        let on_disk = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        (name != on_disk).then_some(name)
    }
}
