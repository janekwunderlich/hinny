// The Hinny name index (Rust): every file and folder name under a folder,
// searched as you type. Built as libhinny_names.a; see src/ffi.rs.
#ifndef HINNY_NAMES_H
#define HINNY_NAMES_H

#include <stdbool.h>
#include <stdint.h>

typedef struct HinnyNames HinnyNames;

typedef struct {
    uint32_t entry;
    float score;
    /// Bits: 1 folder, 2 hidden, 4 in ~/Library, 8 in a code project, 16 inside a bundle,
    /// 32 application, 64 document, 128 code or build file.
    uint8_t traits;
    /// Seconds since 1970.
    uint32_t modified;
} HinnyNameHit;

/// Opens the index at `path` (replaying its journal); NULL if missing or unreadable.
HinnyNames *hinny_names_open(const char *path);
/// Builds an index of everything under `root` at `path` (blocking). Entries, or -1.
int64_t hinny_names_build(const char *root, const char *path);
uint64_t hinny_names_count(const HinnyNames *service);
/// The best `capacity` entries for `query` into `out`; returns how many.
uint32_t hinny_names_search(const HinnyNames *service, const char *query, HinnyNameHit *out, uint32_t capacity);
/// The entry's full path into `buffer` (NUL-terminated); its length, or 0.
uint32_t hinny_names_path(const HinnyNames *service, uint32_t entry, char *buffer, uint32_t capacity);
/// Follows file system changes (and catches up since the last save). False if FSEvents failed.
bool hinny_names_watch(const HinnyNames *service);
/// Saves pending changes and closes.
void hinny_names_close(HinnyNames *service);

#endif
