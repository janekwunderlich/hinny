# hinny-names

macOS file-name index: indexes the names and metadata of every file and folder under a directory in a memory-mapped column store. Searches 9M+ entries as you type (~0.1 ms median per keystroke, ~2 ms full scan) and stays current within ~50 ms via FSEvents. Also exposes a C/Swift static library through `include/hinny_names.h`.

```bash
cargo build --release --offline
target/release/hinny-names build                         # indexes ~ → ~/Library/Application Support/Hinny/names.idx
target/release/hinny-names search "lease" --paths         # top matches (omit --paths for kinds and scores only)
target/release/hinny-names bench --freshness              # keystroke latency and FSEvents freshness
target/release/hinny-names watch                          # apply changes live; journals every 5 s
target/release/hinny-names cpu                            # CPU time per search (use RAYON_NUM_THREADS to compare)
target/release/hinny-names compare                        # baseline scan vs. search-as-you-type latency and agreement
target/release/hinny-names starts                         # rebuild word-start lists (also done on build/compact)
target/release/hinny-names compact                        # time a full compaction rewrite
HINNY_NAMES_POWER=battery target/release/hinny-names bench  # force battery profile regardless of power source
```

## Architecture

- `src/walk.rs`: parallel `getattrlistbulk` traversal; skips dataless iCloud folders and other mounted volumes.
- `src/store.rs`: single memory-mapped columnar file with in-memory overlays for live additions, deletions, and timestamp updates.
- `src/search.rs`: SIMD substring search (`memchr::memmem`) over case-folded names and ranking (baseline Rayon engine kept for comparison).
- `src/fast.rs`: search-as-you-type engine—deduplicates repeat queries, uses word-start lists (`src/starts.rs`), narrows previous keystroke results, or falls back to a GCD worker pool that parks between keystrokes (`src/pool.rs`).
- `src/watch.rs`: FSEvents listener resuming from the persisted event ID.
- `src/power.rs`: IOKit power-source detection adjusting thread counts and QoS between AC and battery.
- `src/ffi.rs`: C interface backing `include/hinny_names.h`.
- **Persistence:** Live changes append to `names.idx.journal` (replayed on open). The ~700 MB base file compacts only when the journal exceeds 2% of entries or 64 MB, at most once per hour while on AC power (forced at 256 MB).
