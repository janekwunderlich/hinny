# hinny-c

The C ABI for Hinny, built for FFI from Swift, Kotlin, C, and any language that calls C. [`include/hinny.h`](include/hinny.h) defines the entire interface: database and namespace handles, one function per operation, JSON strings for schemas/documents/queries/results, and raw `float` arrays for vectors.

```c
char *error = NULL;
hinny_db *db = hinny_open("data", &error);
hinny_namespace *docs = hinny_namespace_open(db, "docs",
    "{\"text\": [{\"name\": \"body\"}], \"ints\": [\"year\"], \"vector\": 384}", &error);

hinny_upsert(docs, "{\"key\": 1, \"text\": {\"body\": \"The lease for the flat\"},"
                   " \"ints\": {\"year\": 2025}}", embedding, 384, &error);

char *hits = hinny_query(docs, "{\"text\": \"apartment lease\","
                   " \"filter\": {\"range\": [\"year\", 2025, null]}}", query, 384, &error);
/* [{"key":1,"score":…,"attributes":{"year":2025},"text":{"body":"…"},"snippet":{…}}] */
hinny_free_string(hits);

int64_t recent = hinny_count(docs, "{\"range\": [\"year\", 2025, null]}", &error);
char *years = hinny_count_by(docs, "year", NULL, 10, &error);   /* [[2025,1]] */
hinny_free_string(years);
```

## Building

```bash
cargo build --release -p hinny-c
# target/release/libhinny_c.a   static library (for embedding into native apps)
# target/release/libhinny_c.so  shared library (.dylib on macOS)
```

Link the static library with `-lpthread -ldl -lm` on Linux, or `-framework Accelerate` on macOS. Run `sh crates/hinny-c/tests/run-c.sh` to build and execute the C smoke test.

## Status

Python ([`bindings/python`](../../bindings/python)) and Node.js ([`bindings/node`](../../bindings/node)) have dedicated native bindings. Thin Swift and Kotlin packages over `hinny.h` do not exist yet, and iOS/Android cross-builds are untested.
