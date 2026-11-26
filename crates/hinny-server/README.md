# hinny-server

An optional HTTP server exposing a Hinny database directory using the same namespaces, schemas, queries, and JSON payloads as the C interface ([`hinny.h`](../hinny-c/include/hinny.h) documents every JSON shape). Supports standalone primary mode and read-only replicas.

## Running it

```bash
cargo build --profile server -p hinny-server
target/server/hinny-server --dir data --listen 127.0.0.1:7878
```

Build with `--profile server` (defined in the workspace `Cargo.toml`): unlike `--release` (which aborts on panic), the `server` profile unwinds so an unexpected panic fails only the affected request rather than terminating the process.

### Flags

- `--dir`: database directory containing namespaces (created if missing).
- `--listen`: bind address and port (default `127.0.0.1:7878`).
- `--token` (or `HINNY_TOKEN`): requires `Authorization: Bearer <token>` on every route except `/healthz`. Without a token, the server refuses non-loopback bind addresses.
- `--max-body-mb`: maximum HTTP request body size in MiB (default `256`).
- `--replica-interval`: how long a namespace's replica file listing is cached before flushing memory again (default `1s`; see [Replicas](#replicas)).

`SIGINT` or `SIGTERM` triggers graceful shutdown: in-flight requests finish, then every open namespace checkpoints and closes cleanly.

## Routes

All request and response bodies are JSON. `{ns}` allows letters, digits, `-`, and `_`.

| Route | Request body | Response |
| :--- | :--- | :--- |
| `GET /healthz` | — | `{"ok": true}` |
| `GET /v1/namespaces` | — | `{"namespaces": [names]}` |
| `PUT /v1/namespaces/{ns}` | schema | `{"name", "schema"}` (opens or creates) |
| `GET /v1/namespaces/{ns}` | — | `{"name", "schema", "stats"}` |
| `DELETE /v1/namespaces/{ns}` | — | `{"deleted": name}` (`423` if open in another process) |
| `POST /v1/namespaces/{ns}/apply` | `[documents and {"delete": key}]` | `{"changes": n}` (one atomic batch) |
| `GET /v1/namespaces/{ns}/documents/{key}` | — | `{"attributes", "text"}` |
| `DELETE /v1/namespaces/{ns}/documents/{key}` | — | `{"ok": true}` |
| `POST /v1/namespaces/{ns}/query` | query object | `{"hits": [...]}` (best first) |
| `POST /v1/namespaces/{ns}/count` | `{"filter"?}` | `{"count": n}` |
| `POST /v1/namespaces/{ns}/count_by` | `{"attribute", "filter"?, "limit"?}` | `{"counts": [[value, n]]}` |
| `POST /v1/namespaces/{ns}/delete_where` | `{"filter"}` | `{"deleted": n}` |
| `POST /v1/namespaces/{ns}/patch` | `{"key", "patch", "if"?}` | `{"patched": true\|false}` |
| `POST /v1/namespaces/{ns}/patch_where` | `{"filter", "patch"}` | `{"patched": n}` |
| `POST /v1/namespaces/{ns}/upsert_if` | `{"document", "if"}` | `{"written": true\|false}` |
| `POST /v1/namespaces/{ns}/delete_if` | `{"key", "if"}` | `{"deleted": true\|false}` |
| `POST /v1/namespaces/{ns}/fork` | `{"name"}` | `{"name", "schema"}` |
| `POST /v1/namespaces/{ns}/flush`, `/optimize`, `/sync` | — | `{"ok": true}` |
| `GET /v1/namespaces/{ns}/replica` | — | `{"state", "id", "schema", "manifest", "files": [{"name", "size"}]}` |
| `GET /v1/namespaces/{ns}/replica/files/{name}` | — | raw file bytes with `Content-Length` |

### Payload notes

- **Schemas:** Namespaces open lazily on first use. Sending `PUT` to an existing namespace validates compatibility and grows the schema where allowed (adding text fields, attributes, or a vector if absent). `"vector": 384` uses cosine similarity; `"vector": {"dimensions": 384, "metric": "euclidean" | "dot" | "cosine"}` selects a metric (immutable once set; mismatch returns `400`).
- **Patches & conditional writes:** A patch accepts any document fields except `key` (`version`, `text`, `sparse`, `ints`, `floats`, `strings`, `tags`, `vector`/`vector_b64`) plus `"remove": [names]`. Because sparse term maps are not stored raw, patching a namespace with sparse fields must either resupply `"sparse"` or include the field in `"remove"` (otherwise `400`). `"if"` takes any filter evaluated atomically with the write under the writer lock. Patching reads and rewrites the document (requiring stored text).
- **Search-as-you-type:** Set `"typing": 32` on a query to treat the final word as an incomplete prefix and expand it to the 32 most frequent matching terms.
- **Compact vectors:** Pass `"vector": [0.1, 0.9, ...]` or `"vector_b64": "..."` (base64-encoded little-endian `f32` bytes) in both documents and queries—roughly a quarter the payload size with no float parsing.

### Examples

```bash
H='Authorization: Bearer secret'   # with --token secret
curl -X PUT localhost:7878/v1/namespaces/docs -H "$H" \
  -d '{"text": [{"name": "body", "language": "english"}], "ints": ["year"], "tags": ["labels"], "vector": 3}'

curl -X POST localhost:7878/v1/namespaces/docs/apply -H "$H" -d '[
  {"key": 1, "text": {"body": "The lease for the flat"}, "ints": {"year": 2025},
   "tags": {"labels": ["home", "contract"]}, "vector": [0.1, 0.9, 0.0]},
  {"delete": 7}]'

curl -X POST localhost:7878/v1/namespaces/docs/query -H "$H" \
  -d '{"text": "apartment lease", "vector": [0.1, 0.8, 0.1], "filter": {"eq": ["labels", "home"]}, "limit": 10}'

curl localhost:7878/v1/namespaces/docs/documents/1 -H "$H"

curl -X POST localhost:7878/v1/namespaces/docs/count_by -H "$H" -d '{"attribute": "labels", "limit": 10}'

curl -X POST localhost:7878/v1/namespaces/docs/delete_where -H "$H" -d '{"filter": {"range": ["year", null, 2020]}}'

curl -X POST localhost:7878/v1/namespaces/docs/fork -H "$H" -d '{"name": "docs-trial"}'

curl -X POST localhost:7878/v1/namespaces/docs/patch -H "$H" \
  -d '{"key": 1, "patch": {"tags": {"labels": ["home", "signed"]}, "remove": ["year"]}, "if": {"eq": ["labels", "home"]}}'

curl -X POST localhost:7878/v1/namespaces/docs/upsert_if -H "$H" \
  -d '{"document": {"key": 1, "ints": {"year": 2026}}, "if": {"range": ["year", null, 2025]}}'

curl -X POST localhost:7878/v1/namespaces/docs/query -H "$H" -d '{"text": "apartment le", "typing": 32}'
```

## Errors

```json
{"error": {"code": "not_found", "message": "no namespace \"docs\""}}
```

| Status | Code | Cause |
| :--- | :--- | :--- |
| `400` | `invalid` | Malformed JSON or invalid schema, document, query, or option |
| `401` | `unauthorized` | Missing or invalid bearer token |
| `404` | `not_found` | Unknown namespace, document, route, or compacted replica file |
| `405` | `read_only` | Write request sent to a read replica |
| `409` | `exists` | Fork target name already exists |
| `413` | `too_large` | Request body exceeds `--max-body-mb` |
| `423` | `locked` | Namespace is locked by another writer or busy past 10 s |
| `504` | `timeout` | Query exceeded `timeout_ms` |
| `500` | `io` | Storage or engine I/O error |

## Replicas

A replica mirrors a primary server's namespaces into its own local directory and answers reads locally, keeping writes routed to the single primary.

```bash
# Primary
target/server/hinny-server --dir data --listen 10.0.0.5:7878 --token secret

# Read replica
target/server/hinny-server --dir copy --listen 127.0.0.1:7879 \
  --replica-of http://10.0.0.5:7878 --primary-token secret --sync-every 1s
```

- `--replica-of`: primary URL (`http://` or `https://`).
- `--primary-token` (or `HINNY_PRIMARY_TOKEN`): bearer token for authenticating to the primary (`--token` remains the replica's own client token).
- `--sync-every`: polling interval across namespaces (default `1s`).
- `--dir`: local mirror directory (non-replica namespaces placed here are left untouched and skipped).

**Served routes:** Read routes (`GET /v1/namespaces`, `GET /v1/namespaces/{ns}`, document fetches, `query`, `count`, `count_by`). All mutating routes—including `flush`, `optimize`, `sync`, and replica endpoints—return `405 read_only`.

**How syncing works:** On startup and every `--sync-every`, the replica fetches each namespace's file listing from the primary. When the state hash changes, it downloads only missing files (segments, deletion files, and text blobs are immutable once written), writes the schema, atomically replaces the manifest last, and prunes obsolete files. New namespaces appear on the next cycle; deleted ones are removed cleanly once in-flight queries finish.

**Freshness & primary overhead:** Acknowledged writes reach replicas in roughly `--replica-interval` + `--sync-every` + transfer time (~2 s with defaults). Each namespace is internally consistent to one primary snapshot, though separate namespaces may sync a moment apart. If the primary is unreachable, the replica continues serving its latest local snapshot. On the primary, generating a listing flushes unflushed memtable data into a segment; caching listings for `--replica-interval` caps this to at most one small segment per namespace per interval.

**Interruption safety:** Files download to `<name>.part` and rename only when complete, and the manifest updates last. An interrupted sync leaves the previous snapshot intact and resumes missing files on the next cycle. If a listed file is merged away before download, the replica retries from a fresh listing.

To embed local replica syncing directly inside a Rust application without running a second HTTP server, use [`hinny-replica`](../hinny-replica/README.md).

## Current limitations

- **Single writer:** One primary process owns writes per database directory; horizontal sharding and object storage are not implemented.
- **Polling replication:** Replicas poll on `--sync-every` rather than receiving push notifications.
- **Open namespaces:** Namespaces stay open in memory until server shutdown.
- **No built-in TLS:** Speaks plain HTTP; terminate TLS at a reverse proxy (Caddy, nginx) outside loopback.
- **Single bearer token:** One shared token grants access to all namespaces.
