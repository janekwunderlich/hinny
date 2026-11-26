/*
 * hinny: an embedded hybrid search database. The C interface.
 *
 * Documents, queries and results are JSON (UTF-8, NUL-terminated); vectors are plain arrays of
 * float. Every handle may be used from several threads at once. One process writes a namespace
 * at a time.
 *
 * Errors: a function that can fail takes `char **error`. On failure it returns NULL (pointers)
 * or -1 (int) and, if `error` isn't NULL, stores a message there, to be freed with
 * hinny_free_string. On success `*error` is left as it was.
 *
 * Strings a function returns belong to the caller: free them with hinny_free_string.
 *
 *   hinny_db *db = hinny_open("data", &err);
 *   hinny_namespace *docs = hinny_namespace_open(db, "docs",
 *       "{\"text\": [{\"name\": \"body\"}], \"ints\": [\"year\"], \"vector\": 384}", &err);
 *   hinny_upsert(docs, "{\"key\": 1, \"text\": {\"body\": \"The lease for the flat\"}, "
 *                      "\"ints\": {\"year\": 2025}}", embedding, 384, &err);
 *   char *hits = hinny_query(docs, "{\"text\": \"apartment lease\", "
 *                      "\"filter\": {\"range\": [\"year\", 2025, null]}}", query, 384, &err);
 */
#ifndef HINNY_H
#define HINNY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hinny_db hinny_db;
typedef struct hinny_namespace hinny_namespace;

/* The library's version, "0.3.0". Not to be freed. */
const char *hinny_version(void);

/* Frees a string a function here returned (or stored in `error`). NULL is fine. */
void hinny_free_string(char *string);

/* A stable key for a string id (FNV-1a, 63 bits), for documents named by strings. */
uint64_t hinny_key(const char *id);

/* Opens (creating it if need be) the database in a directory: a folder of namespaces. */
hinny_db *hinny_open(const char *directory, char **error);

/* Closes the database handle. Namespaces opened from it stay usable until they're closed. */
void hinny_close(hinny_db *db);

/* The names of the database's namespaces, a JSON array of strings. */
char *hinny_namespaces(hinny_db *db, char **error);

/*
 * Opens a namespace, creating it with this schema or checking the schema against the one it has
 * (fields and attributes may be added; kinds and vector dimensions can't change). The schema:
 *
 *   {"text":    [{"name": "title", "boost": 2.0}, {"name": "body", "language": "english"}],
 *    "sparse":  ["terms"],            learned-sparse fields: weighted terms, not text
 *    "ints":    ["year"],             64-bit integer attributes
 *    "floats":  ["score"],            64-bit floating-point attributes
 *    "strings": ["kind"],             string attributes
 *    "tags":    ["labels"],           attributes of several strings (a set)
 *    "vector":  384,                  embedding dimensions (cosine similarity)
 *    "store_text": true}              keep the text for results and snippets (the default)
 *
 * Every part is optional. "boost" defaults to 1; "language" (english, german, french, ...) stems
 * words and drops its stopwords, and without it words are matched as written, lowercased.
 *
 * Vectors are compared by cosine similarity unless the schema names another metric, the one the
 * model that made them was trained for:
 *
 *    "vector": {"dimensions": 384, "metric": "euclidean"}     or "dot", or "cosine"
 *
 * A namespace's metric can't change once it is set: opening it with another is an error. A hit's
 * vector score is higher for a nearer vector under each: the cosine for "cosine" (vectors and
 * queries are normalized), the squared distance negated for "euclidean" (0 for the query itself),
 * the dot product for "dot". Under "euclidean" and "dot" vectors are kept as they are given, not
 * normalized, in half precision.
 *
 * With a NULL schema, opens a namespace that exists as it is (an error if there is none).
 */
hinny_namespace *hinny_namespace_open(hinny_db *db, const char *name, const char *schema, char **error);

/*
 * Deletes a namespace and everything in it. It must not be open: close every handle of it first
 * (hinny_namespace_close), or this fails and leaves it as it is. A fork is a namespace of its
 * own, deleted apart from the one it was made from. An error too if there is no namespace of
 * that name. Returns 0 on success.
 */
int hinny_namespace_delete(hinny_db *db, const char *name, char **error);

/* Closes a namespace handle (writes so far are in the log; hinny_sync makes them durable). */
void hinny_namespace_close(hinny_namespace *ns);

/*
 * Adds a document, or replaces the one with its key:
 *
 *   {"key": 1,                          a number, or a string of digits (keys are 64 bits)
 *    "version": 7,                      optional, kept with the document
 *    "text":    {"body": "..."},
 *    "sparse":  {"terms": {"lease": 1.4, "flat": 0.9}},
 *    "ints":    {"year": 2025},
 *    "floats":  {"score": 0.75},
 *    "strings": {"kind": "pdf"},
 *    "tags":    {"labels": ["tax", "2025"]},
 *    "vector":  [0.1, 0.9, 0.0]}      optional, instead of the `vector` argument (see below)
 *
 * `vector` is the document's embedding, `dimensions` floats (NULL and 0 for none). 0 on success.
 * A document may carry its embedding inside instead, as "vector" (numbers) or "vector_b64"
 * (base64 of the little-endian 32-bit floats: a quarter of the bytes of the same numbers in
 * JSON, and no number parsing): how documents in a batch (hinny_apply) have theirs. Inside or
 * as the argument, not both: a document with an embedding inside and a `vector` argument too is
 * an error.
 */
int hinny_upsert(hinny_namespace *ns, const char *document, const float *vector, size_t dimensions, char **error);

/*
 * Applies a batch as one change: all of it or none, and readers see it at once. A JSON array of
 * documents as for hinny_upsert, each with an optional "vector": [..], and of {"delete": key}.
 */
int hinny_apply(hinny_namespace *ns, const char *changes, char **error);

/* Deletes the document with a key (nothing happens if there is none). */
int hinny_delete(hinny_namespace *ns, uint64_t key, char **error);

/*
 * Changes part of the document with a key and leaves the rest: the text fields, attributes and
 * vector the patch names are replaced, those it lists in "remove" are taken away, and the
 * document's other fields and its version stay as they are. The patch has a document's parts
 * but the key, and "remove":
 *
 *   {"strings": {"state": "final"},
 *    "text":    {"body": "..."},
 *    "ints":    {"year": 2026},       also "floats", "tags", "vector" or "vector_b64"
 *    "sparse":  {"terms": {"lease": 1.4}},
 *    "version": 8,                    optional: a new version
 *    "remove":  ["owner", "note"]}    attributes or text fields to take away
 *
 * A sparse field's terms aren't kept as they were given, so in a namespace with sparse fields a
 * patch must give each one's terms again in "sparse" or name the field in "remove": one that does
 * neither is an error.
 *
 * `condition` is a filter (as in a query) the document must pass for the patch to be made, or
 * NULL for none: checked and written together, so nothing comes between. The document is read
 * and written again whole, so a patch costs what an upsert does, and it needs the text to have
 * been stored ("store_text", the default). Returns 1 if the document was changed, 0 if there
 * is no document with that key or it didn't pass the condition (nothing is written then), -1 on
 * an error.
 */
int hinny_patch(hinny_namespace *ns, uint64_t key, const char *patch, const char *condition, char **error);

/*
 * hinny_patch for every document that passes a filter (as in a query), in batches of 10,000
 * documents each all or nothing. Returns how many documents passed, -1 on an error.
 */
int64_t hinny_patch_where(hinny_namespace *ns, const char *filter, const char *patch, char **error);

/*
 * hinny_upsert, unless there is a document with this key that doesn't pass `condition` (a
 * filter, as in a query): written if there is none yet or if the one there passes, say a
 * revision below the new one. Checked and written together, so two writers that both read
 * revision 4 can't both write revision 5. Returns 1 if it was written, 0 if not, -1 on an error.
 */
int hinny_upsert_if(hinny_namespace *ns, const char *document, const float *vector, size_t dimensions, const char *condition, char **error);

/*
 * hinny_delete, only if the document with that key exists and passes `condition` (a filter, as
 * in a query). Returns 1 if it was deleted, 0 if not, -1 on an error.
 */
int hinny_delete_if(hinny_namespace *ns, uint64_t key, const char *condition, char **error);

/*
 * A document's attributes and stored text, {"attributes": {..}, "text": {..}}, or NULL without
 * an error if there is no document with that key.
 */
char *hinny_get(hinny_namespace *ns, uint64_t key, char **error);

/*
 * Searches. Any of text, vector, sparse terms and a filter, together or alone; with more than
 * one retriever the results are fused into one ranking.
 *
 *   {"text": "apartment lease",
 *    "sparse": {"terms": {"lease": 1.0}},
 *    "filter": {"and": [{"eq": ["kind", "pdf"]}, {"range": ["year", 2020, null]}]},
 *    "limit": 10,                     default 10
 *    "all_words": false,              every word must match
 *    "typing": 32,                    the last word is still being typed: it also matches the
 *                                     words that start with it, the 32 most common ("tax ret"
 *                                     finds "tax return"); 0 or absent for a finished word
 *    "order_by": {"attribute": "year", "descending": true},   instead of by score
 *    "fusion": "convex",              or "rrf"
 *    "text_weight": 0.5,              the text's share in a convex fusion
 *    "with_text": true, "snippet": 240,
 *    "nprobe": 0,                     clusters to probe; 0 for the calibrated default
 *    "timeout_ms": 50}                give up (with an error) after this long
 *
 * Filters: {"eq": [name, value]}, {"in": [name, [values]]}, {"range": [name, min, max]} (numbers,
 * inclusive, null for an open end), {"prefix": [name, "..."]}, {"contains": [name, "..."]},
 * {"glob": [name, "pattern"]}, {"iglob": [name, "pattern"]} (the whole string matches the pattern:
 * "*" is any run of characters, "?" any one, a backslash takes the next as itself; iglob ignores
 * case),
 * {"exists": name}, {"and": [..]}, {"or": [..]}, {"not": {..}}. On tags a condition holds if it
 * holds for one of them: {"eq": ["labels", "tax"]} is "has the tag", {"in": [..]} "has one of".
 *
 * `vector` is the query's embedding, `dimensions` floats (NULL and 0 for none).
 *
 * Returns a JSON array, best first:
 *
 *   [{"key": 1, "score": 0.83, "text_score": 4.1, "vector_score": 0.77,
 *     "attributes": {"year": 2025, "kind": "pdf"}, "text": {"body": "..."},
 *     "snippet": {"field": "body", "text": "...", "highlights": [[4, 9]]}}]
 *
 * Keys are JSON numbers up to 2^64 - 1: read them as 64-bit integers, not as doubles.
 */
char *hinny_query(hinny_namespace *ns, const char *query, const float *vector, size_t dimensions, char **error);

/*
 * Deletes every document that passes a filter (as in a query): a user's, the expired. All of
 * them or, on an error, none. Returns how many were deleted, -1 on an error.
 */
int64_t hinny_delete_where(hinny_namespace *ns, const char *filter, char **error);

/*
 * How many documents pass a filter (as in a query; NULL for every document), counted from the
 * filter's columns without reading the documents. -1 on an error.
 */
int64_t hinny_count(hinny_namespace *ns, const char *filter, char **error);

/*
 * How many documents have each value of an attribute, among those that pass a filter (NULL for
 * every document): a JSON array of [value, count], the most frequent first, at most `limit`
 * (0: every value).
 *
 *   [["pdf", 2], ["jpg", 1]]
 *
 * For attributes of few values (a kind, an owner, a status): it holds a count per value.
 */
char *hinny_count_by(hinny_namespace *ns, const char *attribute, const char *filter, size_t limit, char **error);

/* Makes every write so far durable (an fsync of the log). */
int hinny_sync(hinny_namespace *ns, char **error);

/* Writes what's in memory out as a segment (this also happens on its own). */
int hinny_flush(hinny_namespace *ns, char **error);

/*
 * A namespace of its own, named `name` in the same database, holding what `ns` holds now: a
 * sandbox to try something on real data. The two are apart from then on, and either can be
 * deleted. Takes milliseconds whatever the size, and no more disk until they differ: its files
 * are links to this one's. NULL and an error if there is a namespace of that name. Close the
 * handle with hinny_namespace_close.
 */
hinny_namespace *hinny_namespace_fork(hinny_namespace *ns, const char *name, char **error);

/* Merges everything into one segment: the fastest layout to search, after a bulk load. */
int hinny_optimize(hinny_namespace *ns, char **error);

/* Counts, as JSON: {"documents": .., "segments": .., "in_memory": .., "merging": .., ...}. */
char *hinny_stats(hinny_namespace *ns, char **error);

/*
 * Reads every file against its checksums: {"segments": .., "unchecked_segments": ..,
 * "text_files": ..}, or an error naming the first damaged file.
 */
char *hinny_verify(hinny_namespace *ns, char **error);

#ifdef __cplusplus
}
#endif

#endif /* HINNY_H */
