/* hinny from C: open, write, search, close. Built and run by tests/run-c.sh. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hinny.h"

static void check(int ok, const char *what, char *error) {
    if (!ok) {
        fprintf(stderr, "FAILED: %s%s%s\n", what, error ? ": " : "", error ? error : "");
        exit(1);
    }
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "hinny-c-smoke";
    char *error = NULL;

    hinny_db *db = hinny_open(dir, &error);
    check(db != NULL, "open", error);
    hinny_namespace *docs = hinny_namespace_open(
        db, "docs", "{\"text\": [{\"name\": \"body\", \"language\": \"english\"}], \"ints\": [\"year\"], \"vector\": 4}", &error);
    check(docs != NULL, "namespace", error);

    float lease[4] = {1, 0, 0, 0}, taxes[4] = {0, 1, 0, 0};
    check(hinny_upsert(docs, "{\"key\": 1, \"text\": {\"body\": \"The lease for the flat on Elm Street\"}, \"ints\": {\"year\": 2025}}", lease, 4, &error) == 0, "upsert 1", error);
    check(hinny_upsert(docs, "{\"key\": 2, \"text\": {\"body\": \"Tax return, filed in March\"}, \"ints\": {\"year\": 2024}}", taxes, 4, &error) == 0, "upsert 2", error);

    float query[4] = {0.9f, 0.1f, 0, 0};
    char *hits = hinny_query(docs, "{\"text\": \"apartment lease\", \"filter\": {\"range\": [\"year\", 2025, null]}}", query, 4, &error);
    check(hits != NULL, "query", error);
    printf("%s\n", hits);
    check(strstr(hits, "\"key\":1") != NULL && strstr(hits, "\"key\":2") == NULL, "the lease, and only it", NULL);
    hinny_free_string(hits);

    /* A mistake comes back as a message. */
    check(hinny_query(docs, "{\"filter\": {\"equals\": [\"year\", 1]}}", NULL, 0, &error) == NULL && error != NULL, "an unknown filter is refused", NULL);
    hinny_free_string(error);
    error = NULL;

    check(hinny_count(docs, NULL, &error) == 2 && hinny_count(docs, "{\"range\": [\"year\", 2025, null]}", &error) == 1, "counts", error);
    char *years = hinny_count_by(docs, "year", NULL, 0, &error);
    check(years != NULL && strstr(years, "[2025,1]") != NULL, "counts by year", error);
    hinny_free_string(years);

    /* A patch keeps the rest; a write with a condition says whether it was made. */
    check(hinny_patch(docs, 1, "{\"ints\": {\"year\": 2026}}", NULL, &error) == 1, "patch", error);
    check(hinny_patch(docs, 1, "{\"ints\": {\"year\": 2027}}", "{\"eq\": [\"year\", 2025]}", &error) == 0, "patch refused", error);
    check(hinny_patch_where(docs, "{\"range\": [\"year\", 2025, null]}", "{\"ints\": {\"year\": 2028}}", &error) == 1, "patch_where", error);
    check(hinny_upsert_if(docs, "{\"key\": 1, \"ints\": {\"year\": 1}}", NULL, 0, "{\"eq\": [\"year\", 2025]}", &error) == 0, "upsert_if refused", error);
    check(hinny_delete_if(docs, 2, "{\"eq\": [\"year\", 2024]}", &error) == 1, "delete_if", error);
    char *typing = hinny_query(docs, "{\"text\": \"fla\", \"typing\": 32}", NULL, 0, &error);
    check(typing != NULL && strstr(typing, "\"key\":1") != NULL, "a word being typed", error);
    hinny_free_string(typing);
    check(hinny_upsert(docs, "{\"key\": 2, \"text\": {\"body\": \"Tax return, filed in March\"}, \"ints\": {\"year\": 2024}}", taxes, 4, &error) == 0, "upsert 2 again", error);

    check(hinny_sync(docs, &error) == 0, "sync", error);
    char *stats = hinny_stats(docs, &error);
    check(stats != NULL && strstr(stats, "\"documents\":2") != NULL, "stats", error);
    hinny_free_string(stats);

    hinny_namespace_close(docs);
    check(hinny_namespace_delete(db, "docs", &error) == 0, "delete the namespace", error);
    hinny_close(db);
    printf("hinny %s from C: ok\n", hinny_version());
    return 0;
}
