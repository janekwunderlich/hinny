//! The C interface, called the way C calls it: C strings in, C strings out, errors through a
//! pointer. (tests/smoke.c does the same from C itself; tests/run-c.sh builds and runs it.)

use std::ffi::{CStr, CString, c_char};

use hinny_c::*;

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// A returned string as JSON, freed.
unsafe fn take(p: *mut c_char) -> serde_json::Value {
    assert!(!p.is_null());
    let v = serde_json::from_str(unsafe { CStr::from_ptr(p) }.to_str().unwrap()).unwrap();
    unsafe { hinny_free_string(p) };
    v
}

/// The message in `error`, freed and reset.
unsafe fn message(error: &mut *mut c_char) -> String {
    assert!(!error.is_null(), "an error was expected");
    let m = unsafe { CStr::from_ptr(*error) }.to_str().unwrap().to_string();
    unsafe { hinny_free_string(*error) };
    *error = std::ptr::null_mut();
    m
}

fn dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("hinny-c-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

#[test]
fn a_namespace_from_open_to_close() {
    let d = dir("whole");
    let mut error: *mut c_char = std::ptr::null_mut();
    unsafe {
        let db = hinny_open(c(d.to_str().unwrap()).as_ptr(), &mut error);
        assert!(!db.is_null() && error.is_null());
        let schema = c(r#"{"text": [{"name": "title", "boost": 2.0}, {"name": "body", "language": "english"}], "ints": ["year"], "strings": ["kind"], "vector": 3}"#);
        let ns = hinny_namespace_open(db, c("docs").as_ptr(), schema.as_ptr(), &mut error);
        assert!(!ns.is_null() && error.is_null());
        assert_eq!(take(hinny_namespaces(db, &mut error)), serde_json::json!(["docs"]));

        // One document with its vector beside it, then a batch with vectors inside.
        let doc = c(r#"{"key": 1, "version": 7, "text": {"title": "Lease", "body": "The lease for the flat on Elm Street"}, "ints": {"year": 2025}, "strings": {"kind": "pdf"}}"#);
        assert_eq!(hinny_upsert(ns, doc.as_ptr(), [1.0f32, 0.0, 0.0].as_ptr(), 3, &mut error), 0);
        let batch = c(r#"[
            {"key": "2", "text": {"title": "Tax return", "body": "Filed in March, running late"}, "ints": {"year": 2024}, "strings": {"kind": "pdf"}, "vector": [0.0, 1.0, 0.0]},
            {"key": 3, "text": {"title": "Notes", "body": "Leases and other contracts"}, "ints": {"year": 2023}, "strings": {"kind": "note"}, "vector": [0.9, 0.1, 0.0]},
            {"key": 4, "text": {"body": "to be deleted"}},
            {"delete": 4}
        ]"#);
        assert_eq!(hinny_apply(ns, batch.as_ptr(), &mut error), 0);
        assert!(error.is_null());
        assert_eq!(take(hinny_stats(ns, &mut error))["documents"], 3);

        // Text alone (stemmed: "leases" finds "lease"), best first, with a snippet.
        let hits = take(hinny_query(ns, c(r#"{"text": "leases"}"#).as_ptr(), std::ptr::null(), 0, &mut error));
        let keys: Vec<u64> = hits.as_array().unwrap().iter().map(|h| h["key"].as_u64().unwrap()).collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&1) && keys.contains(&3));
        assert!(hits[0]["snippet"]["text"].as_str().unwrap().to_lowercase().contains("lease"));
        assert_eq!(hits[0]["attributes"]["kind"].as_str().is_some(), true);

        // Hybrid with a filter: one ranking, only what the filter keeps.
        let q = c(r#"{"text": "lease", "filter": {"and": [{"eq": ["kind", "pdf"]}, {"range": ["year", 2025, null]}]}, "limit": 5}"#);
        let hits = take(hinny_query(ns, q.as_ptr(), [1.0f32, 0.0, 0.0].as_ptr(), 3, &mut error));
        assert_eq!(hits.as_array().unwrap().len(), 1);
        assert_eq!(hits[0]["key"], 1);
        assert!(hits[0]["text_score"].is_number() && hits[0]["vector_score"].is_number());

        // A vector alone, ordered by cosine; and by an attribute instead of by score.
        let hits = take(hinny_query(ns, c(r#"{"limit": 2, "with_text": false}"#).as_ptr(), [0.0f32, 1.0, 0.0].as_ptr(), 3, &mut error));
        assert_eq!(hits[0]["key"], 2);
        assert!(hits[0].get("text").is_none());
        let hits = take(hinny_query(ns, c(r#"{"filter": {"exists": "year"}, "order_by": {"attribute": "year", "descending": true}}"#).as_ptr(), std::ptr::null(), 0, &mut error));
        let years: Vec<i64> = hits.as_array().unwrap().iter().map(|h| h["attributes"]["year"].as_i64().unwrap()).collect();
        assert_eq!(years, [2025, 2024, 2023]);

        // Counts: of everything, of what a filter passes, and by an attribute's values.
        assert_eq!(hinny_count(ns, std::ptr::null(), &mut error), 3);
        assert_eq!(hinny_count(ns, c(r#"{"eq": ["kind", "pdf"]}"#).as_ptr(), &mut error), 2);
        let kinds = take(hinny_count_by(ns, c("kind").as_ptr(), std::ptr::null(), 0, &mut error));
        assert_eq!(kinds[0], serde_json::json!(["pdf", 2]));
        assert_eq!(kinds.as_array().unwrap().len(), 2);
        let years = take(hinny_count_by(ns, c("year").as_ptr(), c(r#"{"eq": ["kind", "pdf"]}"#).as_ptr(), 1, &mut error));
        assert_eq!(years.as_array().unwrap().len(), 1);
        assert!(years[0][0].is_i64() && years[0][1] == 1);
        assert!(error.is_null());
        assert_eq!(hinny_count(ns, c(r#"{"equals": ["kind", "pdf"]}"#).as_ptr(), &mut error), -1);
        assert!(message(&mut error).contains("no filter"));

        // Floats: in the schema, in documents, in results and in ranges (a fraction or not).
        let scored = hinny_namespace_open(db, c("scored").as_ptr(), c(r#"{"floats": ["score"], "ints": ["n"]}"#).as_ptr(), &mut error);
        assert!(!scored.is_null() && error.is_null());
        let batch = c(r#"[{"key": 1, "floats": {"score": 0.25}, "ints": {"n": 1}}, {"key": 2, "floats": {"score": 2}, "ints": {"n": 2}}, {"key": 3, "floats": {"score": -1.5}, "ints": {"n": 3}}]"#);
        assert_eq!(hinny_apply(scored, batch.as_ptr(), &mut error), 0);
        assert_eq!(take(hinny_get(scored, 1, &mut error))["attributes"]["score"], 0.25);
        let keys = |q: &str| -> Vec<u64> {
            let mut error: *mut c_char = std::ptr::null_mut();
            let hits = take(hinny_query(scored, c(q).as_ptr(), std::ptr::null(), 0, &mut error));
            assert!(error.is_null());
            hits.as_array().unwrap().iter().map(|h| h["key"].as_u64().unwrap()).collect()
        };
        assert_eq!(keys(r#"{"filter": {"range": ["score", 0.1, null]}, "order_by": {"attribute": "score", "descending": true}}"#), [2, 1]);
        assert_eq!(keys(r#"{"filter": {"range": ["score", null, 0]}}"#), [3]);
        assert_eq!(keys(r#"{"filter": {"eq": ["score", 2]}}"#), [2]);
        assert_eq!(keys(r#"{"filter": {"range": ["n", 1.5, 2.5]}}"#), [2]);
        assert_eq!(take(hinny_count_by(scored, c("score").as_ptr(), std::ptr::null(), 0, &mut error)).as_array().unwrap().len(), 3);
        assert!(error.is_null());
        hinny_namespace_close(scored);

        // Tags: several strings an attribute, a condition on one of them, counted per tag.
        let tagged = hinny_namespace_open(db, c("tagged").as_ptr(), c(r#"{"tags": ["labels"], "strings": ["owner"]}"#).as_ptr(), &mut error);
        assert!(!tagged.is_null() && error.is_null());
        let batch = c(r#"[{"key": 1, "tags": {"labels": ["tax", "2025", "tax"]}, "strings": {"owner": "mia"}}, {"key": 2, "tags": {"labels": ["tax", "inbox"]}}, {"key": 3, "tags": {"labels": []}}]"#);
        assert_eq!(hinny_apply(tagged, batch.as_ptr(), &mut error), 0);
        assert_eq!(take(hinny_get(tagged, 1, &mut error))["attributes"]["labels"], serde_json::json!(["2025", "tax"]));
        let keys = |q: &str| -> Vec<u64> {
            let mut error: *mut c_char = std::ptr::null_mut();
            let hits = take(hinny_query(tagged, c(q).as_ptr(), std::ptr::null(), 0, &mut error));
            assert!(error.is_null());
            let mut keys: Vec<u64> = hits.as_array().unwrap().iter().map(|h| h["key"].as_u64().unwrap()).collect();
            keys.sort_unstable();
            keys
        };
        assert_eq!(keys(r#"{"filter": {"eq": ["labels", "tax"]}}"#), [1, 2]);
        assert_eq!(keys(r#"{"filter": {"in": ["labels", ["inbox", "2025"]]}}"#), [1, 2]);
        assert_eq!(keys(r#"{"filter": {"and": [{"eq": ["labels", "tax"]}, {"eq": ["labels", "inbox"]}]}}"#), [2]);
        assert_eq!(keys(r#"{"filter": {"not": {"exists": "labels"}}}"#), [3]);
        assert_eq!(keys(r#"{"filter": {"prefix": ["labels", "IN"]}}"#), [2]);
        assert_eq!(keys(r#"{"filter": {"glob": ["labels", "t?x"]}}"#), [1, 2]);
        assert_eq!(keys(r#"{"filter": {"glob": ["labels", "IN*"]}}"#), Vec::<u64>::new());
        assert_eq!(keys(r#"{"filter": {"iglob": ["labels", "IN*"]}}"#), [2]);
        assert_eq!(take(hinny_count_by(tagged, c("labels").as_ptr(), std::ptr::null(), 1, &mut error)), serde_json::json!([["tax", 2]]));
        assert!(error.is_null());
        assert_eq!(hinny_upsert(tagged, c(r#"{"key": 4, "tags": {"labels": "tax"}}"#).as_ptr(), std::ptr::null(), 0, &mut error), -1);
        assert!(message(&mut error).contains("array of names"));
        hinny_namespace_close(tagged);

        // Patterns: the whole string, `*` any run, `?` any one; `iglob` whatever the case.
        let files = hinny_namespace_open(db, c("files").as_ptr(), c(r#"{"strings": ["path"]}"#).as_ptr(), &mut error);
        assert!(!files.is_null() && error.is_null());
        let batch = c(r#"[{"key": 1, "strings": {"path": "/home/mia/taxes/2025.pdf"}}, {"key": 2, "strings": {"path": "/home/leo/taxes/2024.PDF"}}, {"key": 3, "strings": {"path": "/home/mia/photos/2025.pdf.jpg"}}]"#);
        assert_eq!(hinny_apply(files, batch.as_ptr(), &mut error), 0);
        let keys = |q: &str| -> Vec<u64> {
            let mut error: *mut c_char = std::ptr::null_mut();
            let hits = take(hinny_query(files, c(q).as_ptr(), std::ptr::null(), 0, &mut error));
            assert!(error.is_null());
            let mut keys: Vec<u64> = hits.as_array().unwrap().iter().map(|h| h["key"].as_u64().unwrap()).collect();
            keys.sort_unstable();
            keys
        };
        assert_eq!(keys(r#"{"filter": {"glob": ["path", "/home/*/taxes/*.pdf"]}}"#), [1]);
        assert_eq!(keys(r#"{"filter": {"iglob": ["path", "/home/*/taxes/*.pdf"]}}"#), [1, 2]);
        assert_eq!(keys(r#"{"filter": {"glob": ["path", "/home/???/taxes/20??.pdf"]}}"#), [1]);
        assert_eq!(keys(r#"{"filter": {"glob": ["path", "*.pdf"]}}"#), [1]);
        assert_eq!(keys(r#"{"filter": {"glob": ["path", "*.pdf*"]}}"#), [1, 3]);
        hinny_namespace_close(files);

        // A document by key; one that isn't there is NULL without an error.
        let got = take(hinny_get(ns, 1, &mut error));
        assert_eq!(got["attributes"]["year"], 2025);
        assert_eq!(got["text"]["title"], "Lease");
        assert!(hinny_get(ns, 99, &mut error).is_null() && error.is_null());
        assert_eq!(hinny_delete(ns, 3, &mut error), 0);
        assert!(hinny_get(ns, 3, &mut error).is_null() && error.is_null());

        // Durable, in a segment, merged, and whole.
        assert_eq!(hinny_sync(ns, &mut error), 0);
        assert_eq!(hinny_flush(ns, &mut error), 0);
        assert_eq!(hinny_optimize(ns, &mut error), 0);
        let stats = take(hinny_stats(ns, &mut error));
        assert_eq!((stats["documents"].as_u64(), stats["segments"].as_u64()), (Some(2), Some(1)));
        let verified = take(hinny_verify(ns, &mut error));
        assert_eq!((verified["segments"].as_u64(), verified["unchecked_segments"].as_u64()), (Some(1), Some(0)));
        assert!(error.is_null());

        hinny_namespace_close(ns);
        hinny_close(db);

        // Reopened: what was written is there.
        let db = hinny_open(c(d.to_str().unwrap()).as_ptr(), &mut error);
        let ns = hinny_namespace_open(db, c("docs").as_ptr(), schema.as_ptr(), &mut error);
        assert!(!ns.is_null() && error.is_null());
        assert_eq!(take(hinny_get(ns, 2, &mut error))["text"]["title"], "Tax return");
        hinny_namespace_close(ns);
        // And without restating the schema; a namespace that isn't there is an error.
        let ns = hinny_namespace_open(db, c("docs").as_ptr(), std::ptr::null(), &mut error);
        assert!(!ns.is_null() && error.is_null());
        assert_eq!(hinny_count(ns, c(r#"{"eq": ["kind", "pdf"]}"#).as_ptr(), &mut error), 2);
        // A fork holds the same, and is apart from then on.
        let fork = hinny_namespace_fork(ns, c("sandbox").as_ptr(), &mut error);
        assert!(!fork.is_null() && error.is_null());
        let all = hinny_count(ns, std::ptr::null(), &mut error);
        assert_eq!(hinny_count(fork, std::ptr::null(), &mut error), all);
        assert_eq!(hinny_delete_where(fork, c(r#"{"eq": ["kind", "nothing"]}"#).as_ptr(), &mut error), 0);
        assert_eq!(hinny_delete_where(fork, c(r#"{"eq": ["key", 1]}"#).as_ptr(), &mut error), 0);
        assert_eq!(hinny_delete(fork, 1, &mut error), 0);
        assert_eq!((hinny_count(fork, std::ptr::null(), &mut error), hinny_count(ns, std::ptr::null(), &mut error)), (all - 1, all));
        // What passes a filter, deleted at once (in the fork only).
        let pdfs = hinny_count(fork, c(r#"{"eq": ["kind", "pdf"]}"#).as_ptr(), &mut error);
        assert!(pdfs > 0);
        assert_eq!(hinny_delete_where(fork, c(r#"{"eq": ["kind", "pdf"]}"#).as_ptr(), &mut error), pdfs);
        assert_eq!(hinny_count(fork, c(r#"{"eq": ["kind", "pdf"]}"#).as_ptr(), &mut error), 0);
        assert_eq!(hinny_count(ns, c(r#"{"eq": ["kind", "pdf"]}"#).as_ptr(), &mut error), 2);
        assert_eq!(hinny_delete_where(fork, c(r#"{"nope": 1}"#).as_ptr(), &mut error), -1);
        assert!(message(&mut error).contains("no filter"));
        assert!(hinny_namespace_fork(ns, c("sandbox").as_ptr(), &mut error).is_null());
        assert!(message(&mut error).contains("already"));
        hinny_namespace_close(fork);
        hinny_namespace_close(ns);
        assert!(hinny_namespace_open(db, c("nothing").as_ptr(), std::ptr::null(), &mut error).is_null());
        assert!(message(&mut error).contains("no namespace"));
        hinny_close(db);
    }
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn patches_conditional_writes_typing_and_deleting_a_namespace() {
    let d = dir("patches");
    let mut error: *mut c_char = std::ptr::null_mut();
    unsafe {
        let db = hinny_open(c(d.to_str().unwrap()).as_ptr(), &mut error);
        let schema = c(r#"{"text": ["title", "body"], "ints": ["rev"], "strings": ["state"], "tags": ["labels"], "vector": 2}"#);
        let ns = hinny_namespace_open(db, c("docs").as_ptr(), schema.as_ptr(), &mut error);
        assert!(!ns.is_null() && error.is_null());
        let batch = c(r#"[
            {"key": 1, "version": 7, "text": {"title": "Tax return", "body": "Filed in March"}, "ints": {"rev": 1}, "strings": {"state": "draft"}, "tags": {"labels": ["a", "b"]}, "vector": [1.0, 0.0]},
            {"key": 2, "text": {"title": "Lease", "body": "The flat on Elm Street"}, "ints": {"rev": 1}, "strings": {"state": "draft"}, "vector": [0.0, 1.0]},
            {"key": 3, "text": {"title": "Notes", "body": "Taxes and other things"}, "ints": {"rev": 1}, "strings": {"state": "draft"}, "vector_b64": "AACAPwAAAAA="}
        ]"#);
        assert_eq!(hinny_apply(ns, batch.as_ptr(), &mut error), 0);
        let keys = |q: &str, v: Option<[f32; 2]>| -> Vec<u64> {
            let mut error: *mut c_char = std::ptr::null_mut();
            let (p, n) = v.as_ref().map_or((std::ptr::null(), 0), |v| (v.as_ptr(), 2));
            let hits = take(hinny_query(ns, c(q).as_ptr(), p, n, &mut error));
            assert!(error.is_null());
            let mut keys: Vec<u64> = hits.as_array().unwrap().iter().map(|h| h["key"].as_u64().unwrap()).collect();
            keys.sort_unstable();
            keys
        };
        // "vector_b64" in a document is the vector [1, 0]: document 3 is as near it as document 1.
        let nearest = take(hinny_query(ns, c(r#"{"limit": 2, "with_text": false}"#).as_ptr(), [1.0f32, 0.0].as_ptr(), 2, &mut error));
        let mut nearest: Vec<u64> = nearest.as_array().unwrap().iter().map(|h| h["key"].as_u64().unwrap()).collect();
        nearest.sort_unstable();
        assert_eq!(nearest, [1, 3]);

        // A patch that changes one attribute keeps the text, the other attributes and the vector.
        assert_eq!(hinny_patch(ns, 1, c(r#"{"strings": {"state": "final"}}"#).as_ptr(), std::ptr::null(), &mut error), 1);
        let got = take(hinny_get(ns, 1, &mut error));
        assert_eq!((got["attributes"]["state"].as_str(), got["attributes"]["rev"].as_i64()), (Some("final"), Some(1)));
        assert_eq!(got["attributes"]["labels"], serde_json::json!(["a", "b"]));
        assert_eq!((got["text"]["title"].as_str(), got["text"]["body"].as_str()), (Some("Tax return"), Some("Filed in March")));
        assert_eq!(keys(r#"{"text": "march", "filter": {"eq": ["state", "final"]}}"#, None), [1]);
        let nearest = take(hinny_query(ns, c(r#"{"limit": 1, "filter": {"eq": ["state", "final"]}}"#).as_ptr(), [1.0f32, 0.0].as_ptr(), 2, &mut error));
        assert_eq!(nearest[0]["key"], 1);
        // New text and vector, a new version; and "remove" takes an attribute and a text field away.
        assert_eq!(hinny_patch(ns, 1, c(r#"{"text": {"body": "Filed in April"}, "vector": [0.0, 1.0], "version": 9, "remove": ["labels", "title"]}"#).as_ptr(), std::ptr::null(), &mut error), 1);
        let got = take(hinny_get(ns, 1, &mut error));
        assert!(got["attributes"].get("labels").is_none() && got["text"].get("title").is_none());
        assert_eq!(got["text"]["body"], "Filed in April");
        assert_eq!(keys(r#"{"text": "april"}"#, None), [1]);
        // A key that isn't there is 0, not an error; so is a condition that doesn't hold.
        assert_eq!(hinny_patch(ns, 99, c(r#"{"ints": {"rev": 2}}"#).as_ptr(), std::ptr::null(), &mut error), 0);
        assert!(error.is_null());
        assert_eq!(hinny_patch(ns, 2, c(r#"{"ints": {"rev": 2}}"#).as_ptr(), c(r#"{"eq": ["state", "final"]}"#).as_ptr(), &mut error), 0);
        assert_eq!(hinny_patch(ns, 2, c(r#"{"ints": {"rev": 2}}"#).as_ptr(), c(r#"{"eq": ["state", "draft"]}"#).as_ptr(), &mut error), 1);
        assert_eq!(take(hinny_get(ns, 2, &mut error))["attributes"]["rev"], 2);
        // Mistakes: a part a patch hasn't, one that isn't JSON, an attribute the schema hasn't.
        assert_eq!(hinny_patch(ns, 2, c(r#"{"key": 2}"#).as_ptr(), std::ptr::null(), &mut error), -1);
        let m = message(&mut error);
        assert!(m.contains("no part \"key\"") && m.contains("remove"), "{m}");
        assert_eq!(hinny_patch(ns, 2, c("{").as_ptr(), std::ptr::null(), &mut error), -1);
        assert!(message(&mut error).contains("isn't JSON"));
        assert_eq!(hinny_patch(ns, 2, c(r#"{"ints": {"month": 3}}"#).as_ptr(), std::ptr::null(), &mut error), -1);
        assert!(message(&mut error).contains("month"));
        assert_eq!(hinny_patch(ns, 2, c("{}").as_ptr(), c(r#"{"nope": 1}"#).as_ptr(), &mut error), -1);
        assert!(message(&mut error).contains("no filter"));
        assert_eq!(hinny_patch(ns, 2, std::ptr::null(), std::ptr::null(), &mut error), -1);
        assert!(message(&mut error).contains("NULL"));

        // Every document that passes a filter, and how many that was.
        assert_eq!(hinny_patch_where(ns, c(r#"{"eq": ["state", "draft"]}"#).as_ptr(), c(r#"{"strings": {"state": "archived"}, "ints": {"rev": 5}}"#).as_ptr(), &mut error), 2);
        assert_eq!(hinny_count(ns, c(r#"{"eq": ["state", "archived"]}"#).as_ptr(), &mut error), 2);
        assert_eq!(hinny_count(ns, c(r#"{"eq": ["rev", 5]}"#).as_ptr(), &mut error), 2);
        assert_eq!(hinny_patch_where(ns, c(r#"{"eq": ["state", "draft"]}"#).as_ptr(), c(r#"{"ints": {"rev": 6}}"#).as_ptr(), &mut error), 0);
        assert_eq!(hinny_patch_where(ns, c(r#"{"nope": 1}"#).as_ptr(), c("{}").as_ptr(), &mut error), -1);
        assert!(message(&mut error).contains("no filter"));
        assert_eq!(hinny_patch_where(ns, c(r#"{"exists": "rev"}"#).as_ptr(), c(r#"{"nothing": 1}"#).as_ptr(), &mut error), -1);
        assert!(message(&mut error).contains("no part"));

        // upsert_if: written where there is no document yet, refused when the condition fails,
        // written when it holds.
        let newer = |rev: i64| c(&format!(r#"{{"key": 10, "text": {{"title": "revision {rev}"}}, "ints": {{"rev": {rev}}}}}"#));
        assert_eq!(hinny_upsert_if(ns, newer(5).as_ptr(), [1.0f32, 0.0].as_ptr(), 2, c(r#"{"range": ["rev", null, 4]}"#).as_ptr(), &mut error), 1);
        assert_eq!(hinny_upsert_if(ns, newer(4).as_ptr(), std::ptr::null(), 0, c(r#"{"range": ["rev", null, 3]}"#).as_ptr(), &mut error), 0);
        assert_eq!(take(hinny_get(ns, 10, &mut error))["attributes"]["rev"], 5);
        assert_eq!(hinny_upsert_if(ns, newer(6).as_ptr(), std::ptr::null(), 0, c(r#"{"range": ["rev", null, 5]}"#).as_ptr(), &mut error), 1);
        assert_eq!(take(hinny_get(ns, 10, &mut error))["text"]["title"], "revision 6");
        assert_eq!(hinny_upsert_if(ns, newer(7).as_ptr(), std::ptr::null(), 0, std::ptr::null(), &mut error), -1);
        assert!(message(&mut error).contains("NULL"));
        assert_eq!(hinny_upsert_if(ns, c(r#"{"text": {"title": "x"}}"#).as_ptr(), std::ptr::null(), 0, c(r#"{"exists": "rev"}"#).as_ptr(), &mut error), -1);
        assert!(message(&mut error).contains("key"));

        // delete_if, both ways.
        assert_eq!(hinny_delete_if(ns, 10, c(r#"{"eq": ["rev", 5]}"#).as_ptr(), &mut error), 0);
        assert_eq!(take(hinny_get(ns, 10, &mut error))["attributes"]["rev"], 6);
        assert_eq!(hinny_delete_if(ns, 10, c(r#"{"eq": ["rev", 6]}"#).as_ptr(), &mut error), 1);
        assert!(hinny_get(ns, 10, &mut error).is_null() && error.is_null());
        assert_eq!(hinny_delete_if(ns, 10, c(r#"{"eq": ["rev", 6]}"#).as_ptr(), &mut error), 0);
        assert_eq!(hinny_delete_if(ns, 3, c(r#"{"nope": 1}"#).as_ptr(), &mut error), -1);
        assert!(message(&mut error).contains("no filter"));

        // typing: the last word is still being typed.
        assert_eq!(keys(r#"{"text": "taxes"}"#, None), [3]);
        assert_eq!(keys(r#"{"text": "tax"}"#, None), Vec::<u64>::new());
        assert_eq!(keys(r#"{"text": "tax", "typing": 32}"#, None), [3]);
        assert_eq!(keys(r#"{"text": "fil", "typing": 0}"#, None), Vec::<u64>::new());
        assert_eq!(keys(r#"{"text": "fil", "typing": 32}"#, None), [1]);
        assert_eq!(keys(r#"{"text": "elm str", "typing": 32, "all_words": true}"#, None), [2]);
        assert!(hinny_query(ns, c(r#"{"text": "x", "typing": "yes"}"#).as_ptr(), std::ptr::null(), 0, &mut error).is_null());
        assert!(message(&mut error).contains("typing"));

        // Deleting a namespace: refused while a handle is open, gone once it is closed.
        assert_eq!(hinny_namespace_delete(db, c("docs").as_ptr(), &mut error), -1);
        assert!(message(&mut error).contains("open for writing"));
        assert_eq!(take(hinny_namespaces(db, &mut error)), serde_json::json!(["docs"]));
        assert!(hinny_count(ns, std::ptr::null(), &mut error) > 0, "still whole after the refusal");
        hinny_namespace_close(ns);
        assert_eq!(hinny_namespace_delete(db, c("docs").as_ptr(), &mut error), 0);
        assert!(error.is_null());
        assert_eq!(take(hinny_namespaces(db, &mut error)), serde_json::json!([]));
        assert!(hinny_namespace_open(db, c("docs").as_ptr(), std::ptr::null(), &mut error).is_null());
        assert!(message(&mut error).contains("no namespace"));
        assert_eq!(hinny_namespace_delete(db, c("docs").as_ptr(), &mut error), -1);
        assert!(message(&mut error).contains("no namespace"));
        assert_eq!(hinny_namespace_delete(db, c("../docs").as_ptr(), &mut error), -1);
        assert!(message(&mut error).contains("letters, digits"));
        assert_eq!(hinny_namespace_delete(db, std::ptr::null(), &mut error), -1);
        assert!(message(&mut error).contains("NULL"));
        hinny_close(db);
    }
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn mistakes_come_back_as_messages() {
    let d = dir("errors");
    let mut error: *mut c_char = std::ptr::null_mut();
    unsafe {
        // NULL where a string or a handle belongs.
        assert!(hinny_open(std::ptr::null(), &mut error).is_null());
        assert!(message(&mut error).contains("NULL"));
        assert_eq!(hinny_sync(std::ptr::null_mut(), &mut error), -1);
        assert!(message(&mut error).contains("NULL"));
        // And nowhere to put the message: still no crash.
        assert!(hinny_open(std::ptr::null(), std::ptr::null_mut()).is_null());

        let db = hinny_open(c(d.to_str().unwrap()).as_ptr(), &mut error);
        // A schema that isn't JSON, has a part there isn't, or names no such language.
        for (schema, says) in [("{", "isn't JSON"), (r#"{"texts": ["body"]}"#, "no part"), (r#"{"text": [{"name": "body", "language": "klingon"}]}"#, "klingon")] {
            assert!(hinny_namespace_open(db, c("docs").as_ptr(), c(schema).as_ptr(), &mut error).is_null());
            let m = message(&mut error);
            assert!(m.contains(says), "{schema}: {m}");
        }
        let ns = hinny_namespace_open(db, c("docs").as_ptr(), c(r#"{"text": ["body"], "ints": ["year"], "vector": 3}"#).as_ptr(), &mut error);
        assert!(!ns.is_null() && error.is_null());
        // Documents: no key, an attribute the schema doesn't have, a vector of the wrong length.
        assert_eq!(hinny_upsert(ns, c(r#"{"text": {"body": "x"}}"#).as_ptr(), std::ptr::null(), 0, &mut error), -1);
        assert!(message(&mut error).contains("key"));
        assert_eq!(hinny_upsert(ns, c(r#"{"key": 1, "ints": {"month": 3}}"#).as_ptr(), std::ptr::null(), 0, &mut error), -1);
        assert!(message(&mut error).contains("month"));
        assert_eq!(hinny_upsert(ns, c(r#"{"key": 1}"#).as_ptr(), [1.0f32, 2.0].as_ptr(), 2, &mut error), -1);
        assert!(!message(&mut error).is_empty());
        // A batch with one bad document changes nothing.
        assert_eq!(hinny_apply(ns, c(r#"[{"key": 5, "text": {"body": "fine"}}, {"key": -1}]"#).as_ptr(), &mut error), -1);
        assert!(message(&mut error).contains("key"));
        assert_eq!(take(hinny_stats(ns, &mut error))["documents"], 0);
        // Queries: a filter there isn't, a part there isn't.
        assert!(hinny_query(ns, c(r#"{"filter": {"equals": ["year", 1]}}"#).as_ptr(), std::ptr::null(), 0, &mut error).is_null());
        assert!(message(&mut error).contains("equals"));
        assert!(hinny_query(ns, c(r#"{"texts": "x"}"#).as_ptr(), std::ptr::null(), 0, &mut error).is_null());
        assert!(message(&mut error).contains("texts"));
        hinny_namespace_close(ns);
        hinny_close(db);
        // Closing NULL, freeing NULL: nothing.
        hinny_namespace_close(std::ptr::null_mut());
        hinny_close(std::ptr::null_mut());
        hinny_free_string(std::ptr::null_mut());
    }
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn string_ids_and_the_version() {
    unsafe {
        let (a, b) = (hinny_key(c("doc-17").as_ptr()), hinny_key(c("doc-18").as_ptr()));
        assert_eq!(a, hinny_key(c("doc-17").as_ptr()));
        assert_ne!(a, b);
        assert!(a < 1 << 63);
        assert_eq!(CStr::from_ptr(hinny_version()).to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }
}

/// Three vectors against the query [1, 0]: by cosine the short [0.5, 0] in the query's direction
/// is first and [3, 1] second; by Euclidean distance [1, 1] is nearer than [3, 1]; by dot product
/// the long [3, 1] is first and the short one last. The scores are worked out by hand: minus the
/// squared distance, the product.
#[test]
fn vectors_are_compared_by_the_schemas_metric() {
    let d = dir("metrics");
    let mut error: *mut c_char = std::ptr::null_mut();
    unsafe {
        let db = hinny_open(c(d.to_str().unwrap()).as_ptr(), &mut error);
        let batch = c(r#"[{"key": 1, "vector": [0.5, 0.0]}, {"key": 2, "vector": [3.0, 1.0]}, {"key": 3, "vector": [1.0, 1.0]}]"#);
        let ranked = |ns: *mut HinnyNamespace| -> Vec<(u64, f64)> {
            let mut error: *mut c_char = std::ptr::null_mut();
            let hits = take(hinny_query(ns, c("{}").as_ptr(), [1.0f32, 0.0].as_ptr(), 2, &mut error));
            hits.as_array().unwrap().iter().map(|h| (h["key"].as_u64().unwrap(), h["score"].as_f64().unwrap())).collect()
        };
        let close = |got: &[(u64, f64)], want: &[(u64, f64)]| got.len() == want.len() && got.iter().zip(want).all(|(g, w)| g.0 == w.0 && (g.1 - w.1).abs() < 1e-4);
        for (metric, want, other) in [
            ("cosine", [(1, 1.0), (2, 0.948683), (3, 0.707107)], "dot"),
            ("euclidean", [(1, -0.25), (3, -1.0), (2, -5.0)], "dot"),
            ("dot", [(2, 3.0), (3, 1.0), (1, 0.5)], "euclidean"),
        ] {
            let schema = c(&format!(r#"{{"vector": {{"dimensions": 2, "metric": "{metric}"}}}}"#));
            let ns = hinny_namespace_open(db, c(metric).as_ptr(), schema.as_ptr(), &mut error);
            assert!(!ns.is_null() && error.is_null(), "{metric}");
            assert_eq!(hinny_apply(ns, batch.as_ptr(), &mut error), 0);
            let got = ranked(ns);
            assert!(close(&got, &want), "{metric}: {got:?}");
            hinny_namespace_close(ns);
            // Opened again as it is, it still compares by its metric; with another, it's refused.
            let ns = hinny_namespace_open(db, c(metric).as_ptr(), std::ptr::null(), &mut error);
            assert!(!ns.is_null() && error.is_null());
            let got = ranked(ns);
            assert!(close(&got, &want), "{metric}, opened again: {got:?}");
            hinny_namespace_close(ns);
            let schema = c(&format!(r#"{{"vector": {{"dimensions": 2, "metric": "{other}"}}}}"#));
            assert!(hinny_namespace_open(db, c(metric).as_ptr(), schema.as_ptr(), &mut error).is_null());
            let m = message(&mut error);
            assert!(m.contains("another schema"), "{metric} as {other}: {m}");
        }
        // A metric there isn't names the three there are.
        assert!(hinny_namespace_open(db, c("other").as_ptr(), c(r#"{"vector": {"dimensions": 2, "metric": "manhattan"}}"#).as_ptr(), &mut error).is_null());
        let m = message(&mut error);
        assert!(m.contains("manhattan") && m.contains("cosine, euclidean, dot"), "{m}");
        hinny_close(db);
    }
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn patches_in_a_namespace_with_sparse_fields_give_the_terms_again() {
    let d = dir("sparse-patches");
    let mut error: *mut c_char = std::ptr::null_mut();
    unsafe {
        let db = hinny_open(c(d.to_str().unwrap()).as_ptr(), &mut error);
        let ns = hinny_namespace_open(db, c("docs").as_ptr(), c(r#"{"text": ["body"], "sparse": ["terms"], "ints": ["year"]}"#).as_ptr(), &mut error);
        assert!(!ns.is_null() && error.is_null());
        assert_eq!(hinny_upsert(ns, c(r#"{"key": 1, "text": {"body": "The lease"}, "sparse": {"terms": {"lease": 1.4}}, "ints": {"year": 2025}}"#).as_ptr(), std::ptr::null(), 0, &mut error), 0);
        let found = || -> Vec<u64> {
            let mut error: *mut c_char = std::ptr::null_mut();
            let hits = take(hinny_query(ns, c(r#"{"sparse": {"terms": {"lease": 1.0}}}"#).as_ptr(), std::ptr::null(), 0, &mut error));
            hits.as_array().unwrap().iter().map(|h| h["key"].as_u64().unwrap()).collect()
        };
        assert_eq!(found(), [1]);
        // Without the terms: refused, and nothing changes.
        assert_eq!(hinny_patch(ns, 1, c(r#"{"ints": {"year": 2026}}"#).as_ptr(), std::ptr::null(), &mut error), -1);
        let m = message(&mut error);
        assert!(m.contains("sparse"), "{m}");
        assert_eq!(hinny_patch_where(ns, c(r#"{"exists": "year"}"#).as_ptr(), c(r#"{"ints": {"year": 2026}}"#).as_ptr(), &mut error), -1);
        assert!(message(&mut error).contains("sparse"));
        assert_eq!(take(hinny_get(ns, 1, &mut error))["attributes"]["year"], 2025);
        // With them: written, and the terms are still found.
        assert_eq!(hinny_patch(ns, 1, c(r#"{"ints": {"year": 2026}, "sparse": {"terms": {"lease": 1.4}}}"#).as_ptr(), std::ptr::null(), &mut error), 1);
        assert_eq!(take(hinny_get(ns, 1, &mut error))["attributes"]["year"], 2026);
        assert_eq!(found(), [1]);
        assert_eq!(hinny_patch_where(ns, c(r#"{"exists": "year"}"#).as_ptr(), c(r#"{"ints": {"year": 2027}, "sparse": {"terms": {"lease": 1.4}}}"#).as_ptr(), &mut error), 1);
        assert_eq!(found(), [1]);
        // Or with the field removed: written, and the terms are gone.
        assert_eq!(hinny_patch(ns, 1, c(r#"{"ints": {"year": 2028}, "remove": ["terms"]}"#).as_ptr(), std::ptr::null(), &mut error), 1);
        assert_eq!(take(hinny_get(ns, 1, &mut error))["attributes"]["year"], 2028);
        assert_eq!(found(), Vec::<u64>::new());
        hinny_namespace_close(ns);
        hinny_close(db);
    }
    std::fs::remove_dir_all(&d).unwrap();
}
