//! The library interface: schemas, documents, text, vector and hybrid queries, filters, order.

use hinny::{Database, Document, Schema, Value, attr};

fn db(name: &str) -> (Database, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("hinny-api-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    (Database::open(&dir).unwrap(), dir)
}

fn schema() -> Schema {
    Schema::new().text("title", 2.0).text("body", 1.0).int("year").string("kind").vector(3)
}

#[test]
fn schemas_grow_and_refuse_incompatible_changes() {
    let (db, dir) = db("schema");
    db.namespace("docs", schema()).unwrap();
    // More attributes and a new text field at the end: fine.
    db.namespace("docs", schema().string("author").text("notes", 1.0)).unwrap();
    // Text fields reordered, or an attribute changing type: refused.
    assert!(db.namespace("docs", Schema::new().text("body", 1.0).text("title", 2.0).vector(3)).is_err());
    let year_as_string = Schema::new().text("title", 2.0).text("body", 1.0).string("year").string("kind").vector(3);
    assert!(db.namespace("docs", year_as_string).is_err());
    assert!(db.namespace("other", Schema::new().int("year").string("year")).is_err());
    assert!(db.namespace("docs", schema().vector(4)).is_err());
    assert!(db.namespace("bad name", schema()).is_err());
    assert_eq!(db.namespaces().unwrap(), vec!["docs".to_string()]);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn documents_are_checked_against_the_schema() {
    let (db, dir) = db("checks");
    let ns = db.namespace("docs", schema()).unwrap();
    assert!(ns.upsert(Document::new(1).text("nope", "x")).is_err());
    assert!(ns.upsert(Document::new(1).string("year", "2025")).is_err());
    assert!(ns.upsert(Document::new(1).int("unknown", 3)).is_err());
    assert!(ns.upsert(Document::new(1).vector(vec![1.0, 0.0])).is_err());
    assert!(ns.query().vector(vec![1.0]).run().is_err());
    ns.upsert(Document::new(1).text("title", "ok").int("year", 2025).vector(vec![1.0, 0.0, 0.0])).unwrap();
    assert_eq!(ns.get(1).unwrap().iter().find(|(n, _)| n == "year").map(|(_, v)| v.clone()), Some(Value::Int(2025)));
    ns.delete(1).unwrap();
    assert!(ns.get(1).is_none());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn text_vector_and_hybrid_queries() {
    let (db, dir) = db("hybrid");
    let ns = db.namespace("docs", schema()).unwrap();
    // 1: the words, a far vector. 2: no words, the nearest vector. 3: some words, a near vector.
    ns.upsert_many([
        Document::new(1).text("title", "apartment lease").text("body", "the lease for the apartment").int("year", 2025).string("kind", "pdf").vector(vec![0.0, 0.0, 1.0]),
        Document::new(2).text("title", "holiday photos").text("body", "beach").int("year", 2024).string("kind", "jpg").vector(vec![1.0, 0.0, 0.0]),
        Document::new(3).text("title", "lease renewal").text("body", "renewing it").int("year", 2026).string("kind", "pdf").vector(vec![0.9, 0.1, 0.0]),
        Document::new(4).text("title", "groceries").text("body", "milk").int("year", 2023).string("kind", "txt").vector(vec![0.0, 1.0, 0.0]),
    ])
    .unwrap();
    let keys = |h: &[hinny::Hit]| h.iter().map(|h| h.key).collect::<Vec<_>>();

    let text = ns.query().text("apartment lease").run().unwrap();
    assert_eq!(keys(&text)[..2], [1, 3]);
    assert!(text[0].text_score.is_some() && text[0].vector_score.is_none());

    let near = ns.query().vector(vec![1.0, 0.0, 0.0]).limit(2).run().unwrap();
    assert_eq!(keys(&near), [2, 3]);
    assert!((near[0].score - 1.0).abs() < 1e-5);

    // Hybrid: the best of each retriever (1 for text, 2 for the vector) and 3, good at both, make
    // the top three. (Scores are scaled over what each retriever found: with only two text matches
    // the weaker gets 0 from text, so 3 can rank third here; real retrievers return dozens.)
    let both = ns.query().text("lease").vector(vec![1.0, 0.0, 0.0]).limit(3).run().unwrap();
    assert!(keys(&both).contains(&1) && keys(&both).contains(&2) && keys(&both).contains(&3));
    // Rank fusion counts 3's two good ranks.
    let rrf = ns.query().text("lease").vector(vec![1.0, 0.0, 0.0]).fusion(hinny::Fusion::Rrf { k: 60.0 }).limit(3).run().unwrap();
    assert!(keys(&rrf)[..2].contains(&3));
    // The weight moves the balance to either side.
    assert_eq!(ns.query().text("lease").vector(vec![1.0, 0.0, 0.0]).text_weight(1.0).run().unwrap()[0].key, 1);
    assert_eq!(ns.query().text("lease").vector(vec![1.0, 0.0, 0.0]).text_weight(0.0).run().unwrap()[0].key, 2);
    let three = both.iter().find(|h| h.key == 3).unwrap();
    assert!(three.text_score.is_some() && three.vector_score.is_some());

    // The filter applies to both retrievers.
    let pdfs = ns.query().text("lease").vector(vec![1.0, 0.0, 0.0]).filter(attr("kind").eq("pdf")).run().unwrap();
    assert!(pdfs.iter().all(|h| h.attr("kind") == Some(&Value::Str("pdf".into()))));
    assert!(!keys(&pdfs).contains(&2));

    // Attributes alone, ordered; and filters combined.
    let newest = ns.query().filter(attr("year").gte(2024)).order_by("year", true).run().unwrap();
    assert_eq!(keys(&newest), [3, 1, 2]);
    let combined = ns.query().filter(attr("kind").eq("pdf") & !attr("year").lt(2026)).run().unwrap();
    assert_eq!(keys(&combined), [3]);
    // Reordering matches by an attribute.
    let by_year = ns.query().text("lease").order_by("year", false).run().unwrap();
    assert_eq!(keys(&by_year), [1, 3]);

    // Durable: reopened from disk (log replay), and after a flush (a segment).
    drop(ns);
    let ns = db.namespace("docs", schema()).unwrap();
    let again = |ns: &hinny::Namespace| keys(&ns.query().text("lease").vector(vec![1.0, 0.0, 0.0]).limit(3).run().unwrap());
    assert_eq!(again(&ns), keys(&both));
    ns.flush().unwrap();
    assert_eq!(again(&ns), keys(&both));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn stored_text_and_snippets() {
    let (db, dir) = db("stored");
    let filler = "Nothing about it here. ".repeat(40);
    let body = format!("{filler}The apartment lease for Elm Street was signed in March. {filler}");
    {
        let ns = db.namespace("docs", schema()).unwrap();
        ns.upsert(Document::new(1).text("title", "Lease agreement").text("body", body.clone()).int("year", 2025).vector(vec![1.0, 0.0, 0.0])).unwrap();
        ns.upsert(Document::new(2).text("title", "Electricity invoice").text("body", "March, due in 14 days").vector(vec![0.0, 1.0, 0.0])).unwrap();
        let check = |label: &str| {
            let hits = ns.query().text("apartment lease March").run().unwrap();
            let h = &hits[0];
            assert_eq!(h.key, 1, "{label}");
            assert_eq!(h.text("title"), Some("Lease agreement"), "{label}");
            assert_eq!(h.text("body"), Some(body.as_str()), "{label}");
            // Stored text isn't an attribute.
            assert_eq!(h.attributes, vec![("year".to_string(), Value::Int(2025))], "{label}");
            let (field, s) = h.snippet.as_ref().unwrap();
            assert_eq!(field, "body", "{label}: the field with all three words");
            let marked: Vec<&str> = s.highlights.iter().map(|r| &s.text[r.clone()]).collect();
            assert_eq!(marked, ["apartment", "lease", "March"], "{label}");
            assert!(s.text.chars().count() <= 245, "{label}");
            assert_eq!(ns.get(1).unwrap(), vec![("year".to_string(), Value::Int(2025))], "{label}");
            assert_eq!(ns.text(2).unwrap(), vec![("title".to_string(), "Electricity invoice".to_string()), ("body".to_string(), "March, due in 14 days".to_string())], "{label}");
        };
        check("in memory");
        ns.flush().unwrap();
        check("in a segment");
        ns.upsert(Document::new(3).text("title", "x").vector(vec![0.0, 0.0, 1.0])).unwrap();
        ns.optimize().unwrap();
        check("merged");
        // A vector query alone: the start of the first stored field.
        let v = ns.query().vector(vec![0.0, 1.0, 0.0]).limit(1).run().unwrap();
        assert_eq!(v[0].snippet.as_ref().map(|(f, s)| (f.as_str(), s.text.as_str(), s.highlights.len())), Some(("title", "Electricity invoice", 0)));
        // Opting out per query.
        let lean = ns.query().text("lease").with_text(false).snippet(0).run().unwrap();
        assert!(lean[0].text.is_empty() && lean[0].snippet.is_none());
    }
    // After a restart, from the log and segments.
    let ns = db.namespace("docs", schema()).unwrap();
    assert_eq!(ns.query().text("lease").run().unwrap()[0].text("body"), Some(body.as_str()));
    // A schema that doesn't store text; and the reserved prefix.
    let bare = db.namespace("bare", Schema::new().text("body", 1.0).store_text(false)).unwrap();
    bare.upsert(Document::new(1).text("body", "the lease")).unwrap();
    let h = &bare.query().text("lease").run().unwrap()[0];
    assert!(h.text.is_empty() && h.snippet.is_none());
    drop(bare);
    assert!(db.namespace("odd", Schema::new().text("body", 1.0).string("_fugu.text.0")).is_err());
    drop(ns);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn analyzers_stem_drop_stopwords_and_stay_fixed() {
    let (db, dir) = db("analysis");
    let english = || Schema::new().analyzer(hinny::Analyzer::english()).text("title", 2.0).text("body", 1.0).text_with("code", 1.0, hinny::Analyzer::PLAIN);
    {
        let ns = db.namespace("docs", english()).unwrap();
        ns.upsert(Document::new(1).text("title", "Filing taxes").text("body", "The tax return was filed in March").text("code", "running")).unwrap();
        ns.upsert(Document::new(2).text("title", "Running shoes").text("body", "A guide to runners and their shoes").text("code", "run")).unwrap();
        let keys = |q: &str| ns.query().text(q).run().unwrap().iter().map(|h| h.key).collect::<Vec<_>>();
        // Stemmed fields find other forms; a plain field only the word as written.
        assert_eq!(keys("taxed filings"), [1]);
        assert_eq!(keys("run")[0], 2);
        // Stopwords aren't part of the query, even when every word must match.
        assert_eq!(ns.query().text("the tax of the march").all_words().run().unwrap().iter().map(|h| h.key).collect::<Vec<_>>(), [1]);
        assert!(keys("the and of").is_empty());
        // Snippets mark what matched, as written.
        let h = &ns.query().text("taxing filed").run().unwrap()[0];
        let (field, s) = h.snippet.as_ref().unwrap();
        assert_eq!(field, "title", "both words, and the higher boost");
        assert_eq!(s.highlights.iter().map(|r| &s.text[r.clone()]).collect::<Vec<_>>(), ["Filing", "taxes"]);
        // The analysis can't change once there's data.
        drop(ns);
        assert!(db.namespace("docs", Schema::new().text("title", 2.0).text("body", 1.0).text("code", 1.0)).is_err());
    }
    // Recorded: a reopened namespace (from the log, then from segments) analyzes the same way.
    for round in 0..2 {
        let ns = db.namespace("docs", english()).unwrap();
        assert_eq!(ns.query().text("taxes").run().unwrap()[0].key, 1, "round {round}");
        ns.flush().unwrap();
    }
    // A namespace made with Lucene's English stopwords (before 0.4), opened with today's default
    // (Snowball's): it keeps the list its segments were made with.
    let lucene = hinny::Analyzer::from_code("english").unwrap();
    {
        let ns = db.namespace("old", Schema::new().text_with("body", 1.0, lucene)).unwrap();
        ns.upsert(Document::new(1).text("body", "were they filing taxes")).unwrap();
        ns.flush().unwrap();
    }
    let ns = db.namespace("old", Schema::new().text_with("body", 1.0, hinny::Analyzer::english())).unwrap();
    assert_eq!(ns.query().text("were").run().unwrap().len(), 1, "\"were\" isn't a stopword in Lucene's list");
    drop(ns);
    // A namespace with no data yet can still change its mind.
    db.namespace("fresh", Schema::new().text("body", 1.0)).unwrap();
    db.namespace("fresh", Schema::new().text_with("body", 1.0, hinny::Analyzer::german())).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sparse_fields_score_by_dot_product() {
    let (db, dir) = db("sparse");
    let schema = || Schema::new().text("body", 1.0).sparse("splade").vector(2);
    let ns = db.namespace("docs", schema()).unwrap();
    // Deterministic weights over a small vocabulary; some terms in every doc, some rare.
    let vocab: Vec<String> = (0..40).map(|i| format!("t{i}")).collect();
    let weights = |k: u64| -> Vec<(String, f32)> { (0..40u64).filter(|i| (k * 7 + i * 13) % 5 < 2).map(|i| (vocab[i as usize].clone(), 0.05 + ((k * 31 + i * 17) % 97) as f32 / 40.0)).collect() };
    let q: Vec<(String, f32)> = vec![("t3".into(), 1.5), ("t7".into(), 0.4), ("t21".into(), 2.25), ("t39".into(), 0.9)];
    let brute = |n: u64| {
        let mut s: Vec<(u64, f32)> = (0..n)
            .map(|k| {
                let w = weights(k);
                // Weights are kept to hundredths.
                let dot: f32 = q.iter().map(|(t, qw)| qw * w.iter().find(|(d, _)| d == t).map_or(0.0, |(_, dw)| (dw * 100.0).round().max(1.0) / 100.0)).sum();
                (k, dot)
            })
            .filter(|(_, d)| *d > 0.0)
            .collect();
        s.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        s
    };
    let check = |label: &str, n: u64| {
        let want = brute(n);
        let got = ns.query().sparse("splade", q.iter().map(|(t, w)| (t.as_str(), *w))).limit(10).run().unwrap();
        assert_eq!(got.len(), 10.min(want.len()), "{label}");
        for (g, w) in got.iter().zip(&want) {
            assert!((g.score - w.1).abs() < 1e-3, "{label}: {} {} against {} {}", g.key, g.score, w.0, w.1);
        }
    };
    ns.upsert_many((0..300).map(|k| Document::new(k).text("body", format!("doc {k}")).sparse("splade", &weights(k)).vector(vec![1.0, k as f32]))).unwrap();
    check("in memory", 300);
    ns.flush().unwrap();
    check("in a segment", 300);
    // Many more, for pruning across blocks; flushed into several segments plus memory.
    for chunk in 0..4u64 {
        ns.upsert_many((300 + chunk * 1500..300 + (chunk + 1) * 1500).map(|k| Document::new(k).text("body", format!("doc {k}")).sparse("splade", &weights(k)).vector(vec![1.0, k as f32]))).unwrap();
        if chunk < 3 {
            ns.flush().unwrap();
        }
    }
    check("segments and memory, pruned", 6300);
    // Text queries don't search the sparse field; hybrid with a vector works.
    assert!(ns.query().text("t3").run().unwrap().is_empty());
    let hybrid = ns.query().sparse("splade", q.iter().map(|(t, w)| (t.as_str(), *w))).vector(vec![1.0, 5.0]).limit(5).run().unwrap();
    assert_eq!(hybrid.len(), 5);
    assert!(hybrid.iter().any(|h| h.vector_score.is_some()));
    // Its terms aren't stored as text.
    assert_eq!(ns.text(5).unwrap(), vec![("body".to_string(), "doc 5".to_string())]);
    assert!(ns.query().sparse("body", [("t3", 1.0)]).run().is_err(), "body isn't a sparse field");
    drop(ns);
    let ns = db.namespace("docs", schema()).unwrap();
    assert_eq!(ns.query().sparse("splade", q.iter().map(|(t, w)| (t.as_str(), *w))).limit(3).run().unwrap().len(), 3);
    drop(ns);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hybrid_rankings_dont_depend_on_how_many_results_are_asked_for() {
    let (db, dir) = db("depth");
    let ns = db.namespace("docs", schema()).unwrap();
    let words = ["lease", "apartment", "contract", "rent", "deposit", "keys", "landlord", "street"];
    let mut seed = 7u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as usize
    };
    for key in 0..600u64 {
        let body: Vec<&str> = (0..6).map(|_| words[next() % words.len()]).collect();
        let v = [(next() % 100) as f32 / 100.0, (next() % 100) as f32 / 100.0, 1.0];
        ns.upsert(Document::new(key).text("title", words[next() % words.len()]).text("body", &body.join(" ")).vector(v.to_vec())).unwrap();
    }
    let keys = |limit: usize| -> Vec<u64> { ns.query().text("lease for the apartment").vector(vec![0.3, 0.6, 1.0]).limit(limit).snippet(0).run().unwrap().iter().map(|h| h.key).collect() };
    let ten = keys(10);
    assert_eq!(ten.len(), 10);
    for limit in [11, 25, 100] {
        assert_eq!(keys(limit)[..10], ten[..], "the top ten must not change with limit {limit}");
    }
    // A limit above a budget raises it: still that many results.
    let small = ns.query().text("lease").vector(vec![0.5, 0.5, 1.0]).candidates(5, 5).limit(30).run().unwrap();
    assert_eq!(small.len(), 30);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn ties_order_by_key_wherever_the_documents_are() {
    // Documents with the same text and vector have the same fused score: they come in key order,
    // the same for any limit, from the memtable, from segments, and after a merge into one.
    let (db, dir) = db("ties");
    let ns = db.namespace("docs", schema()).unwrap();
    let twin = |key: u64| Document::new(key).text("title", "twin").text("body", "the same words exactly").vector(vec![0.6, 0.8, 0.0]);
    let keys = [42u64, 7, 19, 3, 88, 51];
    for (i, &k) in keys.iter().enumerate() {
        ns.upsert(twin(k)).unwrap();
        if i == 2 {
            ns.flush().unwrap(); // half in a segment, half in memory
        }
    }
    for k in 100..140u64 {
        ns.upsert(Document::new(k).text("title", "other").text("body", &format!("unrelated filler {k}")).vector(vec![0.0, 0.0, 1.0])).unwrap();
    }
    let mut sorted = keys.to_vec();
    sorted.sort();
    let ranked = |limit: usize| -> Vec<u64> { ns.query().text("same words").vector(vec![0.6, 0.8, 0.0]).limit(limit).snippet(0).run().unwrap().iter().map(|h| h.key).collect() };
    assert_eq!(ranked(6), sorted);
    assert_eq!(ranked(3), sorted[..3]);
    assert_eq!(ranked(20)[..6], sorted[..]);
    let before = ns.snapshot();
    ns.optimize().unwrap();
    assert_eq!(ranked(6), sorted, "after merging into one segment");
    let old: Vec<u64> = before.query().text("same words").vector(vec![0.6, 0.8, 0.0]).limit(6).snippet(0).run().unwrap().iter().map(|h| h.key).collect();
    assert_eq!(old, sorted, "a snapshot from before the merge");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_query_matching_a_document_on_both_sides_scores_one() {
    let (db, dir) = db("self");
    let ns = db.namespace("docs", schema()).unwrap();
    for k in 0..50u64 {
        let v = vec![(k % 5) as f32, (k % 7) as f32 + 1.0, 1.0];
        ns.upsert(Document::new(k).text("title", "note").text("body", &format!("common words and number{k}")).vector(v)).unwrap();
    }
    // Its own rare word and its own vector: the best of both lists, so 1 after min–max scaling.
    let me = 23u64;
    let v = vec![(me % 5) as f32, (me % 7) as f32 + 1.0, 1.0];
    let (hits, stats) = ns.query().text("number23 common").vector(v.clone()).limit(5).snippet(0).explain().unwrap();
    assert_eq!(hits[0].key, me);
    assert!((hits[0].score - 1.0).abs() < 1e-6, "score {}", hits[0].score);
    let s = stats.expect("a hybrid query is explained");
    assert!(s.both >= 1 && s.fused == s.lexical.found + s.vector.found - s.both, "{s:?}");
    assert_eq!((s.lexical.depth, s.vector.depth), (hinny::api::TEXT_CANDIDATES, hinny::api::VECTOR_CANDIDATES));
    let (lo, hi) = s.vector.scores.unwrap();
    assert!(lo <= hi && hits[0].vector_score == Some(hi));
    // One candidate on each side, the same document: each scales to 1 on its own.
    let (one, _) = ns.query().text("number23").vector(v).candidates(1, 1).limit(1).snippet(0).explain().unwrap();
    assert_eq!((one[0].key, one[0].score), (me, 1.0));
    // A query with one retriever isn't fused, so there's nothing to explain.
    assert!(ns.query().text("number23").explain().unwrap().1.is_none());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_sparse_field_has_its_own_budget() {
    let (db, dir) = db("sparse-budget");
    let ns = db.namespace("docs", Schema::new().text("body", 1.0).sparse("sp").vector(2)).unwrap();
    for k in 0..30u64 {
        ns.upsert(Document::new(k).text("body", &format!("words {k}")).sparse("sp", &[("t1", 1.0 + k as f32 / 10.0), ("t2", 0.5)]).vector(vec![1.0, k as f32])).unwrap();
    }
    let depth = |q: hinny::Query| q.vector(vec![1.0, 3.0]).snippet(0).explain().unwrap().1.unwrap().lexical.depth;
    assert_eq!(depth(ns.query().sparse("sp", [("t1", 1.0)])), hinny::api::SPARSE_CANDIDATES);
    assert_eq!(depth(ns.query().sparse("sp", [("t1", 1.0)]).sparse_candidates(7).limit(3)), 7);
    // With text as well, the larger of the two budgets.
    assert_eq!(depth(ns.query().text("words").sparse("sp", [("t1", 1.0)]).candidates(9, 0).sparse_candidates(4).limit(3)), 9);
    assert_eq!(depth(ns.query().text("words").sparse("sp", [("t1", 1.0)]).candidates(3, 0).sparse_candidates(12).limit(3)), 12);
    // A limit above the budget raises it.
    assert_eq!(depth(ns.query().sparse("sp", [("t1", 1.0)]).sparse_candidates(5).limit(25)), 25);
    std::fs::remove_dir_all(dir).unwrap();
}

/// Float attributes through the library: in the schema (which an earlier release still reads
/// unless it has one), in documents (an integer given for one is that number; NaN is refused),
/// in results, and in conditions written with numbers of either kind.
#[test]
fn float_attributes() {
    let (db, dir) = db("floats");
    let schema = || Schema::new().text("body", 1.0).int("at").float("score").string("kind");
    {
        let ns = db.namespace("docs", schema()).unwrap();
        ns.upsert_many([
            Document::new(1).text("body", "one").float("score", 0.25).int("at", 1_700_000_000_000),
            Document::new(2).text("body", "two").int("score", 2).int("at", 1_700_000_000_500),
            Document::new(3).text("body", "three").float("score", -1.5),
            Document::new(4).text("body", "four").string("kind", "none"),
        ])
        .unwrap();
        assert!(ns.upsert(Document::new(5).float("score", f64::NAN)).is_err());
        assert!(ns.upsert(Document::new(5).float("at", 1.5)).is_err());
        assert!(ns.upsert(Document::new(5).string("score", "high")).is_err());
        assert_eq!(ns.count(None), 4);
        ns.flush().unwrap();
    }
    // The schema as written says it has a float; a namespace without one is as before.
    assert!(std::fs::read_to_string(dir.join("docs").join("schema")).unwrap().starts_with("fugudb schema 3\n"));
    db.namespace("plain", Schema::new().text("body", 1.0).int("year")).unwrap();
    assert!(std::fs::read_to_string(dir.join("plain").join("schema")).unwrap().starts_with("fugudb schema 2\n"));
    // Reopened with the same schema; a float can't become an integer.
    assert!(db.namespace("docs", Schema::new().text("body", 1.0).int("at").int("score").string("kind")).is_err());
    let ns = db.namespace("docs", schema()).unwrap();
    let keys = |f: hinny::Filter| -> Vec<u64> { ns.query().filter(f).order_by("score", false).limit(10).run().unwrap().iter().map(|h| h.key).collect() };
    assert_eq!(ns.get(2).unwrap().iter().find(|(n, _)| n == "score").map(|(_, v)| v.clone()), Some(Value::Float(2.0)));
    assert_eq!(keys(attr("score").gte(0.25)), [1, 2]);
    assert_eq!(keys(attr("score").gt(0.25)), [2]);
    assert_eq!(keys(attr("score").lt(0.25)), [3]);
    assert_eq!(keys(attr("score").lte(2)), [3, 1, 2]);
    assert_eq!(keys(attr("score").gt(2)), Vec::<u64>::new());
    assert_eq!(keys(attr("score").between(-2, 0.5)), [3, 1]);
    assert_eq!(keys(attr("score").eq(2.0)), [2]);
    assert_eq!(keys(attr("score").eq(2)), [2]);
    assert_eq!(keys(attr("score").one_of([0.25, -1.5])), [3, 1]);
    assert_eq!(keys(!attr("score").exists()), [4]);
    assert_eq!(keys(attr("score").gt(f64::INFINITY)), Vec::<u64>::new());
    // An integer of any size is an integer, and a fraction asked of integers finds those in it.
    assert_eq!(keys(attr("at").gte(1_700_000_000_250)), [2]);
    assert_eq!(keys(attr("at").between(1.7e12 - 0.5, 1.7e12 + 0.5)), [1]);
    // Largest first, with text: by the number.
    let by_score: Vec<u64> = ns.query().text("one two three four").order_by("score", true).run().unwrap().iter().map(|h| h.key).collect();
    assert_eq!(by_score, [2, 1, 3, 4]);
    assert_eq!(ns.count_by("score", Some(&attr("score").gte(0)), 0), vec![(Value::Float(0.25), 1), (Value::Float(2.0), 1)]);
    drop(ns);
    std::fs::remove_dir_all(dir).unwrap();
}

/// What a caller is told when something is wrong, and opening a namespace as it is.
#[test]
fn errors_say_what_is_wrong_and_namespaces_open_as_they_are() {
    let (db, dir) = db("errors");
    {
        let ns = db.namespace("docs", schema()).unwrap();
        // The one integer an attribute can't be is refused, not stored as "no value".
        let e = ns.upsert(Document::new(1).text("body", "x").int("year", i64::MIN)).unwrap_err();
        assert!(e.kind() == std::io::ErrorKind::InvalidInput && e.to_string().contains("-2^63"), "{e}");
        assert!(ns.get(1).is_none());
        ns.upsert(Document::new(1).text("body", "x").int("year", i64::MIN + 1)).unwrap();
        assert_eq!(ns.get(1).unwrap(), vec![("year".to_string(), Value::Int(i64::MIN + 1))]);
        // A second handle on the same namespace, in this process: the message doesn't say
        // "another process".
        let Err(e) = db.namespace("docs", schema()) else { panic!("opened twice") };
        assert!(e.kind() == std::io::ErrorKind::WouldBlock && e.to_string().contains("already open for writing"), "{e}");
    }
    // Another schema: both are named as a person would write them.
    let Err(e) = db.namespace("docs", Schema::new().text("body", 1.0).string("year")) else { panic!("another schema accepted") };
    let message = e.to_string();
    assert!(message.contains("it has text title (boost 2), body; int year; string kind; vector 3"), "{message}");
    assert!(message.contains("asked for text body; string year") && !message.contains("Schema {"), "{message}");
    // As it is, without restating the schema; and not what doesn't exist.
    let ns = db.open_namespace("docs").unwrap();
    assert_eq!(ns.get(1).unwrap(), vec![("year".to_string(), Value::Int(i64::MIN + 1))]);
    assert_eq!(ns.query().filter(attr("year").lt(0)).run().unwrap().len(), 1);
    drop(ns);
    let Err(e) = db.open_namespace("nothing") else { panic!("opened a namespace that isn't there") };
    assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
    assert_eq!(db.namespaces().unwrap(), vec!["docs".to_string()]);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tags() {
    let (db, dir) = db("tags");
    let schema = || Schema::new().text("body", 1.0).tags("labels").string("owner").vector(2);
    {
        let ns = db.namespace("notes", schema()).unwrap();
        ns.upsert_many([
            Document::new(1).text("body", "tax return for the year").tags("labels", ["tax", "2025", "tax"]).string("owner", "mia").vector(vec![1.0, 0.0]),
            Document::new(2).text("body", "tax office letter").tags("labels", ["tax", "Inbox"]).string("owner", "ole").vector(vec![0.9, 0.1]),
            Document::new(3).text("body", "holiday plan").tags("labels", ["family"]).vector(vec![0.0, 1.0]),
            // One string is a set of one; no tags, and empty ones, are none.
            Document::new(4).text("body", "tax on the holiday").string("labels", "family").vector(vec![0.5, 0.5]),
            Document::new(5).text("body", "nothing about tax").tags("labels", [""; 2]).vector(vec![0.7, 0.3]),
        ])
        .unwrap();
        assert!(ns.upsert(Document::new(6).int("labels", 3)).is_err());
        assert!(ns.upsert(Document::new(6).tags("owner", ["a", "b"])).is_err());
        let check = |when: &str| {
            let keys = |f: hinny::Filter| -> Vec<u64> {
                let mut keys: Vec<u64> = ns.query().filter(f).limit(10).run().unwrap().iter().map(|h| h.key).collect();
                keys.sort_unstable();
                keys
            };
            assert_eq!(keys(attr("labels").has("tax")), [1, 2], "{when}");
            assert_eq!(keys(attr("labels").eq("tax")), [1, 2], "{when}");
            assert_eq!(keys(attr("labels").has("Tax")), Vec::<u64>::new(), "{when}");
            assert_eq!(keys(attr("labels").has_any(["family", "2025"])), [1, 3, 4], "{when}");
            assert_eq!(keys(attr("labels").has_all(["tax", "Inbox"])), [2], "{when}");
            assert_eq!(keys(attr("labels").has_all(["tax", "family"])), Vec::<u64>::new(), "{when}");
            assert_eq!(keys(!attr("labels").has("tax")), [3, 4, 5], "{when}");
            assert_eq!(keys(attr("labels").ne("tax")), [3, 4, 5], "{when}");
            assert_eq!(keys(attr("labels").exists()), [1, 2, 3, 4], "{when}");
            assert_eq!(keys(attr("labels").prefix("in")), [2], "{when}");
            assert_eq!(keys(attr("labels").contains("AMI")), [3, 4], "{when}");
            assert_eq!(keys(attr("labels").has("tax") & attr("owner").eq("ole")), [2], "{when}");
            assert_eq!(keys(attr("labels").gte(1)), Vec::<u64>::new(), "{when}");
            // With text, with a vector, and with both.
            let found = |q: hinny::Query| -> Vec<u64> { q.filter(attr("labels").has("family")).limit(10).run().unwrap().iter().map(|h| h.key).collect() };
            assert_eq!(found(ns.query().text("tax holiday")), [4, 3], "{when}");
            assert_eq!(found(ns.query().vector(vec![0.0, 1.0])), [3, 4], "{when}");
            assert_eq!(found(ns.query().text("holiday").vector(vec![0.0, 1.0])), [3, 4], "{when}");
            // Read back as a set: each once, in byte order.
            let labels = |key: u64| ns.get(key).unwrap().into_iter().find(|(n, _)| n == "labels").map(|(_, v)| v);
            assert_eq!(labels(1), Some(Value::Tags(vec!["2025".into(), "tax".into()])), "{when}");
            assert_eq!(labels(2), Some(Value::Tags(vec!["Inbox".into(), "tax".into()])), "{when}");
            assert_eq!(labels(4), Some(Value::Tags(vec!["family".into()])), "{when}");
            assert_eq!(labels(5), None, "{when}");
            let hit = &ns.query().text("letter").run().unwrap()[0];
            assert_eq!(hit.attr("labels"), Some(&Value::Tags(vec!["Inbox".into(), "tax".into()])), "{when}");
            // A document counts for each of its tags.
            assert_eq!(ns.count(Some(&attr("labels").has("tax"))), 2, "{when}");
            assert_eq!(ns.count_by("labels", None, 0), vec![(Value::Str("family".into()), 2), (Value::Str("tax".into()), 2), (Value::Str("2025".into()), 1), (Value::Str("Inbox".into()), 1)], "{when}");
            assert_eq!(ns.count_by("labels", Some(&attr("owner").exists()), 1), vec![(Value::Str("tax".into()), 2)], "{when}");
        };
        check("unflushed");
        ns.flush().unwrap();
        check("flushed");
        // A tag taken away, and a document gone: in a second segment, then merged into one.
        ns.upsert(Document::new(2).text("body", "tax office letter").tags("labels", ["Inbox", "tax"]).string("owner", "ole").vector(vec![0.9, 0.1])).unwrap();
        ns.upsert(Document::new(7).text("body", "old").tags("labels", ["tax"]).vector(vec![0.2, 0.8])).unwrap();
        ns.delete(7).unwrap();
        ns.flush().unwrap();
        check("two segments");
        ns.optimize().unwrap();
        check("merged");
    }
    // The schema as written says it has tags, and the namespace opens as it is.
    assert!(std::fs::read_to_string(dir.join("notes").join("schema")).unwrap().starts_with("fugudb schema 4\n"));
    assert!(db.namespace("notes", Schema::new().text("body", 1.0).string("labels").string("owner").vector(2)).is_err());
    let ns = db.open_namespace("notes").unwrap();
    assert_eq!(ns.count(Some(&attr("labels").has_any(["tax", "family"]))), 4);
    drop(ns);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn deleting_what_passes_a_filter() {
    let (db, dir) = db("delete-where");
    let ns = db.namespace("memory", Schema::new().text("note", 1.0).string("user").int("at").tags("labels").vector(2)).unwrap();
    let note = |key: u64, user: &str, at: i64| Document::new(key).text("note", format!("note {key} of {user}")).string("user", user).int("at", at).tags("labels", [if key % 3 == 0 { "pinned" } else { "plain" }]).vector(vec![1.0, 0.0]);
    // In a segment, in a second one, and in memory.
    ns.upsert_many((0..100).map(|k| note(k, ["mia", "ole", "ada"][k as usize % 3], k as i64))).unwrap();
    ns.flush().unwrap();
    ns.upsert_many((100..200).map(|k| note(k, ["mia", "ole", "ada"][k as usize % 3], k as i64))).unwrap();
    ns.flush().unwrap();
    ns.upsert_many((200..260).map(|k| note(k, ["mia", "ole", "ada"][k as usize % 3], k as i64))).unwrap();
    let before = ns.snapshot();
    // A user's, wherever they are.
    assert_eq!(ns.delete_where(&attr("user").eq("ole")).unwrap(), 87);
    assert_eq!((ns.count(None), ns.count(Some(&attr("user").eq("ole")))), (173, 0));
    assert!(ns.get(1).is_none() && ns.get(0).is_some());
    assert!(ns.query().text("ole").run().unwrap().is_empty());
    assert_eq!(ns.query().vector(vec![1.0, 0.0]).limit(1000).run().unwrap().len(), 173);
    // What is old, unless it is pinned; and nothing twice.
    let expired = attr("at").lt(100) & !attr("labels").has("pinned");
    assert_eq!(ns.delete_where(&expired).unwrap(), 33);
    assert_eq!(ns.delete_where(&expired).unwrap(), 0);
    assert_eq!(ns.delete_where(&attr("user").eq("nobody")).unwrap(), 0);
    assert_eq!(ns.count(None), 140);
    assert_eq!(ns.count(Some(&attr("at").lt(100))), 34);
    // A snapshot from before still has them all; and they stay gone after a merge and reopening.
    assert_eq!(before.count(None), 260);
    drop(before);
    ns.optimize().unwrap();
    assert_eq!((ns.count(None), ns.count(Some(&attr("user").eq("ole")))), (140, 0));
    drop(ns);
    let ns = db.open_namespace("memory").unwrap();
    assert_eq!((ns.count(None), ns.count(Some(&attr("user").eq("ole"))), ns.count(Some(&expired))), (140, 0, 0));
    drop(ns);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn patches_and_conditional_writes() {
    use hinny::Patch;
    let (db, dir) = db("patch");
    let schema = || Schema::new().text("title", 2.0).text("body", 1.0).int("rev").string("state").float("score").tags("labels").vector(2);
    let ns = db.namespace("docs", schema()).unwrap();
    let doc = |key: u64| Document::new(key).version(7).text("title", format!("title {key}")).text("body", "the lease for the flat").int("rev", 1).string("state", "draft").float("score", 0.5).tags("labels", ["a", "b"]).vector(vec![1.0, 0.0]);
    ns.upsert_many((1..=40).map(doc)).unwrap();
    ns.flush().unwrap();
    ns.upsert_many((41..=60).map(doc)).unwrap();
    let labels = |key: u64| ns.get(key).unwrap().into_iter().find(|(n, _)| n == "labels").map(|(_, v)| v);
    let value = |key: u64, name: &str| ns.get(key).unwrap().into_iter().find(|(n, _)| n == name).map(|(_, v)| v);
    // In a segment (3) and in memory (50): what is named changes, the rest stays.
    for key in [3, 50] {
        assert!(ns.patch(key, &Patch::new().string("state", "final").tags("labels", ["c"]).text("body", "a new body about tax")).unwrap());
        assert_eq!(value(key, "state"), Some(Value::Str("final".into())), "{key}");
        assert_eq!(labels(key), Some(Value::Tags(vec!["c".into()])), "{key}");
        assert_eq!((value(key, "rev"), value(key, "score")), (Some(Value::Int(1)), Some(Value::Float(0.5))), "{key}");
        assert_eq!(ns.text(key).unwrap(), vec![("title".to_string(), format!("title {key}")), ("body".to_string(), "a new body about tax".to_string())], "{key}");
        assert_eq!(ns.versions()[&key], 7, "{key}");
        // Still found by its title, now by its new body and no longer by its old, and by its vector.
        let found = |q: hinny::Query| q.limit(100).run().unwrap().iter().any(|h| h.key == key);
        assert!(found(ns.query().text(&format!("title {key}")).filter(attr("state").eq("final"))), "{key}");
        assert!(found(ns.query().text("tax")) && !found(ns.query().text("lease flat").all_words()), "{key}");
        assert!(found(ns.query().vector(vec![1.0, 0.0]).filter(attr("labels").has("c"))), "{key}");
    }
    assert_eq!(ns.count(None), 60);
    // Taking away an attribute and a text field, a new vector and version; and what can't be.
    assert!(ns.patch(3, &Patch::new().remove("score").remove("body").vector(vec![0.0, 1.0]).version(9)).unwrap());
    assert_eq!((value(3, "score"), ns.text(3).unwrap().len(), ns.versions()[&3]), (None, 1, 9));
    assert_eq!(ns.query().vector(vec![0.0, 1.0]).limit(1).run().unwrap()[0].key, 3);
    assert!(!ns.patch(999, &Patch::new().int("rev", 2)).unwrap());
    assert!(ns.patch(4, &Patch::new().int("nothing", 2)).is_err());
    assert!(ns.patch(4, &Patch::new().string("rev", "two")).is_err());
    assert!(ns.patch(4, &Patch::new().remove("nothing")).is_err());
    assert!(ns.patch(4, &Patch::new().text("nothing", "x")).is_err());
    assert!(ns.patch(4, &Patch::new().vector(vec![1.0, 0.0, 0.0])).is_err());
    assert_eq!(value(4, "rev"), Some(Value::Int(1)));
    // Every document that passes a filter.
    assert_eq!(ns.patch_where(&attr("state").eq("draft"), &Patch::new().string("state", "archived").int("rev", 2)).unwrap(), 58);
    assert_eq!((ns.count(Some(&attr("state").eq("archived"))), ns.count(Some(&attr("state").eq("final"))), ns.count(Some(&attr("rev").eq(2)))), (58, 2, 58));
    assert_eq!(ns.patch_where(&attr("state").eq("draft"), &Patch::new().int("rev", 3)).unwrap(), 0);

    // Conditional writes: only over an older revision, and where there is none yet.
    let newer = |key: u64, rev: i64| Document::new(key).text("title", format!("rev {rev}")).int("rev", rev).vector(vec![1.0, 0.0]);
    assert!(ns.upsert_if(newer(100, 5), &attr("rev").lt(5)).unwrap());
    assert!(!ns.upsert_if(newer(100, 4), &attr("rev").lt(4)).unwrap());
    assert!(!ns.upsert_if(newer(100, 5), &attr("rev").lt(5)).unwrap());
    assert!(ns.upsert_if(newer(100, 6), &attr("rev").lt(6)).unwrap());
    assert_eq!(value(100, "rev"), Some(Value::Int(6)));
    assert!(ns.patch_if(100, &Patch::new().string("state", "claimed"), &!attr("state").exists()).unwrap());
    assert!(!ns.patch_if(100, &Patch::new().string("state", "claimed again"), &!attr("state").exists()).unwrap());
    assert!(!ns.patch_if(999, &Patch::new().string("state", "x"), &!attr("state").exists()).unwrap());
    assert_eq!(value(100, "state"), Some(Value::Str("claimed".into())));
    assert!(!ns.delete_if(100, &attr("state").eq("free")).unwrap());
    assert!(ns.delete_if(100, &attr("state").eq("claimed")).unwrap());
    assert!(!ns.delete_if(100, &attr("state").eq("claimed")).unwrap());
    assert!(ns.get(100).is_none());

    // All of it holds after a merge and reopening.
    ns.optimize().unwrap();
    drop(ns);
    let ns = db.open_namespace("docs").unwrap();
    assert_eq!((ns.count(None), ns.count(Some(&attr("state").eq("archived"))), ns.count(Some(&attr("labels").has("c")))), (60, 58, 2));
    assert_eq!(ns.text(50).unwrap()[1].1, "a new body about tax");
    drop(ns);
    // A namespace that keeps no text can't be patched: there would be nothing to index again.
    let bare = db.namespace("bare", Schema::new().text("body", 1.0).int("n").store_text(false)).unwrap();
    bare.upsert(Document::new(1).text("body", "words").int("n", 1)).unwrap();
    assert!(bare.patch(1, &Patch::new().int("n", 2)).unwrap_err().to_string().contains("store_text"));
    drop(bare);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_last_word_being_typed() {
    let (db, dir) = db("typing");
    let ns = db.namespace("docs", Schema::new().text("title", 1.0).string("kind")).unwrap();
    let titles = ["tax return 2025", "tax returns of earlier years", "retirement plan", "tax office letter", "the lease", "invoice 17", "invoices of March"];
    ns.upsert_many(titles.iter().enumerate().map(|(i, t)| Document::new(i as u64 + 1).text("title", *t).string("kind", if i % 2 == 0 { "pdf" } else { "mail" }))).unwrap();
    for when in ["memory", "segment"] {
        let keys = |q: hinny::Query| {
            let mut keys: Vec<u64> = q.limit(10).run().unwrap().iter().map(|h| h.key).collect();
            keys.sort_unstable();
            keys
        };
        // A whole word finds itself; one being typed, every word that starts with it.
        assert_eq!(keys(ns.query().text("ret")), Vec::<u64>::new(), "{when}");
        assert_eq!(keys(ns.query().text("ret").typing(32)), [1, 2, 3], "{when}");
        assert_eq!(keys(ns.query().text("tax ret").typing(32).all_words()), [1, 2], "{when}");
        assert_eq!(keys(ns.query().text("inv").typing(32)), [6, 7], "{when}");
        assert_eq!(keys(ns.query().text("inv").typing(32).filter(attr("kind").eq("pdf"))), [7], "{when}");
        // Only the last word: "ta" is not "tax" when another word follows it.
        assert_eq!(keys(ns.query().text("ta return").typing(32).all_words()), Vec::<u64>::new(), "{when}");
        // The complete word is found too, and ranks first when it is what was typed.
        assert_eq!(ns.query().text("return").typing(32).limit(1).run().unwrap()[0].key, 1, "{when}");
        ns.flush().unwrap();
    }
    drop(ns);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_patch_gives_sparse_terms_again_or_is_refused() {
    use hinny::Patch;
    let (db, dir) = db("patch-sparse");
    let ns = db.namespace("docs", Schema::new().text("body", 1.0).sparse("terms").int("rev")).unwrap();
    let terms = [("lease", 1.4f32), ("flat", 0.9)];
    ns.upsert(Document::new(1).text("body", "the lease").sparse("terms", &terms).int("rev", 1)).unwrap();
    ns.upsert(Document::new(2).text("body", "other").sparse("terms", &[("tax", 1.0)]).int("rev", 1)).unwrap();
    let found = |term: &str| -> Vec<(u64, f32)> { ns.query().sparse("terms", [(term, 1.0)]).run().unwrap().iter().map(|h| (h.key, h.score)).collect() };
    let before = found("lease");
    assert_eq!(before.len(), 1);
    for when in ["memory", "segment"] {
        // Sparse terms aren't kept as given, so a patch that leaves them out would lose them:
        // refused, and the document is as it was.
        let refused = ns.patch(1, &Patch::new().int("rev", 2)).unwrap_err();
        assert!(refused.to_string().contains("sparse field \"terms\""), "{when}: {refused}");
        assert_eq!((found("lease"), ns.get(1).unwrap()), (before.clone(), vec![("rev".to_string(), Value::Int(1))]), "{when}");
        // Given again, they are there as before; removed, they are gone and the rest stays.
        assert!(ns.patch(1, &Patch::new().int("rev", 2).sparse("terms", &terms)).unwrap());
        assert_eq!((found("lease"), found("flat").len()), (before.clone(), 1), "{when}");
        assert_eq!(ns.get(1).unwrap(), vec![("rev".to_string(), Value::Int(2))], "{when}");
        assert_eq!(ns.text(1).unwrap(), vec![("body".to_string(), "the lease".to_string())], "{when}");
        ns.upsert(Document::new(1).text("body", "the lease").sparse("terms", &terms).int("rev", 1)).unwrap();
        ns.flush().unwrap();
    }
    assert!(ns.patch(2, &Patch::new().int("rev", 5).remove("terms")).unwrap());
    assert!(found("tax").is_empty());
    assert_eq!(ns.query().text("other").run().unwrap().len(), 1);
    drop(ns);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn equal_scores_come_in_key_order() {
    let (db, dir) = db("vector-ties");
    let ns = db.namespace("docs", Schema::new().text("body", 1.0).vector(3)).unwrap();
    let same = |key: u64| Document::new(key).text("body", "x").vector(vec![0.0, 0.0, 1.0]);
    for key in [5u64, 2, 9, 7] {
        ns.upsert(same(key)).unwrap();
    }
    ns.upsert(Document::new(4).text("body", "y").vector(vec![1.0, 0.0, 0.0])).unwrap();
    // One written again later (as a patch does): its place among the documents is the last.
    ns.upsert(same(2)).unwrap();
    let by_vector = || -> Vec<u64> { ns.query().vector(vec![0.0, 0.0, 1.0]).limit(5).snippet(0).run().unwrap().iter().map(|h| h.key).collect() };
    let by_text = || -> Vec<u64> { ns.query().text("x").limit(5).snippet(0).run().unwrap().iter().map(|h| h.key).collect() };
    for when in ["memory", "segment"] {
        assert_eq!(by_vector(), [2, 5, 7, 9, 4], "{when}");
        assert_eq!(by_text(), [2, 5, 7, 9], "{when}");
        ns.flush().unwrap();
    }
    drop(ns);
    std::fs::remove_dir_all(dir).unwrap();
}
