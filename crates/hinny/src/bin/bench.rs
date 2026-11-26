//! Tier 1 benchmark: our engine against Tantivy 0.27 on the same passages, the same words (one
//! tokenizer rule) and the same BM25, plus a check of our pruning against exhaustive scoring.
//! Prints timings, sizes and counts only, never text.
//!
//!   text-bench build passages.tsv COPIES DIR    build both indexes (COPIES × the passages)
//!   text-bench query passages.tsv COPIES DIR    query both: latency, agreement, pruning

use std::path::{Path, PathBuf};
use std::time::Instant;

use hinny::search::{DECODED_BLOCKS, Hit, Query, Searcher};
use hinny::segment::{self, Segment};
use hinny::tokenize::tokens;
use tantivy::collector::TopDocs;
use tantivy::indexer::NoMergePolicy;
use tantivy::query::{BooleanQuery, BoostQuery, Occur, Query as TQuery, TermQuery};
use tantivy::schema::{FAST, Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions};
use tantivy::{Index, IndexWriter, TantivyDocument, Term};

const HUMAN: &[&str] = &["invoice", "tax return", "flight", "meeting tomorrow", "password reset", "receipt", "contract", "insurance", "rent", "booking confirmation", "refund", "delivery", "appointment", "bank statement", "newsletter"];
const AGENT: &[&str] = &[
    "find the email where the accountant confirms the federal and state tax return was filed and mentions the refund amount",
    "what time does my flight to lisbon leave and which terminal gate and seat number did the airline confirm",
    "invoice from the electricity company for last month with the total amount due and the payment deadline",
    "did anyone send me the signed rental agreement or lease contract for the apartment and when does it start",
    "summarize the messages about the project deadline and who is responsible for the next steps in the plan",
    "the hotel booking confirmation for the conference including check in time address and cancellation policy",
    "which subscription renewals are coming up this month and how much will each one cost me in total",
    "find the receipt for the laptop I bought including the warranty information and the order number",
    "what did the doctor say about the appointment and the test results and when is the follow up visit",
    "emails from the bank about a new card or a change to my account terms and fees",
    "who organized the team offsite and what are the dates the location and the agenda for each day",
    "the password reset or security alert emails for my accounts in the last few weeks and which services",
];
const PREFIX: &[&str] = &["inv", "tax", "fli", "conf", "app", "rec"];
const STOP: &str = "the a an and or of to in on at for from by with about is are was were be my me i did does do what which who when where how this that it its any each one new last few next up";
/// Title matches count double, as in the FTS5 benchmark (bm25(p, 2.0, 1.0)).
const FIELDS: [(u8, f32); 2] = [(0, 2.0), (1, 1.0)];
const PREFIX_EXPANSIONS: usize = 32;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: text-bench build|query passages.tsv COPIES DIR");
        std::process::exit(2);
    }
    let text = std::fs::read_to_string(&args[2]).expect("passages");
    let passages: Vec<(&str, &str)> = text.lines().map(|l| l.split_once('\t').unwrap_or((l, ""))).collect();
    let copies: usize = args[3].parse().unwrap();
    let dir = PathBuf::from(&args[4]);
    std::fs::create_dir_all(&dir).unwrap();
    let count = passages.len() * copies;
    let passages = &passages;
    let field = move |d: usize, f: usize| {
        let p = passages[d % passages.len()];
        if f == 0 { p.0 } else { p.1 }
    };
    match args[1].as_str() {
        "build" => build(&dir, count, &field),
        "query" => query(&dir),
        other => panic!("unknown command {other}"),
    }
}

fn schema() -> (Schema, Field, Field, Field) {
    let mut b = Schema::builder();
    // Frequencies, no positions: what BM25 needs, like our engine.
    let text = TextOptions::default().set_indexing_options(TextFieldIndexing::default().set_tokenizer("default").set_index_option(IndexRecordOption::WithFreqs));
    let title = b.add_text_field("title", text.clone());
    let body = b.add_text_field("body", text);
    let id = b.add_u64_field("id", FAST);
    (b.build(), title, body, id)
}

fn dir_size(p: &Path) -> u64 {
    std::fs::read_dir(p).map_or(0, |entries| entries.flatten().map(|e| if e.path().is_dir() { dir_size(&e.path()) } else { e.metadata().map_or(0, |m| m.len()) }).sum())
}

fn build<'a>(dir: &Path, count: usize, field: &(dyn Fn(usize, usize) -> &'a str + Sync)) {
    let text_mb = (0..count).map(|d| field(d, 0).len() + field(d, 1).len()).sum::<usize>() as f64 / 1e6;
    println!("{count} passages, {text_mb:.0} MB of text, {} threads", rayon::current_num_threads());

    let ours = dir.join("ours.seg");
    let t = Instant::now();
    segment::build(&ours, 2, count, field).unwrap();
    let size = std::fs::metadata(&ours).unwrap().len();
    let seg = Segment::open(&ours).unwrap();
    println!("ours:    built in {:.1} s, {:.0} MB ({:.0}% of the text), {} terms", t.elapsed().as_secs_f64(), size as f64 / 1e6, size as f64 / 1e6 / text_mb * 100.0, seg.term_count());

    let tdir = dir.join("tantivy");
    let _ = std::fs::remove_dir_all(&tdir);
    std::fs::create_dir_all(&tdir).unwrap();
    let (schema, title, body, id) = schema();
    let t = Instant::now();
    let index = Index::create_in_dir(&tdir, schema).unwrap();
    let threads = rayon::current_num_threads().min(8);
    let mut writer: IndexWriter = index.writer_with_num_threads(threads, threads * 400_000_000).unwrap();
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for d in 0..count {
        let mut doc = TantivyDocument::default();
        doc.add_text(title, field(d, 0));
        doc.add_text(body, field(d, 1));
        doc.add_u64(id, d as u64);
        writer.add_document(doc).unwrap();
    }
    writer.commit().unwrap();
    let indexed = t.elapsed().as_secs_f64();
    // One segment, its best case for queries (and like ours).
    let segments = index.searchable_segment_ids().unwrap();
    let n_segments = segments.len();
    if segments.len() > 1 {
        writer.merge(&segments).wait().unwrap();
    }
    writer.garbage_collect_files().wait().unwrap();
    writer.wait_merging_threads().unwrap();
    let size = dir_size(&tdir);
    println!("tantivy: built in {:.1} s ({indexed:.1} s indexing into {n_segments} segments, then merged to one), {:.0} MB ({:.0}% of the text)", t.elapsed().as_secs_f64(), size as f64 / 1e6, size as f64 / 1e6 / text_mb * 100.0);
}

#[derive(Clone)]
enum Source {
    Words(String, bool),
    Prefix(&'static str),
}

struct Set {
    name: &'static str,
    queries: Vec<Source>,
}

fn query_sets() -> Vec<Set> {
    let stop: Vec<&str> = STOP.split(' ').collect();
    let dropped = |q: &str| tokens(q).into_iter().filter(|w| !stop.contains(&w.as_str())).collect::<Vec<_>>().join(" ");
    vec![
        Set { name: "human (1-3 words, AND)", queries: HUMAN.iter().map(|q| Source::Words(q.to_string(), true)).collect() },
        Set { name: "agent (OR of all words)", queries: AGENT.iter().map(|q| Source::Words(q.to_string(), false)).collect() },
        Set { name: "agent (OR, filler words dropped)", queries: AGENT.iter().map(|q| Source::Words(dropped(q), false)).collect() },
        Set { name: "prefix (top 32 completions, OR)", queries: PREFIX.iter().map(|p| Source::Prefix(p)).collect() },
    ]
}

/// Our query, and the identical (field, word, boost) groups for Tantivy.
fn ours(seg: &Segment, source: &Source) -> Query {
    match source {
        Source::Words(text, all) => Query::parse(seg, text, &FIELDS, *all),
        Source::Prefix(p) => Query::prefix(seg, p, &FIELDS, PREFIX_EXPANSIONS).0,
    }
}

fn groups_for_tantivy(seg: &Segment, source: &Source) -> (Vec<Vec<(u8, String, f32)>>, bool) {
    let present = |w: &str| -> Vec<(u8, String, f32)> { FIELDS.iter().filter(|&&(f, _)| seg.lookup(f, w).is_some()).map(|&(f, b)| (f, w.to_string(), b)).collect() };
    match source {
        Source::Words(text, all) => {
            let mut seen: Vec<String> = Vec::new();
            for w in tokens(text) {
                if !seen.contains(&w) {
                    seen.push(w);
                }
            }
            (seen.iter().map(|w| present(w)).filter(|g| !g.is_empty()).collect(), *all)
        }
        Source::Prefix(p) => (vec![Query::prefix(seg, p, &FIELDS, PREFIX_EXPANSIONS).1.iter().flat_map(|w| present(w)).collect()], false),
    }
}

fn tantivy_query(groups: &[Vec<(u8, String, f32)>], all: bool, fields: [Field; 2]) -> Box<dyn TQuery> {
    let clause = |(f, w, boost): &(u8, String, f32)| -> Box<dyn TQuery> {
        let tq = TermQuery::new(Term::from_field_text(fields[*f as usize], w), IndexRecordOption::WithFreqs);
        if *boost == 1.0 { Box::new(tq) } else { Box::new(BoostQuery::new(Box::new(tq), *boost)) }
    };
    if all && groups.len() > 1 {
        Box::new(BooleanQuery::new(groups.iter().map(|g| (Occur::Must, Box::new(BooleanQuery::new(g.iter().map(|c| (Occur::Should, clause(c))).collect())) as Box<dyn TQuery>)).collect()))
    } else {
        Box::new(BooleanQuery::new(groups.iter().flatten().map(|c| (Occur::Should, clause(c))).collect()))
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// Same scores in the same order (within float rounding); ties may hold different documents.
fn same_scores(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-4 * y.abs().max(1.0))
}

fn query(dir: &Path) {
    let seg = Segment::open(&dir.join("ours.seg")).unwrap();
    let searcher = Searcher::new(&seg);
    let index = Index::open_in_dir(dir.join("tantivy")).unwrap();
    let reader = index.reader().unwrap();
    let ts = reader.searcher();
    let schema = index.schema();
    let fields = [schema.get_field("title").unwrap(), schema.get_field("body").unwrap()];
    let ids: Vec<_> = ts.segment_readers().iter().map(|r| r.fast_fields().u64("id").unwrap()).collect();
    println!("{} passages; ours {:.0} MB, Tantivy {} segment(s)", seg.docs, seg.bytes().len() as f64 / 1e6, ts.segment_readers().len());

    // Length table: ours must be Tantivy's.
    let same_table = (0..=255u8).all(|i| hinny::norms::TABLE[i as usize] == tantivy::fieldnorm::FieldNormReader::id_to_fieldnorm(i));
    println!("field length table identical to Tantivy's: {same_table}");

    let sets = query_sets();
    let mut total_blocks = 0u64;
    let mut decoded_blocks = 0u64;
    for set in &sets {
        for k in [10usize, 200] {
            let (mut ours_t, mut tant_t) = (Vec::new(), Vec::new());
            let (mut agree, mut exact, mut verified, mut n) = (0, 0, 0, 0);
            for source in &set.queries {
                let q = ours(&seg, source);
                let (groups, all) = groups_for_tantivy(&seg, source);
                // An AND word missing from the index: no results in either engine.
                let impossible = q.impossible;
                // Pruning check and agreement (outside the timings).
                let fast = searcher.search(&q, k);
                let slow = searcher.exhaustive(&q, k);
                let tant: Vec<Hit> = if impossible {
                    Vec::new()
                } else {
                    let tq = tantivy_query(&groups, all, fields);
                    ts.search(&*tq, &TopDocs::with_limit(k).order_by_score()).unwrap().iter().map(|(s, a)| Hit { doc: ids[a.segment_ord as usize].first(a.doc_id).unwrap(), score: *s }).collect()
                };
                let scores = |h: &[Hit]| h.iter().map(|x| x.score).collect::<Vec<_>>();
                verified += same_scores(&scores(&fast), &scores(&slow)) as usize;
                agree += same_scores(&scores(&fast), &scores(&tant)) as usize;
                exact += (fast.iter().map(|h| h.doc).collect::<Vec<_>>() == tant.iter().map(|h| h.doc).collect::<Vec<_>>()) as usize;
                n += 1;
                if k == 10 {
                    let before = DECODED_BLOCKS.load(std::sync::atomic::Ordering::Relaxed);
                    let _ = searcher.search(&q, k);
                    decoded_blocks += DECODED_BLOCKS.load(std::sync::atomic::Ordering::Relaxed) - before;
                    total_blocks += q.groups.iter().flatten().map(|c| c.info.blocks() as u64).sum::<u64>();
                }
                for _ in 0..7 {
                    // Each engine builds its query inside the timing (our lookups and completions too).
                    let t = Instant::now();
                    let q = ours(&seg, source);
                    std::hint::black_box(searcher.search(&q, k));
                    ours_t.push(t.elapsed().as_secs_f64() * 1000.0);
                    if !impossible {
                        let t = Instant::now();
                        let tq = tantivy_query(&groups, all, fields);
                        std::hint::black_box(ts.search(&*tq, &TopDocs::with_limit(k).order_by_score()).unwrap());
                        tant_t.push(t.elapsed().as_secs_f64() * 1000.0);
                    }
                }
            }
            ours_t.sort_by(f64::total_cmp);
            tant_t.sort_by(f64::total_cmp);
            println!(
                "  {:<34} k={k:<4} ours p50 {:6.3} p99 {:6.3} ms | tantivy p50 {:6.3} p99 {:6.3} ms | scores = exhaustive {verified}/{n}, = tantivy {agree}/{n}, same docs {exact}/{n}",
                set.name,
                percentile(&ours_t, 0.5),
                percentile(&ours_t, 0.99),
                percentile(&tant_t, 0.5),
                percentile(&tant_t, 0.99)
            );
        }
    }
    println!("pruning (k=10): decoded {decoded_blocks} of {total_blocks} blocks the queries' lists have ({:.1}%)", decoded_blocks as f64 / total_blocks.max(1) as f64 * 100.0);
}
