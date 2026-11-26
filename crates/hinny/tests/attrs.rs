//! Attributes: filtered search is exact, attribute-only queries order correctly, and attributes
//! survive checkpoints, merges, deletes and a crash.

use std::collections::HashMap;

use hinny::attrs::{Filter, Value};
use hinny::index::{Doc, Found, Index, Request};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn word(&mut self) -> String {
        let u = (self.next() % 1_000_000) as f64 / 1_000_000.0;
        format!("w{}", (801f64.powf(u) - 1.0) as usize)
    }
}

const FIELDS: [(u8, f32); 1] = [(0, 1.0)];
const SENDERS: [&str; 4] = ["Mia Müller <mia@example.org>", "Bank <noreply@bank.example>", "Ole <ole@example.com>", "Tax Office <post@tax.example>"];
const LABELS: [&str; 6] = ["inbox", "Work", "tax-2025", "family", "Übung", "tax-2024"];

fn doc(rng: &mut Rng, key: u64) -> Doc {
    let text: Vec<String> = (0..5 + rng.next() % 80).map(|_| rng.word()).collect();
    let mut attrs = vec![("date".to_string(), Value::Int((rng.next() % 1000) as i64)), ("from".to_string(), Value::Str(SENDERS[(rng.next() % 4) as usize].to_string()))];
    if key % 3 == 0 {
        attrs.push(("folder".to_string(), Value::Str("Documents/Taxes".into())));
    }
    // Up to three labels (as they are read back: each once, in byte order), and now and then one
    // few documents have.
    let mut labels: Vec<String> = (0..rng.next() % 4).map(|_| LABELS[(rng.next() % 6) as usize].to_string()).collect();
    if rng.next() % 16 == 0 {
        labels.push(format!("thread-{}", key % 40));
    }
    labels.sort();
    labels.dedup();
    if !labels.is_empty() {
        attrs.push(("labels".to_string(), Value::Tags(labels)));
    }
    Doc { vector: None, key, version: 1, passages: vec![vec![text.join(" ")]], attrs }
}

fn attr<'a>(attrs: &'a [(String, Value)], name: &str) -> Option<&'a Value> {
    attrs.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

/// A number as a float, for comparing an integer with a float (those used here are small).
fn number(v: &Value) -> Option<f64> {
    match v {
        Value::Int(x) => Some(*x as f64),
        Value::Float(x) => Some(*x),
        Value::Str(_) | Value::Tags(_) => None,
    }
}

/// A pattern against a whole string, the slow and obvious way (characters, trying every split
/// for a `*`): what `hinny::attrs::glob` is checked against. No backslashes here.
fn wild(text: &str, pattern: &str) -> bool {
    fn go(t: &[char], p: &[char]) -> bool {
        match p.split_first() {
            None => t.is_empty(),
            Some(('*', rest)) => (0..=t.len()).any(|i| go(&t[i..], rest)),
            Some(('?', rest)) => !t.is_empty() && go(&t[1..], rest),
            Some((c, rest)) => t.first() == Some(c) && go(&t[1..], rest),
        }
    }
    go(&text.chars().collect::<Vec<_>>(), &pattern.chars().collect::<Vec<_>>())
}

/// A document's strings in an attribute: its one, or its tags.
fn strings<'a>(attrs: &'a [(String, Value)], name: &str) -> Vec<&'a str> {
    match attr(attrs, name) {
        Some(Value::Str(s)) => vec![s.as_str()],
        Some(Value::Tags(tags)) => tags.iter().map(String::as_str).collect(),
        _ => Vec::new(),
    }
}

fn passes(attrs: &[(String, Value)], f: &Filter) -> bool {
    match f {
        Filter::Range(n, lo, hi) => attr(attrs, n).and_then(number).is_some_and(|x| lo.is_none_or(|lo| x >= lo as f64) && hi.is_none_or(|hi| x <= hi as f64)),
        Filter::FloatRange(n, lo, hi) => attr(attrs, n).and_then(number).is_some_and(|x| lo.is_none_or(|lo| x >= lo) && hi.is_none_or(|hi| x <= hi)),
        Filter::Eq(n, v) if number(v).is_some() => attr(attrs, n).and_then(number).is_some_and(|x| Some(x) == number(v)),
        Filter::In(n, vs) => vs.iter().any(|v| passes(attrs, &Filter::Eq(n.clone(), v.clone()))),
        Filter::Or(fs) => fs.iter().any(|f| passes(attrs, f)),
        Filter::Contains(n, p) => strings(attrs, n).iter().any(|s| s.to_lowercase().contains(&p.to_lowercase())),
        Filter::Prefix(n, p) => strings(attrs, n).iter().any(|s| s.to_lowercase().starts_with(&p.to_lowercase())),
        Filter::Glob(n, p, without_case) => strings(attrs, n).iter().any(|s| if *without_case { wild(&s.to_lowercase(), &p.to_lowercase()) } else { wild(s, p) }),
        Filter::Exists(n) => attr(attrs, n).is_some(),
        // A string is one of a document's tags.
        Filter::Eq(n, Value::Str(x)) if matches!(attr(attrs, n), Some(Value::Tags(_))) => strings(attrs, n).contains(&x.as_str()),
        Filter::Eq(n, v) => attr(attrs, n) == Some(v),
        Filter::And(fs) => fs.iter().all(|f| passes(attrs, f)),
        Filter::Not(f) => !passes(attrs, f),
    }
}

#[test]
fn filtered_search_and_queries_are_exact_across_segments() {
    let dir = std::env::temp_dir().join(format!("hinny-attrs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.auto_merge = false;
    let mut live: HashMap<u64, Doc> = HashMap::new();
    for round in 0..5 {
        for _ in 0..400 {
            let key = rng.next() % 1200;
            if rng.next() % 8 == 0 {
                ix.delete(key).unwrap();
                live.remove(&key);
            } else {
                let d = doc(&mut rng, key);
                ix.upsert(&d).unwrap();
                live.insert(key, d);
            }
        }
        if round < 4 {
            ix.checkpoint().unwrap();
        }
    }
    let filters = [
        Filter::Range("date".into(), Some(200), Some(600)),
        Filter::Contains("from".into(), "MÜLLER".into()),
        Filter::And(vec![Filter::Exists("folder".into()), Filter::Range("date".into(), None, Some(500))]),
        Filter::Not(Box::new(Filter::Contains("from".into(), "bank".into()))),
        Filter::Eq("labels".into(), Value::Str("Work".into())),
        Filter::Eq("labels".into(), Value::Str("work".into())),
        Filter::In("labels".into(), vec![Value::Str("inbox".into()), Value::Str("thread-7".into()), Value::Str("none".into())]),
        Filter::Not(Box::new(Filter::Eq("labels".into(), Value::Str("inbox".into())))),
        Filter::And(vec![Filter::Eq("labels".into(), Value::Str("Work".into())), Filter::Eq("labels".into(), Value::Str("family".into()))]),
        Filter::Prefix("labels".into(), "TAX-".into()),
        Filter::Glob("labels".into(), "tax-202?".into(), false),
        Filter::Glob("labels".into(), "*B*".into(), true),
        Filter::Glob("from".into(), "*<*@example.org>".into(), false),
        Filter::Glob("from".into(), "b*".into(), true),
        Filter::Not(Box::new(Filter::Glob("from".into(), "*bank*".into(), false))),
        Filter::Contains("labels".into(), "üb".into()),
        Filter::Eq("labels".into(), Value::Str("thread-11".into())),
        Filter::And(vec![Filter::Exists("labels".into()), Filter::Range("date".into(), Some(500), None)]),
    ];
    let check = |ix: &Index, label: &str| {
        for f in &filters {
            for _ in 0..30 {
                let text = (0..1 + rng_words()).map(|i| format!("w{}", i * 7 % 50)).collect::<Vec<_>>().join(" ");
                let all = ix.search(&Request { text: &text, fields: &FIELDS, k: 100_000, all: false, expand: 0, prefix_last: 0, filter: None, sparse: &[], stop: None });
                let want: Vec<&Found> = all.iter().filter(|h| passes(&live[&h.key].attrs, f)).collect();
                let got = ix.search(&Request { text: &text, fields: &FIELDS, k: 10, all: false, expand: 0, prefix_last: 0, filter: Some(f), sparse: &[], stop: None });
                assert_eq!(got.len(), want.len().min(10), "{label} {f:?} {text}");
                for (a, b) in got.iter().zip(&want) {
                    assert!((a.score - b.score).abs() <= 1e-4 * b.score.max(1.0), "{label} {f:?}: {a:?} vs {b:?}");
                    assert!(passes(&live[&a.key].attrs, f));
                }
            }
            // Newest first by date, attributes only.
            let got = ix.query(Some(f), Some(("date", true)), 15);
            let mut want: Vec<(i64, u64)> = live.values().filter(|d| passes(&d.attrs, f)).map(|d| (match attr(&d.attrs, "date") { Some(Value::Int(x)) => *x, _ => i64::MIN }, d.key)).collect();
            want.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            assert_eq!(got.iter().map(|h| h.key).collect::<Vec<_>>(), want.iter().take(15).map(|w| w.1).collect::<Vec<_>>(), "{label} query {f:?}");
        }
        for (key, d) in live.iter().take(50) {
            let mut got = ix.get(*key).unwrap();
            let mut want = d.attrs.clone();
            got.sort_by(|a, b| a.0.cmp(&b.0));
            want.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(got, want, "{label} get {key}");
        }
    };
    check(&ix, "segments + memory");
    // A crash: replayed from the log.
    ix.simulate_crash();
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.auto_merge = false;
    check(&ix, "after replay");
    ix.checkpoint().unwrap();
    assert!(ix.start_merge(true));
    ix.wait_merge().unwrap();
    assert_eq!(ix.stats().segments, 1);
    check(&ix, "merged");
    drop(ix);
    std::fs::remove_dir_all(&dir).unwrap();
}

fn rng_words() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    N.fetch_add(1, Ordering::Relaxed) % 4
}

#[test]
fn attribute_queries_keep_the_first_items_in_order() {
    let dir = std::env::temp_dir().join(format!("hinny-attr-order-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.auto_merge = false;
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    // Per live item: its "date" (missing, a string, or an integer of few values, so ties are many).
    let mut live: HashMap<u64, Option<i64>> = HashMap::new();
    let put = |ix: &mut Index, rng: &mut Rng, live: &mut HashMap<u64, Option<i64>>| {
        // Keys on both sides of 2^63.
        let key = if rng.next() % 2 == 0 { rng.next() % 500 } else { u64::MAX - rng.next() % 500 };
        if rng.next() % 9 == 0 {
            ix.delete(key).unwrap();
            live.remove(&key);
            return;
        }
        let (attrs, date) = match rng.next() % 6 {
            0 => (vec![], None),
            1 => (vec![("date".to_string(), Value::Str("soon".into()))], None),
            _ => {
                let x = (rng.next() % 7) as i64 * 1000 - 3000;
                (vec![("date".to_string(), Value::Int(x)), ("kind".to_string(), Value::Str(if x > 0 { "late" } else { "early" }.into()))], Some(x))
            }
        };
        let passages = (0..1 + rng.next() % 5).map(|p| vec![format!("passage {p}")]).collect();
        ix.upsert(&Doc { key, version: rng.next() >> 1, passages, attrs, vector: None }).unwrap();
        live.insert(key, date);
    };
    // The order asked for, the old way: every live item's stored attributes sorted, each item once.
    // (Stored, not given: a raw index keeps a name's first type, so an integer "date" after a
    // string one in the same chunk isn't kept.)
    let want = |ix: &hinny::index::View, live: &HashMap<u64, Option<i64>>, f: Option<&Filter>, order: Option<bool>, limit: usize| -> Vec<u64> {
        let mut all: Vec<(i64, u64)> = live
            .keys()
            .filter_map(|&k| {
                let attrs = ix.get(k).expect("live");
                (f.is_none() || passes(&attrs, f.unwrap())).then(|| {
                    let date = match attr(&attrs, "date") {
                        Some(Value::Int(x)) => *x,
                        _ => i64::MIN,
                    };
                    (if order.is_some() { date } else { k as i64 }, k)
                })
            })
            .collect();
        all.sort_by(|a, b| if order == Some(true) { b.0.cmp(&a.0) } else { a.0.cmp(&b.0) }.then(a.1.cmp(&b.1)));
        all.into_iter().take(limit).map(|(_, k)| k).collect()
    };
    let late = Filter::Eq("kind".into(), Value::Str("late".into()));
    let check = |ix: &hinny::index::View, live: &HashMap<u64, Option<i64>>, at: &str| {
        for limit in [0, 1, 10, 1000, 100_000] {
            for order in [None, Some(false), Some(true)] {
                for f in [None, Some(&late)] {
                    let got = ix.query(f, order.map(|d| ("date", d)), limit);
                    assert_eq!(got.iter().map(|h| h.key).collect::<Vec<_>>(), want(ix, live, f, order, limit), "{at}: limit {limit}, order {order:?}, filter {f:?}");
                    assert!(got.iter().all(|h| h.passage == 0), "each item at its first passage");
                }
            }
        }
    };
    for round in 0..4 {
        for _ in 0..400 {
            put(&mut ix, &mut rng, &mut live);
        }
        check(&ix.view(), &live, &format!("round {round}"));
        if round < 3 {
            ix.checkpoint().unwrap();
        }
    }
    // A pinned view keeps its answers through more writes and a merge.
    let (pinned, then) = (ix.view(), live.clone());
    for _ in 0..400 {
        put(&mut ix, &mut rng, &mut live);
    }
    ix.checkpoint().unwrap();
    assert!(ix.start_merge(true));
    ix.wait_merge().unwrap();
    check(&ix.view(), &live, "merged");
    check(&pinned, &then, "pinned");
    drop((pinned, ix));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Ordered queries over segments of several zones (1,024 passages each): by key, and by columns
/// whose values follow the keys, are scattered with many ties, or are missing from a third of
/// the documents, smallest and largest first, with filters that pass a third, a hundredth, a
/// stretch of the keys, or nothing. Each against every live document sorted.
#[test]
fn ordered_queries_find_the_first_items_without_reading_every_passage() {
    let dir = std::env::temp_dir().join(format!("hinny-attr-zones-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.auto_merge = false;
    ix.flush_passages = 1_000_000;
    let mut rng = Rng(0xA076_1D64_78BD_642F);
    let mut live: HashMap<u64, Vec<(String, Value)>> = HashMap::new();
    for round in 0..4 {
        for i in 0..3500u64 {
            // Mostly new keys in order, some written again, some from 2^63 on.
            let key = match rng.next() % 10 {
                0 => rng.next() % (round * 3500 + i + 1),
                1 => u64::MAX - rng.next() % 2000,
                _ => round * 3500 + i,
            };
            if rng.next() % 12 == 0 {
                ix.delete(key).unwrap();
                live.remove(&key);
                continue;
            }
            let mut attrs = vec![
                ("seq".to_string(), Value::Int((round * 3500 + i) as i64 * 60 + (rng.next() % 600) as i64)),
                ("few".to_string(), Value::Int((rng.next() % 40) as i64 - 20)),
                ("kind".to_string(), Value::Str(["a", "b", "c"][(rng.next() % 3) as usize].to_string())),
                ("one".to_string(), Value::Int((rng.next() % 100) as i64)),
            ];
            if rng.next() % 3 != 0 {
                attrs.push(("some".to_string(), Value::Int((rng.next() % 5000) as i64)));
            }
            let passages = (0..1 + (rng.next() % 8 == 0) as usize).map(|p| vec![format!("passage {p}")]).collect();
            ix.upsert(&Doc { key, version: 1, passages, attrs: attrs.clone(), vector: None }).unwrap();
            live.insert(key, attrs);
        }
        // Three segments of several zones, and the last round in memory.
        if round < 3 {
            ix.checkpoint().unwrap();
        }
    }
    assert_eq!(ix.stats().segments, 3);
    let filters = [
        None,
        Some(Filter::Eq("kind".into(), Value::Str("a".into()))),
        Some(Filter::Eq("one".into(), Value::Int(7))),
        Some(Filter::Range("seq".into(), Some(200_000), Some(260_000))),
        Some(Filter::And(vec![Filter::Exists("some".into()), Filter::Not(Box::new(Filter::Eq("kind".into(), Value::Str("b".into()))))])),
        Some(Filter::Eq("one".into(), Value::Int(1000))),
    ];
    let check = |ix: &Index, at: &str| {
        for f in &filters {
            for order in [None, Some(("seq", true)), Some(("seq", false)), Some(("few", true)), Some(("few", false)), Some(("some", true)), Some(("some", false))] {
                let mut all: Vec<(i64, u64)> = live
                    .iter()
                    .filter(|(_, attrs)| f.as_ref().is_none_or(|f| passes(attrs, f)))
                    .map(|(&k, attrs)| {
                        let v = match order {
                            None => k as i64,
                            Some((name, _)) => match attr(attrs, name) {
                                Some(Value::Int(x)) => *x,
                                _ => i64::MIN,
                            },
                        };
                        (v, k)
                    })
                    .collect();
                all.sort_by(|a, b| if order.is_some_and(|o| o.1) { b.0.cmp(&a.0) } else { a.0.cmp(&b.0) }.then(a.1.cmp(&b.1)));
                assert_eq!(ix.count(f.as_ref()), all.len() as u64, "{at}: count of {f:?}");
                for limit in [1, 10, 300, 20_000] {
                    let got = ix.query(f.as_ref(), order, limit);
                    assert_eq!(got.iter().map(|h| h.key).collect::<Vec<_>>(), all.iter().take(limit).map(|a| a.1).collect::<Vec<_>>(), "{at}: {f:?}, {order:?}, limit {limit}");
                    assert!(got.iter().all(|h| h.passage == 0), "each item at its first passage");
                }
            }
        }
    };
    // How many items have each value: among all, and among what a filter passes; strings and
    // integers, a column a third of the items lack, and one no item has.
    let counts = |ix: &Index, at: &str| {
        for f in &filters {
            for name in ["kind", "few", "some", "nothing"] {
                let mut want: HashMap<String, u64> = HashMap::new();
                for attrs in live.values().filter(|attrs| f.as_ref().is_none_or(|f| passes(attrs, f))) {
                    if let Some(v) = attr(attrs, name) {
                        *want.entry(format!("{v:?}")).or_default() += 1;
                    }
                }
                let got = ix.count_by(name, f.as_ref(), 0);
                assert_eq!(got.iter().map(|(v, n)| (format!("{v:?}"), *n)).collect::<HashMap<_, _>>(), want, "{at}: {name} among {f:?}");
                assert_eq!(got.len(), want.len(), "{at}: each value once");
                // The most frequent first, and of the equally frequent the smaller value.
                let in_order = got.windows(2).all(|w| {
                    w[0].1 > w[1].1
                        || w[0].1 == w[1].1
                            && match (&w[0].0, &w[1].0) {
                                (Value::Int(a), Value::Int(b)) => a < b,
                                (Value::Str(a), Value::Str(b)) => a < b,
                                _ => false,
                            }
                });
                assert!(in_order, "{at}: {name} among {f:?}: {got:?}");
                assert_eq!(ix.count_by(name, f.as_ref(), 2), got[..got.len().min(2)], "{at}: the two most frequent");
            }
        }
    };
    check(&ix, "three segments and memory");
    counts(&ix, "three segments and memory");
    ix.checkpoint().unwrap();
    assert!(ix.start_merge(true));
    ix.wait_merge().unwrap();
    assert_eq!(ix.stats().segments, 1);
    check(&ix, "merged");
    counts(&ix, "merged");
    drop(ix);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A segment without deletions, every item one passage: counts come from the filter's bits and
/// the dictionary's own counts, no document read; and they are the documents'.
#[test]
fn counts_of_a_whole_segment_are_its_documents() {
    let dir = std::env::temp_dir().join(format!("hinny-attr-counts-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.auto_merge = false;
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut live: HashMap<u64, Doc> = HashMap::new();
    for key in 0..3000 {
        let d = doc(&mut rng, key);
        ix.upsert(&d).unwrap();
        live.insert(key, d);
    }
    ix.checkpoint().unwrap();
    assert_eq!((ix.stats().segments, ix.stats().memory, ix.stats().deleted), (1, 0, 0));
    assert_eq!(ix.count(None), 3000);
    let folder = Filter::Exists("folder".into());
    assert_eq!(ix.count(Some(&folder)), 1000);
    let from = |live: &HashMap<u64, Doc>, f: Option<&Filter>| -> Vec<(Value, u64)> {
        let mut want: HashMap<&str, u64> = HashMap::new();
        for d in live.values().filter(|d| f.is_none_or(|f| passes(&d.attrs, f))) {
            if let Some(Value::Str(s)) = attr(&d.attrs, "from") {
                *want.entry(s.as_str()).or_default() += 1;
            }
        }
        let mut want: Vec<(Value, u64)> = want.into_iter().map(|(s, n)| (Value::Str(s.to_string()), n)).collect();
        want.sort_by(|a, b| b.1.cmp(&a.1).then(format!("{:?}", a.0).cmp(&format!("{:?}", b.0))));
        want
    };
    assert_eq!(ix.count_by("from", None, 0), from(&live, None));
    assert_eq!(ix.count_by("from", Some(&folder), 0), from(&live, Some(&folder)));
    assert_eq!(ix.count_by("from", None, 0).iter().map(|c| c.1).sum::<u64>(), 3000);
    // Tags: a document counts for each of its own, and a tag's count is the documents that
    // have it.
    let labels = |live: &HashMap<u64, Doc>, f: Option<&Filter>| -> Vec<(Value, u64)> {
        let mut want: HashMap<&str, u64> = HashMap::new();
        for d in live.values().filter(|d| f.is_none_or(|f| passes(&d.attrs, f))) {
            for tag in strings(&d.attrs, "labels") {
                *want.entry(tag).or_default() += 1;
            }
        }
        let mut want: Vec<(Value, u64)> = want.into_iter().map(|(s, n)| (Value::Str(s.to_string()), n)).collect();
        want.sort_by(|a, b| b.1.cmp(&a.1).then(format!("{:?}", a.0).cmp(&format!("{:?}", b.0))));
        want
    };
    let work = Filter::Eq("labels".into(), Value::Str("Work".into()));
    let counted = ix.count_by("labels", None, 0);
    assert_eq!(counted, labels(&live, None));
    assert!(counted.len() > 6 && counted.iter().map(|c| c.1).sum::<u64>() > 3000, "{counted:?}");
    assert_eq!(ix.count_by("labels", Some(&folder), 0), labels(&live, Some(&folder)));
    assert_eq!(ix.count_by("labels", Some(&work), 0), labels(&live, Some(&work)));
    for (tag, n) in &counted {
        assert_eq!(ix.count(Some(&Filter::Eq("labels".into(), tag.clone()))), *n, "{tag:?}");
    }
    assert_eq!(ix.count_by("labels", None, 3), labels(&live, None)[..3]);
    // And still, once some are deleted and some written again.
    for key in (0..3000).step_by(7) {
        ix.delete(key).unwrap();
        live.remove(&key);
    }
    for key in (1..3000).step_by(50) {
        let d = doc(&mut rng, key);
        ix.upsert(&d).unwrap();
        live.insert(key, d);
    }
    assert_eq!(ix.count(None), live.len() as u64);
    assert_eq!(ix.count_by("from", None, 0), from(&live, None));
    assert_eq!(ix.count_by("from", Some(&folder), 0), from(&live, Some(&folder)));
    assert_eq!(ix.count_by("labels", None, 0), labels(&live, None));
    assert_eq!(ix.count_by("labels", Some(&work), 0), labels(&live, Some(&work)));
    assert_eq!(ix.count(Some(&work)), live.values().filter(|d| passes(&d.attrs, &work)).count() as u64);
    drop(ix);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A segment keeps the documents its most recent conditions match: asked again, a filter gives
/// the same documents; what is kept stays within its bound however many different filters come;
/// and deletions after a filter was kept are still seen (they aren't part of what's kept).
#[test]
fn kept_filters_change_no_answer_and_stay_bounded() {
    let dir = std::env::temp_dir().join(format!("hinny-attr-kept-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.auto_merge = false;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut live: HashMap<u64, Doc> = HashMap::new();
    for key in 0..2000 {
        let d = doc(&mut rng, key);
        ix.upsert(&d).unwrap();
        live.insert(key, d);
    }
    ix.checkpoint().unwrap();
    assert_eq!(ix.stats().kept_filters, 0);
    let keys = |ix: &Index, f: &Filter| -> Vec<u64> { ix.query(Some(f), None, 100_000).iter().map(|h| h.key).collect() };
    let want = |live: &HashMap<u64, Doc>, f: &Filter| -> Vec<u64> {
        let mut keys: Vec<u64> = live.values().filter(|d| passes(&d.attrs, f)).map(|d| d.key).collect();
        keys.sort_unstable();
        keys
    };
    let bank = Filter::Contains("from".into(), "bank".into());
    let both = Filter::And(vec![bank.clone(), Filter::Range("date".into(), Some(100), Some(700)), Filter::Not(Box::new(Filter::Exists("folder".into())))]);
    for _ in 0..3 {
        assert_eq!(keys(&ix, &bank), want(&live, &bank));
        assert_eq!(keys(&ix, &both), want(&live, &both));
        assert_eq!(ix.count(Some(&both)), want(&live, &both).len() as u64);
    }
    // The condition on the sender and the one on the folder: the range isn't kept, nor the whole.
    assert_eq!(ix.stats().kept_filters, 2);
    // Deleted and rewritten documents: the kept bits are the segment's, the answers the index's.
    for key in (0..2000).step_by(3) {
        ix.delete(key).unwrap();
        live.remove(&key);
    }
    for key in (1..2000).step_by(40) {
        let d = doc(&mut rng, key);
        ix.upsert(&d).unwrap();
        live.insert(key, d);
    }
    assert_eq!(keys(&ix, &bank), want(&live, &bank));
    assert_eq!(keys(&ix, &both), want(&live, &both));
    // Many different conditions: no more are kept than a segment keeps, and each still right.
    for date in 0..100 {
        let f = Filter::Eq("date".into(), Value::Int(date));
        assert_eq!(keys(&ix, &f), want(&live, &f), "date {date}");
    }
    let stats = ix.stats();
    assert_eq!(stats.kept_filters, hinny::segment::KEPT_FILTERS, "one segment's worth");
    assert_eq!(stats.kept_filter_bytes, hinny::segment::KEPT_FILTERS * 2000usize.div_ceil(64) * 8);
    assert_eq!(keys(&ix, &bank), want(&live, &bank), "asked again after it was dropped");
    drop(ix);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Float attributes: filters (ranges with and without fractions, equality, sets, on a float
/// column and on an integer one), the order of results and counts by value are the numbers',
/// in memory, in segments, after a crash replayed from the log, and after a merge.
#[test]
fn floats_filter_order_and_count_as_numbers() {
    let dir = std::env::temp_dir().join(format!("hinny-attr-floats-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.auto_merge = false;
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let special = [0.0, -0.0, 1e300, -1e300, f64::INFINITY, f64::NEG_INFINITY, 5e-324, 0.1, 1.0, -1.0, 2.5, 1.0000000000000002];
    let mut live: HashMap<u64, Vec<(String, Value)>> = HashMap::new();
    for round in 0..3 {
        for i in 0..1500u64 {
            let key = round * 1500 + i - (rng.next() % 4 == 0) as u64 * (rng.next() % (round * 1500 + i + 1));
            let score = match rng.next() % 10 {
                0 => special[(rng.next() % special.len() as u64) as usize],
                // Few distinct values (so there are counts above one), halves and quarters.
                1..=3 => (rng.next() % 40) as f64 / 4.0 - 5.0,
                _ => (rng.next() % 2_000_000) as f64 / 1000.0 - 1000.0,
            };
            let mut attrs = vec![("n".to_string(), Value::Int((rng.next() % 200) as i64 - 100))];
            if rng.next() % 5 != 0 {
                attrs.push(("score".to_string(), Value::Float(score)));
            }
            ix.upsert(&Doc { key, version: 1, passages: vec![vec![format!("doc {key}")]], attrs: attrs.clone(), vector: None }).unwrap();
            // −0.0 is kept as 0.0.
            for (_, v) in &mut attrs {
                if *v == Value::Float(0.0) {
                    *v = Value::Float(0.0);
                }
            }
            live.insert(key, attrs);
        }
        if round < 2 {
            ix.checkpoint().unwrap();
        }
    }
    let score = || "score".to_string();
    let mut filters = vec![
        Filter::Exists(score()),
        Filter::FloatRange(score(), Some(0.0), None),
        Filter::FloatRange(score(), None, Some(-0.0)),
        Filter::FloatRange(score(), Some(-4.25), Some(4.25)),
        Filter::FloatRange(score(), Some(0.1), Some(0.1)),
        Filter::FloatRange(score(), Some(1.0000000000000002), Some(2.5)),
        Filter::FloatRange(score(), Some(f64::NEG_INFINITY), Some(-1e300)),
        Filter::FloatRange(score(), Some(1e301), None),
        Filter::FloatRange(score(), Some(f64::NAN), None),
        Filter::FloatRange(score(), Some(5.0), Some(4.0)),
        Filter::Range(score(), Some(-3), Some(3)),
        Filter::Range(score(), None, Some(-1000)),
        Filter::Eq(score(), Value::Float(2.5)),
        Filter::Eq(score(), Value::Float(-0.0)),
        Filter::Eq(score(), Value::Int(1)),
        Filter::Eq(score(), Value::Float(f64::INFINITY)),
        Filter::Eq(score(), Value::Float(f64::NAN)),
        Filter::In(score(), vec![Value::Float(-1.0), Value::Int(2), Value::Float(1e300), Value::Float(123.456)]),
        Filter::FloatRange("n".into(), Some(-10.5), Some(10.5)),
        Filter::FloatRange("n".into(), Some(0.1), Some(0.9)),
        Filter::FloatRange("n".into(), Some(-1e30), Some(-99.5)),
        Filter::Eq("n".into(), Value::Float(7.0)),
        Filter::Eq("n".into(), Value::Float(7.5)),
        Filter::And(vec![Filter::FloatRange(score(), Some(-100.0), Some(100.0)), Filter::Not(Box::new(Filter::Range("n".into(), Some(0), None)))]),
        Filter::Or(vec![Filter::FloatRange(score(), Some(999.5), None), Filter::Eq("n".into(), Value::Int(-100))]),
    ];
    for _ in 0..60 {
        let (a, b) = ((rng.next() % 2_100_000) as f64 / 1000.0 - 1050.0, (rng.next() % 2_100_000) as f64 / 1000.0 - 1050.0);
        filters.push(Filter::FloatRange(score(), Some(a.min(b)), Some(a.max(b))));
    }
    // NaN compares with nothing, as the oracle's comparisons already say.
    let check = |ix: &Index, at: &str| {
        for f in &filters {
            let mut want: Vec<u64> = live.iter().filter(|(_, attrs)| passes(attrs, f)).map(|(&k, _)| k).collect();
            want.sort_unstable();
            let mut got: Vec<u64> = ix.query(Some(f), None, 100_000).iter().map(|h| h.key).collect();
            got.sort_unstable();
            assert_eq!(got, want, "{at}: {f:?}");
            assert_eq!(ix.count(Some(f)), want.len() as u64, "{at}: count of {f:?}");
        }
        // The order of results: by the number, a document without one as the smallest.
        for descending in [false, true] {
            for f in [None, Some(&filters[3])] {
                let mut all: Vec<(f64, u64)> = live
                    .iter()
                    .filter(|(_, attrs)| f.is_none_or(|f| passes(attrs, f)))
                    .map(|(&k, attrs)| (attr(attrs, "score").and_then(number).unwrap_or(f64::NEG_INFINITY - 0.0), k, attr(attrs, "score").is_some()))
                    .map(|(x, k, has)| (if has { x } else { f64::NAN }, k))
                    .collect();
                // (NaN stands for "none" here: before every number.)
                let rank = |x: f64| if x.is_nan() { (0, 0.0) } else { (1, x) };
                all.sort_by(|a, b| {
                    let (ra, rb) = (rank(a.0), rank(b.0));
                    let by = ra.0.cmp(&rb.0).then(ra.1.total_cmp(&rb.1));
                    if descending { by.reverse() } else { by }.then(a.1.cmp(&b.1))
                });
                for limit in [1, 25, 4000] {
                    let got: Vec<u64> = ix.query(f, Some(("score", descending)), limit).iter().map(|h| h.key).collect();
                    assert_eq!(got, all.iter().take(limit).map(|a| a.1).collect::<Vec<_>>(), "{at}: descending {descending}, limit {limit}, {f:?}");
                }
            }
        }
        // What a document's attributes come back as, and counts by value.
        for (key, attrs) in live.iter().take(200) {
            let mut got = ix.get(*key).unwrap();
            let mut want = attrs.clone();
            got.sort_by(|a, b| a.0.cmp(&b.0));
            want.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(got, want, "{at}: document {key}");
        }
        let mut want: HashMap<u64, u64> = HashMap::new();
        for attrs in live.values() {
            if let Some(Value::Float(x)) = attr(attrs, "score") {
                *want.entry(x.to_bits()).or_default() += 1;
            }
        }
        let got = ix.count_by("score", None, 0);
        assert_eq!(got.len(), want.len(), "{at}: distinct scores");
        for (v, n) in &got {
            let Value::Float(x) = v else { panic!("{at}: {v:?} isn't a float") };
            assert_eq!(want.get(&x.to_bits()), Some(n), "{at}: count of {x}");
        }
        assert!(got.windows(2).all(|w| w[0].1 >= w[1].1), "{at}: most frequent first");
    };
    check(&ix, "segments and memory");
    ix.simulate_crash();
    let mut ix = Index::open(&dir, 1).unwrap();
    ix.auto_merge = false;
    check(&ix, "after replay");
    ix.checkpoint().unwrap();
    assert!(ix.start_merge(true));
    ix.wait_merge().unwrap();
    assert_eq!(ix.stats().segments, 1);
    check(&ix, "merged");
    drop(ix);
    std::fs::remove_dir_all(&dir).unwrap();
}
