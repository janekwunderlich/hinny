//! Pruning must never change the results: block-max MaxScore, scoring term at a time, the
//! intersection, and scoring only a filter's passages, against scoring every document, on a synthetic corpus with a skewed vocabulary (a few very common words, a long
//! tail), over many random OR and AND queries and several k.

use hinny::bitmap::Bitmap;
use hinny::search::{Hit, Query, Searcher, Sink, TopK};
use hinny::segment::{self, Segment};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Roughly Zipf: word i with probability ∝ 1/(i+1).
    fn word(&mut self, vocabulary: usize) -> String {
        let u = (self.next() % 1_000_000) as f64 / 1_000_000.0;
        let i = ((vocabulary as f64 + 1.0).powf(u) - 1.0) as usize;
        format!("w{}", i.min(vocabulary - 1))
    }
}

#[test]
fn pruned_results_equal_exhaustive() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let docs: Vec<[String; 2]> = (0..20_000)
        .map(|_| {
            let title: Vec<String> = (0..1 + rng.next() % 8).map(|_| rng.word(3000)).collect();
            let body: Vec<String> = (0..rng.next() % 300).map(|_| rng.word(3000)).collect();
            [title.join(" "), body.join(" ")]
        })
        .collect();
    let dir = std::env::temp_dir().join(format!("hinny-text-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("seg");
    segment::build(&path, 2, docs.len(), &|d, f| docs[d][f].as_str()).unwrap();
    let seg = Segment::open(&path).unwrap();
    let searcher = Searcher::new(&seg);
    let fields = [(0u8, 2.0f32), (1u8, 1.0f32)];
    let (mut checked, mut taat) = (0, 0);
    for round in 0..400 {
        let words: Vec<String> = (0..1 + rng.next() % 12).map(|_| rng.word(3000)).collect();
        let all = round % 3 == 0;
        let q = Query::parse(&seg, &words.join(" "), &fields, all && words.len() <= 4);
        taat += searcher.term_at_a_time(&q, 10) as usize;
        for k in [1, 10, 200] {
            let slow = searcher.exhaustive(&q, k);
            // Whichever way `search` goes, and term at a time whether or not it would.
            let mut ways = vec![searcher.search(&q, k)];
            if !q.all {
                ways.push(searcher.accumulate(&q, k));
            }
            for fast in ways {
                assert_eq!(fast.len(), slow.len(), "round {round} k {k}");
                for (a, b) in fast.iter().zip(&slow) {
                    // Same scores (summed in another order: within rounding); ties may swap documents.
                    assert!((a.score - b.score).abs() <= 1e-4 * b.score.max(1.0), "round {round} k {k}: {a:?} vs {b:?}");
                }
            }
            checked += 1;
        }
    }
    // Both ways were taken.
    assert!(taat > 20 && taat < 380, "{taat} of 400 term at a time");
    let (q, _) = Query::prefix(&seg, "w1", &fields, 32);
    assert_eq!(searcher.search(&q, 50).len(), 50);
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(checked, 1200);
}

/// A filter few pass leads the search (`Searcher::search_among`): the lists jump to its passages.
/// The same results as scoring every passage and dropping those that don't pass, for filters of
/// every density, words in any number and "all words".
#[test]
fn scoring_a_filters_passages_equals_filtering_every_score() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let docs: Vec<[String; 2]> = (0..6000)
        .map(|_| {
            let title: Vec<String> = (0..1 + rng.next() % 8).map(|_| rng.word(800)).collect();
            let body: Vec<String> = (0..rng.next() % 200).map(|_| rng.word(800)).collect();
            [title.join(" "), body.join(" ")]
        })
        .collect();
    let dir = std::env::temp_dir().join(format!("hinny-among-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("seg");
    segment::build(&path, 2, docs.len(), &|d, f| docs[d][f].as_str()).unwrap();
    let seg = Segment::open(&path).unwrap();
    let searcher = Searcher::new(&seg);
    let fields = [(0u8, 2.0f32), (1u8, 1.0f32)];
    let n = docs.len();
    let mut found = 0;
    for round in 0..300 {
        // One passage in two, in fifty, in a thousand; a stretch of them; one; none; all.
        let pick = rng.next();
        let among = match round % 7 {
            0 => Bitmap::from_words(n, |w| pick.rotate_left(w as u32) & 0x5555_5555_5555_5555),
            1 => Bitmap::from_words(n, |w| if (w as u64 + pick) % 3 == 0 { 1 << (pick % 64) } else { 0 }),
            2 => Bitmap::from_words(n, |w| if (w as u64 + pick) % 16 == 0 { 1 << (pick % 64) } else { 0 }),
            3 => Bitmap::from_words(n, |w| if (20..24).contains(&(w as u64 + pick % 40)) { u64::MAX } else { 0 }),
            4 => Bitmap::from_words(n, |w| if w as u64 == pick % (n as u64 / 64) { 1 << (pick % 64) } else { 0 }),
            5 => Bitmap::new(n),
            _ => Bitmap::from_words(n, |_| u64::MAX),
        };
        let words: Vec<String> = (0..1 + rng.next() % 5).map(|_| rng.word(800)).collect();
        let q = Query::parse(&seg, &words.join(" "), &fields, round % 3 == 0);
        for k in [1, 10, 200] {
            let passes = |d: u32| among.get(d);
            let run = |among_only: bool| -> Vec<Hit> {
                let mut top = TopK::new(k);
                let mut sink = Sink { top: &mut top, deleted: None, filter: (!among_only).then_some(&passes as &dyn Fn(u32) -> bool), ordinal: 0, stop: &hinny::stop::Stop::NEVER, key: None };
                match among_only {
                    true => searcher.search_among(&q, &among, &mut sink),
                    false => searcher.exhaustive_into(&q, &mut sink),
                }
                top.into_sorted()
            };
            let (fast, slow) = (run(true), run(false));
            assert_eq!(fast.len(), slow.len(), "round {round} k {k}: {words:?}");
            for (a, b) in fast.iter().zip(&slow) {
                assert!((a.score - b.score).abs() <= 1e-4 * b.score.max(1.0), "round {round} k {k}: {a:?} vs {b:?}");
                assert!(among.get(a.doc as u32), "round {round}: {a:?} doesn't pass");
            }
            found += fast.len();
        }
    }
    assert!(found > 5000, "{found} results in all");
    std::fs::remove_dir_all(&dir).unwrap();
}
