//! How a text field's words become terms: the tokenizer's words (letters and digits, lowercased;
//! see `tokenize`), then optionally without stopwords and stemmed (Snowball). Indexing, queries
//! and snippets all go through a field's analyzer, so "taxes" in a query finds "tax" and marks
//! "taxed".
//!
//! An index records its fields' analyzers (in the manifest): they're part of what its segments
//! mean, so they can't change once it holds data.
//!
//! The default is no stemming and no stopwords: the same terms as Tantivy's default analyzer, and
//! right for any language. English stopwords are Snowball's (as Tantivy's and LanceDB's English
//! analyzers have them), or Lucene's 33 in indexes made before 0.4 (revision 1, kept as they were
//! made); Snowball's list scored higher on four of five BEIR datasets (FiQA 0.250 → 0.256, SciFact
//! 0.685 → 0.688). German are common function words after Snowball's list. Other languages are
//! stemmed without stopwords.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Range;

use rust_stemmers::{Algorithm, Stemmer};

use crate::tokenize::for_each_token_span;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Language {
    Arabic,
    Danish,
    Dutch,
    English,
    Finnish,
    French,
    German,
    Greek,
    Hungarian,
    Italian,
    Norwegian,
    Portuguese,
    Romanian,
    Russian,
    Spanish,
    Swedish,
    Tamil,
    Turkish,
}

const LANGUAGES: [(Language, &str, Algorithm); 18] = [
    (Language::Arabic, "arabic", Algorithm::Arabic),
    (Language::Danish, "danish", Algorithm::Danish),
    (Language::Dutch, "dutch", Algorithm::Dutch),
    (Language::English, "english", Algorithm::English),
    (Language::Finnish, "finnish", Algorithm::Finnish),
    (Language::French, "french", Algorithm::French),
    (Language::German, "german", Algorithm::German),
    (Language::Greek, "greek", Algorithm::Greek),
    (Language::Hungarian, "hungarian", Algorithm::Hungarian),
    (Language::Italian, "italian", Algorithm::Italian),
    (Language::Norwegian, "norwegian", Algorithm::Norwegian),
    (Language::Portuguese, "portuguese", Algorithm::Portuguese),
    (Language::Romanian, "romanian", Algorithm::Romanian),
    (Language::Russian, "russian", Algorithm::Russian),
    (Language::Spanish, "spanish", Algorithm::Spanish),
    (Language::Swedish, "swedish", Algorithm::Swedish),
    (Language::Tamil, "tamil", Algorithm::Tamil),
    (Language::Turkish, "turkish", Algorithm::Turkish),
];

impl Language {
    pub fn name(self) -> &'static str {
        LANGUAGES.iter().find(|(l, _, _)| *l == self).unwrap().1
    }

    pub fn from_name(name: &str) -> Option<Language> {
        LANGUAGES.iter().find(|(_, n, _)| n.eq_ignore_ascii_case(name)).map(|(l, _, _)| *l)
    }

    fn algorithm(self) -> Algorithm {
        LANGUAGES.iter().find(|(l, _, _)| *l == self).unwrap().2
    }

    fn stopwords(self, revision: u8) -> &'static [&'static str] {
        match self {
            Language::English if revision >= 2 => ENGLISH,
            Language::English => LUCENE_ENGLISH,
            Language::German => GERMAN,
            _ => &[],
        }
    }
}

/// A text field's analysis: the tokenizer's words, optionally stemmed in a language, optionally
/// without that language's stopwords. Or a sparse field: terms that come with weights (a learned
/// sparse model's output), scored by dot product.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Analyzer {
    pub language: Option<Language>,
    pub stopwords: bool,
    pub sparse: bool,
    /// Which stopword list: 1 is the one indexes before 0.4 were made with, 2 the current one. They
    /// differ only for English (Lucene's 33 words, then Snowball's); 1 for everything else.
    pub revision: u8,
}

/// Sparse weights are kept in hundredths (a posting's "term frequency" is weight × 100).
pub const SPARSE_SCALE: f32 = 100.0;

/// Sparse weights are below this: in hundredths, they're kept in 32 bits (with room for rounding).
pub const MAX_SPARSE_WEIGHT: f32 = 40_000_000.0;

/// A sparse field's terms and weights as the field's text: "term\tweight" lines.
pub fn encode_sparse<S: AsRef<str>>(terms: &[(S, f32)]) -> String {
    let mut out = String::new();
    for (t, w) in terms {
        let t = t.as_ref();
        if t.is_empty() || t.contains(['\t', '\n']) || !w.is_finite() || *w <= 0.0 || *w >= MAX_SPARSE_WEIGHT {
            continue;
        }
        // In hundredths, as they're kept: integer formatting, several times faster than a float's
        // shortest form (which had been a fifth of indexing a sparse field).
        let q = ((*w * SPARSE_SCALE).round() as u64).max(1);
        use std::fmt::Write;
        let _ = writeln!(out, "{t}\t{}.{:02}", q / 100, q % 100);
    }
    out
}

/// The terms and quantized weights (≥ 1) of a sparse field's text.
pub fn decode_sparse(text: &str) -> impl Iterator<Item = (&str, u32)> {
    text.lines().filter_map(|l| {
        let (t, w) = l.split_once('\t')?;
        // Hundredths as `encode_sparse` writes them, else any number; a weight that isn't one, or
        // is outside 32 bits of hundredths, leaves the term out.
        let fixed = w.split_once('.').filter(|(_, f)| f.len() == 2 && f.bytes().all(|b| b.is_ascii_digit())).map(|(i, f)| i.parse::<u32>().ok()?.checked_mul(100)?.checked_add(f.parse::<u32>().ok()?));
        let q = match fixed {
            Some(q) => q?,
            None => {
                let w = w.parse::<f32>().ok()?;
                if !(w.is_finite() && w > 0.0 && w < MAX_SPARSE_WEIGHT) {
                    return None;
                }
                (w * SPARSE_SCALE).round() as u32
            }
        };
        Some((t, q.max(1)))
    })
}

impl Analyzer {
    /// Words as written (lowercased): no stemming, no stopwords.
    pub const PLAIN: Analyzer = Analyzer { language: None, stopwords: false, sparse: false, revision: 1 };

    /// Weighted terms (from a learned sparse model), scored by dot product with the query's.
    pub const SPARSE: Analyzer = Analyzer { language: None, stopwords: false, sparse: true, revision: 1 };

    /// Stemmed in `language`, without its stopwords (where hinny has a list).
    pub fn language(language: Language) -> Analyzer {
        Analyzer { language: Some(language), stopwords: true, sparse: false, revision: Self::current_revision(language) }
    }

    fn current_revision(language: Language) -> u8 {
        if language == Language::English { 2 } else { 1 }
    }

    /// The same analysis apart from the stopword list's revision: an index made by an earlier
    /// version, asked for with today's default.
    pub fn same_but_revision(&self, other: &Analyzer) -> bool {
        Analyzer { revision: 1, ..*self } == Analyzer { revision: 1, ..*other }
    }

    pub fn english() -> Analyzer {
        Analyzer::language(Language::English)
    }

    pub fn german() -> Analyzer {
        Analyzer::language(Language::German)
    }

    pub fn without_stopwords(mut self, drop: bool) -> Analyzer {
        self.stopwords = drop;
        self.revision = match self.language {
            Some(l) if drop => Self::current_revision(l),
            _ => 1,
        };
        self
    }

    pub fn is_plain(&self) -> bool {
        self.language.is_none()
    }

    /// How it's written in a manifest or schema: "plain", "english@2" (Snowball's stopwords),
    /// "english" (Lucene's, revision 1), "english/keep" (stopwords kept).
    pub fn code(&self) -> String {
        if self.sparse {
            return "sparse".into();
        }
        match self.language {
            None => "plain".into(),
            Some(l) if self.stopwords && self.revision > 1 => format!("{}@{}", l.name(), self.revision),
            Some(l) if self.stopwords => l.name().into(),
            Some(l) => format!("{}/keep", l.name()),
        }
    }

    pub fn from_code(code: &str) -> Option<Analyzer> {
        if code == "plain" {
            return Some(Analyzer::PLAIN);
        }
        if code == "sparse" {
            return Some(Analyzer::SPARSE);
        }
        let (code, keep) = match code.strip_suffix("/keep") {
            Some(n) => (n, true),
            None => (code, false),
        };
        let (name, revision) = match code.split_once('@') {
            Some((n, r)) => (n, r.parse().ok()?),
            None => (code, 1),
        };
        let language = Language::from_name(name)?;
        // With stopwords kept, the list doesn't matter: revision 1.
        let revision = if keep { 1 } else { revision };
        Some(Analyzer { language: Some(language), stopwords: !keep, sparse: false, revision })
    }

    /// The term for one of the tokenizer's words, or none for a stopword (and in a sparse field,
    /// which text queries don't search).
    pub fn term<'a>(&self, word: &'a str) -> Option<Cow<'a, str>> {
        if self.sparse {
            return None;
        }
        match self.language {
            None => Some(Cow::Borrowed(word)),
            Some(l) => {
                if self.stopwords && l.stopwords(self.revision).binary_search(&word).is_ok() {
                    return None;
                }
                Some(Stemmer::create(l.algorithm()).stem(word))
            }
        }
    }

    /// Calls `f` with each term of `text` and the byte range of the word it came from.
    pub fn for_each_span(&self, text: &str, mut f: impl FnMut(&str, Range<usize>)) {
        if self.sparse {
            return;
        }
        match self.language {
            None => for_each_token_span(text, f),
            Some(l) => {
                let stemmer = Stemmer::create(l.algorithm());
                let stop = if self.stopwords { l.stopwords(self.revision) } else { &[] };
                let mut stem = String::new();
                for_each_token_span(text, |w, r| {
                    if stop.binary_search(&w).is_err() {
                        if stem_cached(l, &stemmer, w, &mut stem) {
                            f(&stem, r);
                        } else {
                            f(w, r);
                        }
                    }
                });
            }
        }
    }

    /// Calls `f` with each term of `text`.
    pub fn for_each(&self, text: &str, mut f: impl FnMut(&str)) {
        self.for_each_span(text, |t, _| f(t));
    }

    /// The terms of `text`, in order.
    pub fn terms(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        self.for_each(text, |t| out.push(t.to_string()));
        out
    }
}

// Stems this thread has made, per language (word → its stem, or none when that's the word).
// Words are Zipf-distributed, so most of a corpus' tokens are a few thousand words: looking one
// up is a fraction of running Snowball on it (stemming had been a third of indexing an English
// field). Bounded by starting over when full, at 32,768 words (about 2 MB) a thread: snippets
// stem on query threads too.
thread_local! {
    static STEMS: RefCell<Vec<HashMap<Box<str>, Option<Box<str>>, crate::hash::FxBuild>>> = const { RefCell::new(Vec::new()) };
}
const STEMS_CACHED: usize = 1 << 15;

/// Puts `word`'s stem in `out` and says so, or says it's `word` itself.
fn stem_cached(l: Language, stemmer: &Stemmer, word: &str, out: &mut String) -> bool {
    STEMS.with(|c| {
        let mut c = c.borrow_mut();
        if c.is_empty() {
            c.resize_with(LANGUAGES.len(), Default::default);
        }
        let m = &mut c[l as usize];
        if let Some(stem) = m.get(word) {
            return stem.as_deref().map(|s| out.replace_range(.., s)).is_some();
        }
        if m.len() >= STEMS_CACHED {
            m.clear();
        }
        let stem = match stemmer.stem(word) {
            Cow::Borrowed(_) => None,
            Cow::Owned(s) => Some(s.into_boxed_str()),
        };
        let changed = stem.as_deref().map(|s| out.replace_range(.., s)).is_some();
        m.insert(word.into(), stem);
        changed
    })
}

/// Snowball's English stopwords, sorted, without its contractions ("don't" is two tokenizer words,
/// which the list doesn't stop).
const ENGLISH: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
];

/// Lucene's English stopwords (`EnglishAnalyzer`), sorted: indexes made before 0.4 (revision 1).
const LUCENE_ENGLISH: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
];

/// Common German function words (after Snowball's German stop list), lowercased, sorted.
const GERMAN: &[&str] = &[
    "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am", "an", "ander", "andere", "anderem", "anderen", "anderer", "anderes", "anderm", "andern", "anderr", "anders", "auch", "auf", "aus", "bei", "bin", "bis", "bist", "da", "damit", "dann", "das", "dass", "dasselbe", "dazu", "daß", "dein", "deine", "deinem", "deinen", "deiner", "deines", "dem", "demselben", "den", "denn", "denselben", "der", "derer", "derselbe", "derselben", "des", "desselben", "dessen", "dich", "die", "dies", "diese", "dieselbe", "dieselben", "diesem", "diesen", "dieser", "dieses", "dir", "doch", "dort", "du", "durch", "ein", "eine", "einem", "einen", "einer", "eines", "einig", "einige", "einigem", "einigen", "einiger", "einiges", "einmal", "er", "es", "etwas", "euch", "euer", "eure", "eurem", "euren", "eurer", "eures", "für", "gegen", "gewesen", "hab", "habe", "haben", "hat", "hatte", "hatten", "hier", "hin", "hinter", "ich", "ihm", "ihn", "ihnen", "ihr", "ihre", "ihrem", "ihren", "ihrer", "ihres", "im", "in", "indem", "ins", "ist", "jede", "jedem", "jeden", "jeder", "jedes", "jene", "jenem", "jenen", "jener", "jenes", "jetzt", "kann", "kein", "keine", "keinem", "keinen", "keiner", "keines", "können", "könnte", "machen", "man", "manche", "manchem", "manchen", "mancher", "manches", "mein", "meine", "meinem", "meinen", "meiner", "meines", "mich", "mir", "mit", "muss", "musste", "nach", "nicht", "nichts", "noch", "nun", "nur", "ob", "oder", "ohne", "sehr", "sein", "seine", "seinem", "seinen", "seiner", "seines", "selbst", "sich", "sie", "sind", "so", "solche", "solchem", "solchen", "solcher", "solches", "soll", "sollte", "sondern", "sonst", "um", "und", "uns", "unser", "unsere", "unserem", "unseren", "unserer", "unseres", "unter", "viel", "vom", "von", "vor", "war", "waren", "warst", "was", "weg", "weil", "weiter", "welche", "welchem", "welchen", "welcher", "welches", "wenn", "werde", "werden", "wie", "wieder", "will", "wir", "wird", "wirst", "wo", "wollen", "wollte", "während", "würde", "würden", "zu", "zum", "zur", "zwar", "zwischen", "über",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_weights_round_trip_within_their_range_and_are_left_out_outside_it() {
        let text = encode_sparse(&[("a", 0.004), ("b", 1.5), ("c", 39_999_000.0), ("d", 40_000_000.0), ("e", 5e7), ("f", f32::INFINITY), ("g", -1.0), ("h", f32::NAN)]);
        let got: Vec<(&str, u32)> = decode_sparse(&text).collect();
        assert_eq!(got.iter().map(|(t, _)| *t).collect::<Vec<_>>(), ["a", "b", "c"]);
        assert_eq!(got[0].1, 1, "the smallest kept weight is a hundredth");
        assert_eq!(got[1].1, 150);
        assert!(got[2].1.abs_diff(3_999_900_000) < 1000, "{}", got[2].1); // within an f32 of it
        // Written by hand: out of range, or not numbers, rather than wrapped.
        for line in ["x\t50000000.00", "x\t42949673.00", "x\t1.x0", "x\tnan", "x\tinf", "x\t-3", "x\t1e12"] {
            assert_eq!(decode_sparse(line).count(), 0, "{line:?}");
        }
        assert_eq!(decode_sparse("x\t2.5").collect::<Vec<_>>(), [("x", 250)]);
        assert_eq!(decode_sparse("x\t42949672.95").collect::<Vec<_>>(), [("x", u32::MAX)]);
    }

    #[test]
    fn stopword_lists_are_sorted_and_lowercase() {
        for list in [ENGLISH, LUCENE_ENGLISH, GERMAN] {
            assert!(list.windows(2).all(|w| w[0] < w[1]), "sorted, without duplicates");
            assert!(list.iter().all(|w| crate::tokenize::tokens(w) == [w.to_string()]), "single tokenizer words");
        }
    }

    #[test]
    fn english_and_german() {
        let en = Analyzer::english();
        assert_eq!(en.terms("The taxes were filed, and taxing was avoided"), ["tax", "file", "tax", "avoid"]);
        // Indexes made before 0.4 keep Lucene's list.
        let lucene = Analyzer::from_code("english").unwrap();
        assert_eq!(lucene.revision, 1);
        assert_eq!(lucene.terms("The taxes were filed"), ["tax", "were", "file"]);
        assert!(en.same_but_revision(&lucene) && en != lucene);
        assert_eq!(en.term("the"), None);
        assert_eq!(en.term("running").as_deref(), Some("run"));
        let de = Analyzer::german();
        assert_eq!(de.terms("Die Steuererklärungen wurden für das Jahr eingereicht"), ["steuererklar", "wurd", "jahr", "eingereicht"]);
        assert_eq!(Analyzer::PLAIN.terms("The Taxes"), ["the", "taxes"]);
        // Spans point at the words as written.
        let text = "Taxes, filed";
        let mut spans = Vec::new();
        en.for_each_span(text, |t, r| spans.push((t.to_string(), &text[r])));
        assert_eq!(spans, [("tax".to_string(), "Taxes"), ("file".to_string(), "filed")]);
    }

    #[test]
    fn codes_round_trip() {
        assert_eq!(Analyzer::english().code(), "english@2");
        for a in [Analyzer::PLAIN, Analyzer::SPARSE, Analyzer::english(), Analyzer::from_code("english").unwrap(), Analyzer::english().without_stopwords(false), Analyzer::german().without_stopwords(false), Analyzer::language(Language::French)] {
            assert_eq!(Analyzer::from_code(&a.code()), Some(a), "{}", a.code());
        }
        assert_eq!(Analyzer::from_code("klingon"), None);
    }
}
