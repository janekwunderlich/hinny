//! Snippets: the part of a text that best matches a query, with the query's words marked.
//!
//! Words match as the index matches them (the same tokenizer), so a highlighted word is one that
//! counted for the score. Each word weighs what it's worth (`snippet_weighted`: the index's idf,
//! so "to" counts for little and a rare name for a lot): the window of about `size` characters
//! whose distinct words weigh the most wins (then the most matches, then the earliest). Words
//! that weigh under a tenth of the heaviest aren't marked. The window is cut at word boundaries,
//! with "…" where it was cut, and runs of whitespace become one space.

use std::collections::HashMap;
use std::ops::Range;

use crate::analysis::Analyzer;
use crate::tokenize::tokens;

#[derive(Clone, Debug, PartialEq)]
pub struct Snippet {
    pub text: String,
    /// Byte ranges in `text` of the words that matched.
    pub highlights: Vec<Range<usize>>,
}

/// How well a snippet matched: the weight of the distinct query words in it, then matches.
pub type Score = (f32, usize);

/// The best snippet of `text` for `query`, about `size` characters long, every word weighing the
/// same; and its score.
pub fn snippet(text: &str, query: &str, size: usize) -> (Snippet, Score) {
    let mut weights: Vec<(String, f32)> = Vec::new();
    for t in tokens(query) {
        if !weights.iter().any(|(w, _)| *w == t) {
            weights.push((t, 1.0));
        }
    }
    snippet_weighted(text, &weights, size)
}

/// The best snippet of `text` for these words and weights, words as written (lowercased).
pub fn snippet_weighted(text: &str, words: &[(String, f32)], size: usize) -> (Snippet, Score) {
    snippet_analyzed(text, words, size, Analyzer::PLAIN)
}

/// The best snippet of `text` for these terms and weights (as `Index::term_weights` gives for a
/// field), the text's words analyzed as the field's are: a term "tax" marks "taxes" and "taxed".
pub fn snippet_analyzed(text: &str, words: &[(String, f32)], size: usize, analyzer: Analyzer) -> (Snippet, Score) {
    let heaviest = words.iter().map(|(_, w)| *w).fold(0.0, f32::max);
    // Looked up once per word of the text: FxHash, not SipHash.
    let mut terms: HashMap<String, usize, crate::hash::FxBuild> = HashMap::default();
    let mut weight: Vec<f32> = Vec::new();
    for (t, w) in words {
        if *w > 0.0 && *w >= heaviest * 0.1 && !terms.contains_key(t) {
            terms.insert(t.clone(), weight.len());
            weight.push(*w);
        }
    }
    // Matches with their start in characters (counted as we go).
    let mut matches: Vec<(Range<usize>, usize, usize)> = Vec::new(); // (bytes, term, char start)
    let (mut chars, mut at) = (0, 0);
    if !terms.is_empty() {
        analyzer.for_each_span(text, |w, r| {
            if let Some(&t) = terms.get(w) {
                chars += text[at..r.start].chars().count();
                at = r.start;
                matches.push((r, t, chars));
            }
        });
    }
    let char_len = |r: &Range<usize>| text[r.clone()].chars().count();
    // The run of matches spanning at most `size` characters with the most distinct terms.
    let (mut best, mut best_score): (Range<usize>, Score) = (0..0, (0.0, 0));
    let mut counts = vec![0usize; terms.len()];
    let (mut covered, mut lo) = (0.0f32, 0);
    for hi in 0..matches.len() {
        if counts[matches[hi].1] == 0 {
            covered += weight[matches[hi].1];
        }
        counts[matches[hi].1] += 1;
        while matches[hi].2 + char_len(&matches[hi].0) - matches[lo].2 > size.max(1) && lo < hi {
            counts[matches[lo].1] -= 1;
            if counts[matches[lo].1] == 0 {
                covered -= weight[matches[lo].1];
            }
            lo += 1;
        }
        let score = (covered, hi + 1 - lo);
        if score.0 > best_score.0 + 1e-6 || ((score.0 - best_score.0).abs() <= 1e-6 && score.1 > best_score.1) {
            best_score = score;
            best = lo..hi + 1;
        }
    }
    // The window in characters: the matches, with some context before them.
    let (start_char, span) = match (matches.get(best.start), best.end.checked_sub(1).and_then(|i| matches.get(i))) {
        (Some(first), Some(last)) => (first.2, last.2 + char_len(&last.0) - first.2),
        _ => (0, 0),
    };
    let lead = (size.saturating_sub(span) / 2).min(size / 4);
    let from_char = start_char.saturating_sub(lead);
    let (mut from, mut to) = (text.len(), text.len());
    for (n, (i, _)) in text.char_indices().enumerate() {
        if n == from_char {
            from = i;
        }
        if n == from_char + size {
            to = i;
            break;
        }
    }
    if size == 0 {
        to = from;
    }
    // Whole words: start after a space near the cut, end before one.
    if from > 0 {
        if let Some(p) = text[from..to].char_indices().take(25).find(|(_, c)| c.is_whitespace()).map(|(p, _)| p) {
            if matches.get(best.start).is_none_or(|m| from + p < m.0.start) {
                from += p;
            }
        }
    }
    if to < text.len() {
        if let Some(p) = text[from..to].char_indices().rev().take(25).find(|(_, c)| c.is_whitespace()).map(|(p, _)| p) {
            if matches.get(best.end.wrapping_sub(1)).is_none_or(|m| from + p >= m.0.end) {
                to = from + p;
            }
        }
    }
    // The text, whitespace collapsed, with the matches' positions carried over.
    let inside: Vec<&(Range<usize>, usize, usize)> = matches.iter().filter(|m| m.0.start >= from && m.0.end <= to).collect();
    let mut out = String::with_capacity(to - from + 8);
    let mut highlights = Vec::with_capacity(inside.len());
    if from > 0 {
        out.push('…');
    }
    let mut next = inside.iter().peekable();
    let mut open: Option<usize> = None;
    for (i, c) in text[from..to].char_indices() {
        let i = from + i;
        if open.is_none() && next.peek().is_some_and(|m| m.0.start == i) {
            open = Some(out.len());
        }
        if c.is_whitespace() {
            if !out.is_empty() && !out.ends_with(' ') && !out.ends_with('…') {
                out.push(' ');
            }
        } else {
            out.push(c);
        }
        if let (Some(s), Some(m)) = (open, next.peek()) {
            if i + c.len_utf8() == m.0.end {
                highlights.push(s..out.len());
                open = None;
                next.next();
            }
        }
    }
    while out.ends_with(' ') {
        out.pop();
    }
    if to < text.len() && !out.is_empty() {
        out.push('…');
    }
    let distinct: std::collections::HashSet<usize> = inside.iter().map(|m| m.1).collect();
    (Snippet { text: out, highlights }, (distinct.iter().map(|&t| weight[t]).sum(), inside.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marked(s: &Snippet) -> String {
        let mut out = String::new();
        let mut at = 0;
        for h in &s.highlights {
            out.push_str(&s.text[at..h.start]);
            out.push('[');
            out.push_str(&s.text[h.clone()]);
            out.push(']');
            at = h.end;
        }
        out.push_str(&s.text[at..]);
        out
    }

    #[test]
    fn short_text_whole_with_marks() {
        let (s, score) = snippet("The  apartment\n\nlease, signed in March.", "apartment lease", 200);
        assert_eq!(marked(&s), "The [apartment] [lease], signed in March.");
        assert_eq!(score, (2.0, 2));
    }

    #[test]
    fn picks_the_window_with_the_most_distinct_words() {
        let filler = "nothing to see here. ".repeat(30);
        let text = format!("{filler}the lease was mentioned once. {filler}The apartment lease for Elm Street, signed in March. {filler}");
        let (s, score) = snippet(&text, "apartment lease March", 120);
        assert_eq!(score, (3.0, 3));
        let m = marked(&s);
        assert!(m.contains("[apartment] [lease]") && m.contains("[March]"), "{m}");
        assert!(m.starts_with('…') && m.ends_with('…'), "{m}");
        assert!(s.text.chars().count() <= 124, "{}", s.text.chars().count());
    }

    #[test]
    fn matches_as_the_index_does() {
        // Case and accents as written; the tokenizer lowercases.
        let (s, _) = snippet("ÉTÉ in Größe: TAX-return filed", "été größe tax return", 100);
        assert_eq!(marked(&s), "[ÉTÉ] in [Größe]: [TAX]-[return] filed");
        // Part of a word isn't a match.
        let (s, score) = snippet("taxes and taxation", "tax", 100);
        assert_eq!((s.highlights.len(), score), (0, (0.0, 0)));
    }

    #[test]
    fn no_match_is_the_start() {
        let text = "First words of a long text. ".repeat(20);
        let (s, score) = snippet(&text, "absent", 40);
        assert!(s.text.starts_with("First words") && s.text.ends_with('…'), "{}", s.text);
        assert_eq!(score, (0.0, 0));
        let (s, _) = snippet(&text, "", 40);
        assert!(s.text.starts_with("First words"));
    }

    #[test]
    fn highlights_point_at_the_words() {
        let text = "Zahlung für die Wohnung: die Miete für März, 1.200 €, ist eingegangen. ".repeat(5);
        let (s, _) = snippet(&text, "miete märz", 60);
        assert!(!s.highlights.is_empty());
        for h in &s.highlights {
            assert!(["miete", "märz"].contains(&s.text[h.clone()].to_lowercase().as_str()), "{:?}", &s.text[h.clone()]);
        }
    }

    #[test]
    fn rare_words_outweigh_common_ones() {
        let filler = "and so it goes on. ".repeat(20);
        let text = format!("how to do it and what to do: {filler} the rerank threshold is set by the bound. {filler}");
        let words = vec![("how".to_string(), 0.2), ("to".to_string(), 0.1), ("rerank".to_string(), 4.0), ("bound".to_string(), 3.0)];
        let (s, score) = snippet_weighted(&text, &words, 80);
        let marked: Vec<&str> = s.highlights.iter().map(|r| &s.text[r.clone()]).collect();
        assert_eq!(marked, ["rerank", "bound"]);
        assert!((score.0 - 7.0).abs() < 1e-6);
        // Under a tenth of the heaviest: neither chosen for nor marked.
        let (s, _) = snippet_weighted("how to rerank", &words, 80);
        assert_eq!(s.highlights.iter().map(|r| &s.text[r.clone()]).collect::<Vec<_>>(), ["rerank"]);
    }
}
