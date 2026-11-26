//! Words for the text index: runs of letters and digits, lowercased, under 40 bytes as written
//! (longer runs are hashes, base64 and the like, never typed). Exactly Tantivy's default analyzer
//! (SimpleTokenizer, RemoveLongFilter(40) on the original text, then LowerCaser), so the two engines
//! index the same words.

/// A word is kept if its original bytes are fewer than this.
pub const TOKEN_LIMIT: usize = 40;

/// Calls `f` with each word of `text`, lowercased into a reused buffer.
#[inline]
pub fn for_each_token(text: &str, mut f: impl FnMut(&str)) {
    for_each_token_span(text, |w, _| f(w));
}

/// `for_each_token`, with each word's byte range in `text` (as written, before lowercasing).
pub fn for_each_token_span(text: &str, mut f: impl FnMut(&str, std::ops::Range<usize>)) {
    let mut word = String::with_capacity(64);
    // Where the word started, as written (lowercasing can change its length).
    let mut start = 0;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // ASCII fast path: nearly all of mail and documents.
        let b = bytes[i];
        if b < 0x80 {
            if b.is_ascii_alphanumeric() {
                if word.is_empty() {
                    start = i;
                }
                word.push(b.to_ascii_lowercase() as char);
            } else {
                flush(&mut word, start..i, &mut f);
            }
            i += 1;
            continue;
        }
        let c = text[i..].chars().next().unwrap();
        if c.is_alphanumeric() {
            if word.is_empty() {
                start = i;
            }
            word.extend(c.to_lowercase());
        } else {
            flush(&mut word, start..i, &mut f);
        }
        i += c.len_utf8();
    }
    flush(&mut word, start..bytes.len(), &mut f);
}

#[inline]
fn flush(word: &mut String, span: std::ops::Range<usize>, f: &mut impl FnMut(&str, std::ops::Range<usize>)) {
    if !word.is_empty() {
        if span.len() < TOKEN_LIMIT {
            f(word, span);
        }
        word.clear();
    }
}

pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for_each_token(text, |t| out.push(t.to_string()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_and_lowercases() {
        assert_eq!(tokens("Re: Tax-Return 2025 (filed)!"), ["re", "tax", "return", "2025", "filed"]);
        assert_eq!(tokens("Größe ÉTÉ naïve"), ["größe", "été", "naïve"]);
        assert_eq!(tokens(&"x".repeat(40)), Vec::<String>::new());
        assert_eq!(tokens(&"x".repeat(39)).len(), 1);
        assert_eq!(tokens("日本語 テキスト"), ["日本語", "テキスト"]);
    }

    #[test]
    fn spans_are_the_words_as_written() {
        let text = "Re: Tax-Return ÉTÉ 2025, naïve  日本語!";
        let mut spans = Vec::new();
        for_each_token_span(text, |w, r| spans.push((w.to_string(), text[r].to_string())));
        assert_eq!(spans.iter().map(|(w, _)| w.as_str()).collect::<Vec<_>>(), tokens(text));
        assert_eq!(spans.iter().map(|(_, o)| o.as_str()).collect::<Vec<_>>(), ["Re", "Tax", "Return", "ÉTÉ", "2025", "naïve", "日本語"]);
    }
}
