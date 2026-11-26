//! Folding names for matching: lowercase, no accents ("Résumé" → "resume"), so what's typed matches
//! however the file was named. Beyond lowercasing: ß → ss, ligatures (ﬁ → fi), æ → ae, full-width
//! letters and digits → ASCII, final sigma → σ, hiragana → katakana. ASCII names (nearly all) take
//! a byte-wise fast path.

pub fn fold(name: &[u8]) -> Vec<u8> {
    if name.is_ascii() {
        return name.to_ascii_lowercase();
    }
    let mut out = Vec::with_capacity(name.len());
    let mut buf = [0u8; 4];
    for c in String::from_utf8_lossy(name).chars() {
        // Combining marks: APFS keeps names as typed, and decomposed names carry accents this way.
        if ('\u{0300}'..='\u{036F}').contains(&c) {
            continue;
        }
        // Full-width forms (Ｆｉｌｅ２０２５) are ASCII in another width.
        let c = if ('\u{FF01}'..='\u{FF5E}').contains(&c) { char::from_u32(c as u32 - 0xFEE0).unwrap_or(c) } else { c };
        for lower in c.to_lowercase() {
            if let Some(expanded) = expand(lower) {
                out.extend_from_slice(expanded.as_bytes());
                continue;
            }
            let base = strip_accent(lower);
            out.extend_from_slice(base.encode_utf8(&mut buf).as_bytes());
        }
    }
    out
}

/// Letters that fold to several (case folding and the common transliterations).
fn expand(c: char) -> Option<&'static str> {
    Some(match c {
        'ß' | 'ẞ' => "ss",
        'æ' => "ae",
        'œ' => "oe",
        'þ' => "th",
        'ﬀ' => "ff",
        'ﬁ' => "fi",
        'ﬂ' => "fl",
        'ﬃ' => "ffi",
        'ﬄ' => "ffl",
        'ﬅ' | 'ﬆ' => "st",
        'ſ' => "s",
        'ς' => "σ",
        _ => return None,
    })
}

/// Precomposed Latin letters to their base letter (the common ones in names: Latin-1 and Extended-A).
fn strip_accent(c: char) -> char {
    match c {
        'à'..='å' | 'ā' | 'ă' | 'ą' => 'a',
        'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => 'c',
        'ď' | 'đ' => 'd',
        'è'..='ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => 'e',
        'ĝ' | 'ğ' | 'ġ' | 'ģ' => 'g',
        'ĥ' | 'ħ' => 'h',
        'ì'..='ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' => 'i',
        'ĵ' => 'j',
        'ķ' => 'k',
        'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => 'l',
        'ñ' | 'ń' | 'ņ' | 'ň' => 'n',
        'ò'..='ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => 'o',
        'ŕ' | 'ŗ' | 'ř' => 'r',
        'ś' | 'ŝ' | 'ş' | 'š' => 's',
        'ţ' | 'ť' | 'ŧ' => 't',
        'ù'..='ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => 'u',
        'ŵ' => 'w',
        'ý' | 'ÿ' | 'ŷ' => 'y',
        'ź' | 'ż' | 'ž' => 'z',
        // Hiragana and katakana are one script for search (ふぁいる finds ファイル).
        '\u{3041}'..='\u{3096}' => char::from_u32(c as u32 + 0x60).unwrap_or(c),
        _ => c,
    }
}

#[cfg(test)]
mod tests {
    use super::fold;

    #[test]
    fn folds_case_and_accents() {
        assert_eq!(fold(b"Lease_2025.PDF"), b"lease_2025.pdf");
        assert_eq!(fold("Résumé".as_bytes()), b"resume");
        // Decomposed: e + combining acute.
        assert_eq!(fold("Re\u{301}sume\u{301}".as_bytes()), b"resume");
        assert_eq!(fold("Übersicht".as_bytes()), b"ubersicht");
        assert_eq!(fold("Straße".as_bytes()), b"strasse");
        assert_eq!(fold("ﬁnal Æsir".as_bytes()), b"final aesir");
        assert_eq!(fold("Ｆｉｌｅ２０２５".as_bytes()), b"file2025");
        assert_eq!(fold("ふぁいる".as_bytes()), fold("ファイル".as_bytes()));
    }
}
