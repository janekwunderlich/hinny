//! Where a string occurs in many bytes, whatever the case of its ASCII letters.
//!
//! A column's distinct strings are stored one after another (attrs.rs, `Layout`), so "contains"
//! is one pass over them rather than a test per string. Sixteen places at a time: the bytes
//! there and the bytes where an occurrence's last byte would be are compared with the string's
//! first and last byte, and only where both agree is the rest compared (after Wojciech Muła,
//! "SIMD-friendly algorithms for substring searching").

/// Calls `each` with every place in `hay` where `needle` starts, in order, case of ASCII letters
/// aside. `needle` is ASCII without capitals and not empty.
pub fn find_without_case(hay: &[u8], needle: &[u8], mut each: impl FnMut(usize)) {
    debug_assert!(!needle.is_empty() && needle.iter().all(|b| b.is_ascii() && !b.is_ascii_uppercase()));
    let m = needle.len();
    if m == 0 || hay.len() < m {
        return;
    }
    // SAFETY: NEON is part of aarch64.
    #[cfg(target_arch = "aarch64")]
    let mut at = unsafe { neon(hay, needle, &mut each) };
    // SAFETY: SSE2 is part of x86-64.
    #[cfg(target_arch = "x86_64")]
    let mut at = unsafe { sse2(hay, needle, &mut each) };
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let mut at = 0;
    // What's left, and everything where there are no such instructions.
    let (first, any) = (needle[0], letter(needle[0]));
    while at + m <= hay.len() {
        if hay[at] | any == first && hay[at..at + m].eq_ignore_ascii_case(needle) {
            each(at);
        }
        at += 1;
    }
}

/// The bit that tells a letter's two cases apart, if `b` is a letter: with it set, both compare
/// equal to the small one. Nothing for other bytes, which have one form.
fn letter(b: u8) -> u8 {
    if b.is_ascii_lowercase() { 0x20 } else { 0 }
}

/// The places before the last 16 + needle bytes; returns where it stopped.
#[cfg(target_arch = "aarch64")]
unsafe fn neon(hay: &[u8], needle: &[u8], each: &mut impl FnMut(usize)) -> usize {
    use std::arch::aarch64::*;
    let m = needle.len();
    let (first, last) = (needle[0], needle[m - 1]);
    let mut at = 0;
    // SAFETY: sixteen places at `at` read hay[at..at + 16] and hay[at + m - 1..at + m + 15], and
    // the loops stop while that is inside `hay`.
    unsafe {
        let (first, last, any_first, any_last) = (vdupq_n_u8(first), vdupq_n_u8(last), vdupq_n_u8(letter(first)), vdupq_n_u8(letter(last)));
        let both = |at: usize| {
            let a = vld1q_u8(hay.as_ptr().add(at));
            let b = vld1q_u8(hay.as_ptr().add(at + m - 1));
            vandq_u8(vceqq_u8(vorrq_u8(a, any_first), first), vceqq_u8(vorrq_u8(b, any_last), last))
        };
        let mut sixteen = |at: usize, both: uint8x16_t| {
            // Four bits a place.
            let mut bits = vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(both))));
            while bits != 0 {
                let i = bits.trailing_zeros() as usize / 4;
                if hay[at + i..at + i + m].eq_ignore_ascii_case(needle) {
                    each(at + i);
                }
                bits &= !(0xF << (i * 4));
            }
        };
        // Sixty-four places a round: most rounds no place agrees, and that is one test.
        while at + m + 63 <= hay.len() {
            let found = [both(at), both(at + 16), both(at + 32), both(at + 48)];
            if vmaxvq_u8(vorrq_u8(vorrq_u8(found[0], found[1]), vorrq_u8(found[2], found[3]))) != 0 {
                for (i, f) in found.into_iter().enumerate() {
                    sixteen(at + i * 16, f);
                }
            }
            at += 64;
        }
        while at + m + 15 <= hay.len() {
            sixteen(at, both(at));
            at += 16;
        }
    }
    at
}

/// The places before the last 16 + needle bytes; returns where it stopped.
#[cfg(target_arch = "x86_64")]
unsafe fn sse2(hay: &[u8], needle: &[u8], each: &mut impl FnMut(usize)) -> usize {
    use std::arch::x86_64::*;
    let m = needle.len();
    let (first, last) = (needle[0], needle[m - 1]);
    let mut at = 0;
    // SAFETY: sixteen places at `at` read hay[at..at + 16] and hay[at + m - 1..at + m + 15], and
    // the loops stop while that is inside `hay`.
    unsafe {
        let (first, last, any_first, any_last) = (_mm_set1_epi8(first as i8), _mm_set1_epi8(last as i8), _mm_set1_epi8(letter(first) as i8), _mm_set1_epi8(letter(last) as i8));
        let both = |at: usize| {
            let a = _mm_loadu_si128(hay.as_ptr().add(at) as *const __m128i);
            let b = _mm_loadu_si128(hay.as_ptr().add(at + m - 1) as *const __m128i);
            _mm_and_si128(_mm_cmpeq_epi8(_mm_or_si128(a, any_first), first), _mm_cmpeq_epi8(_mm_or_si128(b, any_last), last))
        };
        let mut sixteen = |at: usize, both: __m128i| {
            let mut bits = _mm_movemask_epi8(both) as u32;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                if hay[at + i..at + i + m].eq_ignore_ascii_case(needle) {
                    each(at + i);
                }
                bits &= bits - 1;
            }
        };
        // Sixty-four places a round: most rounds no place agrees, and that is one test.
        while at + m + 63 <= hay.len() {
            let found = [both(at), both(at + 16), both(at + 32), both(at + 48)];
            if _mm_movemask_epi8(_mm_or_si128(_mm_or_si128(found[0], found[1]), _mm_or_si128(found[2], found[3]))) != 0 {
                for (i, f) in found.into_iter().enumerate() {
                    sixteen(at + i * 16, f);
                }
            }
            at += 64;
        }
        while at + m + 15 <= hay.len() {
            sixteen(at, both(at));
            at += 16;
        }
    }
    at
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every place, found one by one.
    fn slowly(hay: &[u8], needle: &[u8]) -> Vec<usize> {
        (0..(hay.len() + 1).saturating_sub(needle.len())).filter(|&at| hay[at..at + needle.len()].eq_ignore_ascii_case(needle)).collect()
    }

    #[test]
    fn finds_what_a_test_of_every_place_finds() {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        // Few different bytes, so there are occurrences, some overlapping, some cut off by the end;
        // capitals, a byte next to a letter's (`@` is not `` ` ``), and bytes outside ASCII.
        let letters = b"abAB.@`_1\xC3\xA4";
        for round in 0..400 {
            let hay: Vec<u8> = (0..next() % 300).map(|_| letters[(next() % letters.len() as u64) as usize]).collect();
            for needle in [&b"a"[..], b"ab", b"aba", b"b.a", b"@", b"`a", b"_1", b"abab", b"aaaaaaaaaaaaaaaaaaaa", b"ab.ab@ab_ab1abab"] {
                let mut found = Vec::new();
                find_without_case(&hay, needle, |at| found.push(at));
                assert_eq!(found, slowly(&hay, needle), "round {round}, {:?} in {:?}", String::from_utf8_lossy(needle), String::from_utf8_lossy(&hay));
            }
        }
        let mut found = Vec::new();
        find_without_case(b"", b"a", |at| found.push(at));
        find_without_case(b"a", b"ab", |at| found.push(at));
        assert!(found.is_empty());
        find_without_case(b"Report, REPORT and report.", b"report", |at| found.push(at));
        assert_eq!(found, [0, 8, 19]);
    }
}
