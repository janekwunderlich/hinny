//! Which of a column's codes are in given runs: a bit per code.
//!
//! What a condition on an attribute comes down to (attrs.rs, `Layout::in_runs`): the codes of a
//! thousand documents, one, two, four or eight bytes each, and a few runs of codes that pass. A
//! code is in a run if its distance from the run's first, as an unsigned number, is at most the
//! run's length: one comparison for both ends. Sixty-four codes make a word of bits; with NEON
//! or AVX2 the comparisons' results are gathered into it sixteen or thirty-two at a time, which
//! is where a loop written code by code spends its time (a condition on a million one-byte
//! codes, with the bits it makes: 0.21 ms as the compiler vectorizes that loop, 0.06 ms here).

/// The most runs `in_runs` takes.
pub const FEW: usize = 8;

/// Sets `out` to a bit per code of `codes` (`width` bytes each: 1, 2, 4 or 8): whether it is in
/// one of `runs`, each a first code and how many more follow it. At most `FEW` runs; a first
/// code fits the width; `out` has a word per 64 codes.
pub fn in_runs(codes: &[u8], width: usize, runs: &[(u64, u64)], out: &mut [u64]) {
    debug_assert!(runs.len() <= FEW && matches!(width, 1 | 2 | 4 | 8) && codes.len() % width == 0 && out.len() == (codes.len() / width).div_ceil(64));
    // A run that goes on past the largest code of this width ends there: the distance from its
    // first code wraps around for codes under it, and must stay above the run's length.
    let most = if width == 8 { u64::MAX } else { (1u64 << (width * 8)) - 1 };
    let mut fitted = [(0u64, 0u64); FEW];
    for (f, &(lo, span)) in fitted.iter_mut().zip(runs) {
        debug_assert!(lo <= most);
        *f = (lo, span.min(most - lo.min(most)));
    }
    let runs = &fitted[..runs.len().min(FEW)];
    // Whole words with the processor's instructions, what's left one code at a time.
    #[allow(unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "aarch64")]
    if width < 8 {
        // SAFETY: NEON is part of aarch64.
        done = unsafe { neon::whole_words(codes, width, runs, out) };
    }
    #[cfg(target_arch = "x86_64")]
    if width < 8 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2.
        done = unsafe { avx2::whole_words(codes, width, runs, out) };
    }
    one_by_one(&codes[done * 64 * width..], width, runs, &mut out[done..]);
}

fn one_by_one(codes: &[u8], width: usize, runs: &[(u64, u64)], out: &mut [u64]) {
    macro_rules! compare {
        ($t:ty) => {{
            const W: usize = std::mem::size_of::<$t>();
            for (w, chunk) in out.iter_mut().zip(codes.chunks(64 * W)) {
                let mut bits = 0u64;
                for &(lo, span) in runs {
                    let (lo, span) = (lo as $t, span as $t);
                    bits |= chunk.chunks_exact(W).enumerate().fold(0u64, |bits, (i, c)| bits | ((<$t>::from_le_bytes(c.try_into().unwrap()).wrapping_sub(lo) <= span) as u64) << i);
                }
                *w = bits;
            }
        }};
    }
    match width {
        1 => compare!(u8),
        2 => compare!(u16),
        4 => compare!(u32),
        _ => compare!(u64),
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// The words all of whose 64 codes are there; returns how many.
    pub unsafe fn whole_words(codes: &[u8], width: usize, runs: &[(u64, u64)], out: &mut [u64]) -> usize {
        let words = codes.len() / (64 * width);
        // SAFETY: word `w` reads codes[w * 64 * width..(w + 1) * 64 * width], which is there.
        unsafe {
            let p = codes.as_ptr();
            // A byte per code that is in a run, sixteen codes at a time: all ones or nothing.
            macro_rules! sixteen {
                (u8, $at:expr) => {{
                    let c = vld1q_u8($at);
                    runs.iter().fold(vdupq_n_u8(0), |m, &(lo, span)| vorrq_u8(m, vcleq_u8(vsubq_u8(c, vdupq_n_u8(lo as u8)), vdupq_n_u8(span as u8))))
                }};
                (u16, $at:expr) => {{
                    let (a, b) = (vld1q_u16($at as *const u16), vld1q_u16($at.add(16) as *const u16));
                    let (a, b) = runs.iter().fold((vdupq_n_u16(0), vdupq_n_u16(0)), |(ma, mb), &(lo, span)| {
                        let (lo, span) = (vdupq_n_u16(lo as u16), vdupq_n_u16(span as u16));
                        (vorrq_u16(ma, vcleq_u16(vsubq_u16(a, lo), span)), vorrq_u16(mb, vcleq_u16(vsubq_u16(b, lo), span)))
                    });
                    // One byte of each.
                    vuzp1q_u8(vreinterpretq_u8_u16(a), vreinterpretq_u8_u16(b))
                }};
                (u32, $at:expr) => {{
                    let c = [vld1q_u32($at as *const u32), vld1q_u32($at.add(16) as *const u32), vld1q_u32($at.add(32) as *const u32), vld1q_u32($at.add(48) as *const u32)];
                    let m = runs.iter().fold([vdupq_n_u32(0); 4], |m, &(lo, span)| {
                        let (lo, span) = (vdupq_n_u32(lo as u32), vdupq_n_u32(span as u32));
                        [0, 1, 2, 3].map(|i| vorrq_u32(m[i], vcleq_u32(vsubq_u32(c[i], lo), span)))
                    });
                    let (a, b) = (vuzp1q_u16(vreinterpretq_u16_u32(m[0]), vreinterpretq_u16_u32(m[1])), vuzp1q_u16(vreinterpretq_u16_u32(m[2]), vreinterpretq_u16_u32(m[3])));
                    vuzp1q_u8(vreinterpretq_u8_u16(a), vreinterpretq_u8_u16(b))
                }};
            }
            // Sixty-four such bytes as a word of bits: each byte keeps the bit of its place in
            // its eight, and neighbours are added until eight are one byte (as simdjson does).
            let place = vld1q_u8([1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128].as_ptr());
            macro_rules! words {
                ($t:tt, $bytes:expr) => {
                    for (w, word) in out[..words].iter_mut().enumerate() {
                        let at = p.add(w * 64 * $bytes);
                        let m = [sixteen!($t, at), sixteen!($t, at.add(16 * $bytes)), sixteen!($t, at.add(32 * $bytes)), sixteen!($t, at.add(48 * $bytes))].map(|m| vandq_u8(m, place));
                        let sum = vpaddq_u8(vpaddq_u8(m[0], m[1]), vpaddq_u8(m[2], m[3]));
                        *word = vgetq_lane_u64::<0>(vreinterpretq_u64_u8(vpaddq_u8(sum, sum)));
                    }
                };
            }
            match width {
                1 => words!(u8, 1),
                2 => words!(u16, 2),
                _ => words!(u32, 4),
            }
        }
        words
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// The words all of whose 64 codes are there; returns how many.
    #[target_feature(enable = "avx2")]
    pub unsafe fn whole_words(codes: &[u8], width: usize, runs: &[(u64, u64)], out: &mut [u64]) -> usize {
        let words = codes.len() / (64 * width);
        // SAFETY: word `w` reads codes[w * 64 * width..(w + 1) * 64 * width], which is there.
        unsafe {
            let p = codes.as_ptr();
            let load = |at: *const u8| _mm256_loadu_si256(at as *const __m256i);
            // There is no unsigned comparison: a distance is at most the length if the smaller
            // of the two is the distance.
            match width {
                1 => {
                    // Thirty-two codes, a bit each.
                    let half = |at: *const u8| {
                        let c = load(at);
                        let m = runs.iter().fold(_mm256_setzero_si256(), |m, &(lo, span)| {
                            let x = _mm256_sub_epi8(c, _mm256_set1_epi8(lo as u8 as i8));
                            _mm256_or_si256(m, _mm256_cmpeq_epi8(_mm256_min_epu8(x, _mm256_set1_epi8(span as u8 as i8)), x))
                        });
                        _mm256_movemask_epi8(m) as u32 as u64
                    };
                    for (w, word) in out[..words].iter_mut().enumerate() {
                        let at = p.add(w * 64);
                        *word = half(at) | half(at.add(32)) << 32;
                    }
                }
                2 => {
                    let half = |at: *const u8| {
                        let (a, b) = (load(at), load(at.add(32)));
                        let (a, b) = runs.iter().fold((_mm256_setzero_si256(), _mm256_setzero_si256()), |(ma, mb), &(lo, span)| {
                            let (lo, span) = (_mm256_set1_epi16(lo as u16 as i16), _mm256_set1_epi16(span as u16 as i16));
                            let (xa, xb) = (_mm256_sub_epi16(a, lo), _mm256_sub_epi16(b, lo));
                            (_mm256_or_si256(ma, _mm256_cmpeq_epi16(_mm256_min_epu16(xa, span), xa)), _mm256_or_si256(mb, _mm256_cmpeq_epi16(_mm256_min_epu16(xb, span), xb)))
                        });
                        // A byte of each; packing works on each half of a register, so the
                        // quarters come out as a's, b's, a's, b's and are put back in order.
                        let bytes = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packs_epi16(a, b));
                        _mm256_movemask_epi8(bytes) as u32 as u64
                    };
                    for (w, word) in out[..words].iter_mut().enumerate() {
                        let at = p.add(w * 128);
                        *word = half(at) | half(at.add(64)) << 32;
                    }
                }
                _ => {
                    // Eight codes, a bit each.
                    let eight = |at: *const u8| {
                        let c = load(at);
                        let m = runs.iter().fold(_mm256_setzero_si256(), |m, &(lo, span)| {
                            let x = _mm256_sub_epi32(c, _mm256_set1_epi32(lo as u32 as i32));
                            _mm256_or_si256(m, _mm256_cmpeq_epi32(_mm256_min_epu32(x, _mm256_set1_epi32(span as u32 as i32)), x))
                        });
                        _mm256_movemask_ps(_mm256_castsi256_ps(m)) as u32 as u64
                    };
                    for (w, word) in out[..words].iter_mut().enumerate() {
                        let at = p.add(w * 256);
                        *word = (0..8).fold(0u64, |bits, i| bits | eight(at.add(i * 32)) << (i * 8));
                    }
                }
            }
        }
        words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bit_per_code_in_a_run() {
        let mut rng = 0xA076_1D64_78BD_642Fu64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for width in [1usize, 2, 4, 8] {
            let most = if width == 8 { u64::MAX } else { (1u64 << (width * 8)) - 1 };
            for round in 0..200 {
                // Codes of few values (so runs have some), of any value, and the largest there are.
                let n = [0usize, 1, 63, 64, 65, 128, 1000, 1024][round % 8];
                let few = round % 3 == 0;
                let values: Vec<u64> = (0..n).map(|_| if few { next() % 20 } else if next() % 16 == 0 { most - next() % 3 } else { next() & most }).collect();
                let codes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()[..width].to_vec()).collect();
                // Runs as `Layout::in_runs` hands them over: starting where a code can be,
                // reaching as far as they like.
                let runs: Vec<(u64, u64)> = (0..round % (FEW + 1))
                    .map(|_| {
                        let lo = if few { next() % 20 } else { next() & most };
                        (lo, [0, 1, 5, next() % 1000, most, u64::MAX - lo][(next() % 6) as usize].min(u64::MAX - lo))
                    })
                    .collect();
                let mut out = vec![u64::MAX; n.div_ceil(64)];
                in_runs(&codes, width, &runs, &mut out);
                for (d, v) in values.iter().enumerate() {
                    let want = runs.iter().any(|&(lo, span)| *v >= lo && v - lo <= span);
                    assert_eq!(out[d / 64] >> (d % 64) & 1 == 1, want, "width {width}, round {round}, code {d} = {v}, runs {runs:?}");
                }
                // And nothing past the last code.
                if n % 64 != 0 {
                    assert_eq!(out[n / 64] >> (n % 64), 0, "width {width}, round {round}");
                }
            }
        }
    }
}
