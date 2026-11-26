//! 128 integers at a time, each in the fewest bits the largest needs: the SIMD-BP128 codec (Lemire
//! & Boytsov, "Decoding billions of integers per second through vectorization", 2015) in its
//! vertical layout. Four lanes: value `i` goes to lane `i % 4`, and each lane packs its 32 values
//! into `width` 32-bit words, interleaved with the other lanes' words. Packing and unpacking then
//! work on four values per instruction (one NEON register), with no branches inside a block; the
//! width is a const parameter, so each of the 33 widths compiles to its own unrolled loop.
//!
//! A block of 128 values at width `w` takes `16 × w` bytes.

pub const BLOCK: usize = 128;

/// Bits needed for the largest value.
pub fn width(values: &[u32]) -> u8 {
    let max = values.iter().fold(0, |a, &b| a | b);
    (32 - max.leading_zeros()) as u8
}

/// Packs `values` (up to 128; missing ones count as 0) at `width` bits, appending `16 × width` bytes.
pub fn pack(values: &[u32], width: u8, out: &mut Vec<u8>) {
    let mut block = [0u32; BLOCK];
    block[..values.len()].copy_from_slice(values);
    let mut words = [0u32; 4 * 32];
    let n = 4 * width as usize;
    seq!(width, pack_w, &block, &mut words[..n]);
    for w in &words[..n] {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

/// Unpacks a block packed at `width` from `bytes` (`16 × width` of them) into `out`.
#[inline]
pub fn unpack(bytes: &[u8], width: u8, out: &mut [u32; BLOCK]) {
    let n = 4 * width as usize;
    debug_assert!(bytes.len() >= 4 * n);
    // The data is 4-byte aligned in the file (blocks are multiples of 16 bytes from an aligned start).
    let words = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, n) };
    seq!(width, unpack_w, words, out);
}

macro_rules! seq {
    ($width:expr, $f:ident, $($arg:expr),*) => {
        match $width {
            0 => $f::<0>($($arg),*), 1 => $f::<1>($($arg),*), 2 => $f::<2>($($arg),*), 3 => $f::<3>($($arg),*),
            4 => $f::<4>($($arg),*), 5 => $f::<5>($($arg),*), 6 => $f::<6>($($arg),*), 7 => $f::<7>($($arg),*),
            8 => $f::<8>($($arg),*), 9 => $f::<9>($($arg),*), 10 => $f::<10>($($arg),*), 11 => $f::<11>($($arg),*),
            12 => $f::<12>($($arg),*), 13 => $f::<13>($($arg),*), 14 => $f::<14>($($arg),*), 15 => $f::<15>($($arg),*),
            16 => $f::<16>($($arg),*), 17 => $f::<17>($($arg),*), 18 => $f::<18>($($arg),*), 19 => $f::<19>($($arg),*),
            20 => $f::<20>($($arg),*), 21 => $f::<21>($($arg),*), 22 => $f::<22>($($arg),*), 23 => $f::<23>($($arg),*),
            24 => $f::<24>($($arg),*), 25 => $f::<25>($($arg),*), 26 => $f::<26>($($arg),*), 27 => $f::<27>($($arg),*),
            28 => $f::<28>($($arg),*), 29 => $f::<29>($($arg),*), 30 => $f::<30>($($arg),*), 31 => $f::<31>($($arg),*),
            _ => $f::<32>($($arg),*),
        }
    };
}
use seq;

#[inline(always)]
fn mask<const W: u32>() -> u32 {
    if W == 32 { u32::MAX } else { (1u32 << W) - 1 }
}

fn pack_w<const W: u32>(input: &[u32; BLOCK], out: &mut [u32]) {
    if W == 0 {
        return;
    }
    let mut acc = [0u32; 4];
    let mut bits = 0u32;
    let mut o = 0;
    for i in 0..32 {
        let v: [u32; 4] = std::array::from_fn(|l| input[i * 4 + l] & mask::<W>());
        for l in 0..4 {
            acc[l] |= v[l] << bits;
        }
        bits += W;
        if bits >= 32 {
            out[o..o + 4].copy_from_slice(&acc);
            o += 4;
            bits -= 32;
            for l in 0..4 {
                acc[l] = if bits > 0 { v[l] >> (W - bits) } else { 0 };
            }
        }
    }
}

#[inline(always)]
fn unpack_w<const W: u32>(words: &[u32], out: &mut [u32; BLOCK]) {
    if W == 0 {
        *out = [0; BLOCK];
        return;
    }
    let mut bits = 0u32;
    let mut w = 0;
    let mut cur: [u32; 4] = [words[0], words[1], words[2], words[3]];
    for i in 0..32 {
        let mut v = [0u32; 4];
        for l in 0..4 {
            v[l] = cur[l] >> bits;
        }
        bits += W;
        if bits >= 32 {
            bits -= 32;
            w += 4;
            if w < 4 * W as usize {
                cur = [words[w], words[w + 1], words[w + 2], words[w + 3]];
                if bits > 0 {
                    for l in 0..4 {
                        v[l] |= cur[l] << (W - bits);
                    }
                }
            }
        }
        for l in 0..4 {
            out[i * 4 + l] = v[l] & mask::<W>();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_width() {
        let mut seed = 0x9E37_79B9u32;
        for w in 0..=32u8 {
            for len in [1usize, 77, 128] {
                let values: Vec<u32> = (0..len)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        if w == 0 { 0 } else if w == 32 { seed } else { seed & ((1 << w) - 1) }
                    })
                    .collect();
                assert!(width(&values) <= w);
                let mut bytes = Vec::new();
                pack(&values, w, &mut bytes);
                assert_eq!(bytes.len(), 16 * w as usize);
                // Aligned copy, as in the file.
                let aligned: Vec<u32> = bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                let raw = unsafe { std::slice::from_raw_parts(aligned.as_ptr() as *const u8, bytes.len()) };
                let mut out = [0u32; BLOCK];
                unpack(raw, w, &mut out);
                assert_eq!(&out[..len], &values[..], "width {w}, len {len}");
            }
        }
    }
}
