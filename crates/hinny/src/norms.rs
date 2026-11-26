//! Field lengths in one byte: exact up to 23 words, then 4 significant bits (Lucene's SmallFloat
//! `intToByte4`, the table Tantivy uses too). BM25 only needs the length roughly, and a byte per
//! document per field keeps the lengths of a million passages in 1 MB per field.

pub static TABLE: std::sync::LazyLock<[u32; 256]> = std::sync::LazyLock::new(|| std::array::from_fn(|id| byte4_to_int(id as u32)));

const FREE: u32 = 24;

fn int4_to_long(i: u32) -> u64 {
    let bits = (i & 7) as u64;
    let shift = (i >> 3) as i32 - 1;
    if shift == -1 { bits } else { (bits | 8) << shift }
}

fn byte4_to_int(b: u32) -> u32 {
    if b < FREE { b } else { (FREE as u64 + int4_to_long(b - FREE)).min(u32::MAX as u64) as u32 }
}

/// The id of the largest table length not above `len`.
pub fn id(len: u32) -> u8 {
    match TABLE.binary_search(&len) {
        Ok(i) => i as u8,
        Err(i) => (i - 1) as u8,
    }
}

/// BM25's length normalization for each id: k1 × (1 − b + b × len / average).
pub fn tf_cache(average: f32) -> [f32; 256] {
    std::array::from_fn(|id| crate::K1 * (1.0 - crate::B + crate::B * TABLE[id] as f32 / average))
}

#[cfg(test)]
mod tests {
    #[test]
    fn table_is_lucenes() {
        let t = &*super::TABLE;
        assert_eq!(&t[..26], &(0..26).collect::<Vec<u32>>()[..]);
        assert!(t.windows(2).all(|w| w[0] < w[1]));
        // 40, 42, 44, …: 41 rounds down.
        assert_eq!(t[super::id(41) as usize], 40);
        assert_eq!(t[super::id(37) as usize], 37);
    }
}
