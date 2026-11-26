//! Vectors: nearest-neighbour search with 1-bit RaBitQ codes over clusters, reranked at full
//! precision where the codes can't decide.
//!
//! The design follows turbopuffer's ANN v3 for the reasons that hold on one machine:
//! - **Search is memory-bandwidth bound**, so each vector is scanned as a 1-bit code (RaBitQ, Gao &
//!   Long, SIGMOD 2024): 32× smaller than float32, 64 bytes for 512 dimensions. Full-precision
//!   vectors stay in the memory-mapped file and are read only for the few candidates the codes
//!   can't rule out.
//! - **RaBitQ's error bound decides the rerank.** A code gives an unbiased estimate of the
//!   distance and a bound on its error, so a candidate whose lower bound can't beat the current
//!   k-th exact distance is dropped without reading its vector. Recall stays near exact.
//! - **Clusters keep nearby vectors together** (centroid-based, like SPFresh): a query probes the
//!   nearest clusters, streaming each one's codes contiguously. Clusters are built per immutable
//!   segment (at checkpoint and merge), so updates never rebalance anything in place: the log and
//!   merges do that.
//! - **Filters apply during the search** (native filtering): a candidate is checked before it can
//!   enter the top k, and probing continues until k matches are found, so a selective filter
//!   turns into an exact scan of its matches instead of losing recall.
//!
//! A namespace compares its vectors by one `Metric`: cosine (vectors normalized on the way in; a
//! score is the cosine, 1 − d²/2), Euclidean distance (vectors as they come; a score is −d²) or
//! dot product (as they come; the score is the product). Cosine and Euclidean distance are one
//! search: the codes estimate squared Euclidean distances and the rerank computes them. Dot
//! product is the same scan with one exact term per vector, over lists that are of directions
//! (`search_lists`).
//!
//! The rotation is a randomized Hadamard transform (three rounds of random signs and a fast
//! Walsh–Hadamard transform over the dimensions padded to a power of two): a random orthogonal
//! rotation in O(D log D), as RaBitQ implementations use, instead of a dense D×D matrix.

use std::collections::BinaryHeap;

use rayon::prelude::*;

/// Error-bound constant. RaBitQ's paper uses 1.9 for an exact query; the query here is quantized
/// to 4 bits, which adds a little error: at 1.9, 1.8% of bounds fell below the true distance in
/// the test below, at 2.5 under 0.4% (and a violation only matters for a true near neighbour).
/// Those that do matter set a ceiling no amount of probing lifts: a million vectors in a
/// thousand topics stopped at recall 0.997 with every code scanned. At 3.0 they stop at 0.999,
/// for a third more vectors read at full precision (0.107 → 0.115 ms a search there), and the
/// calibrated budget is then smaller where it had been making up for the bound.
const EPSILON: f32 = 3.0;
/// Below this many vectors a segment is one cluster.
const MIN_CLUSTERED: usize = 4096;

/// How a namespace's vectors are compared: what "nearest" means in it, fixed once it holds
/// vectors. A search keeps the k vectors with the least key, and a score is what the key says,
/// higher for nearer whatever the metric (a hybrid query scales scores, and couldn't otherwise).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Metric {
    /// The cosine of the angle between two vectors. Vectors and queries are normalized, the
    /// key is their squared distance d², and a score is the cosine, 1 − d²/2.
    #[default]
    Cosine,
    /// Euclidean distance, between vectors as they come. The key is d² and a score is −d²:
    /// 0 for the query itself, lower the farther.
    Euclidean,
    /// The dot product ⟨q, x⟩ of vectors as they come, which is the score; the key is −⟨q, x⟩.
    /// A long vector scores higher than a short one in the same direction.
    Dot,
}

impl Metric {
    /// Its name, as schemas and manifests write it.
    pub fn name(self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::Euclidean => "euclidean",
            Metric::Dot => "dot",
        }
    }

    pub fn from_name(name: &str) -> Option<Metric> {
        [Metric::Cosine, Metric::Euclidean, Metric::Dot].into_iter().find(|m| m.name() == name)
    }

    /// Its number, as a segment's vector header keeps it.
    pub fn code(self) -> u32 {
        self as u32
    }

    pub fn from_code(code: u32) -> Option<Metric> {
        [Metric::Cosine, Metric::Euclidean, Metric::Dot].into_iter().find(|m| m.code() == code)
    }

    /// Makes `v` what this metric stores and searches with: normalized for cosine, as it is
    /// otherwise. False for a vector it can't compare: one that isn't finite, and for cosine
    /// one of no length.
    pub fn prepare(self, v: &mut [f32]) -> bool {
        match self {
            Metric::Cosine => normalize(v),
            Metric::Euclidean | Metric::Dot => v.iter().all(|x| x.is_finite()),
        }
    }

    /// The score a key stands for (see the variants).
    #[inline]
    pub fn score(self, key: f32) -> f32 {
        match self {
            Metric::Cosine => 1.0 - key / 2.0,
            Metric::Euclidean | Metric::Dot => -key,
        }
    }

    /// The key a score came from.
    #[inline]
    pub fn key(self, score: f32) -> f32 {
        match self {
            Metric::Cosine => 2.0 - 2.0 * score,
            Metric::Euclidean | Metric::Dot => -score,
        }
    }
}

/// The largest size a component of a stored vector can have: rows are kept in half precision,
/// which ends here. (A cosine vector is normalized first, so its components never get near.)
pub const LARGEST_COMPONENT: f32 = 65504.0;

pub fn normalize(v: &mut [f32]) -> bool {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !(n > 0.0) || !n.is_finite() {
        return false;
    }
    v.iter_mut().for_each(|x| *x /= n);
    true
}

#[inline]
pub fn dist2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if x86::avx2() {
        // SAFETY: the processor has AVX2 and FMA.
        return unsafe { x86::dist2(a, b) };
    }
    // Sixteen accumulators, which the compiler keeps in four NEON registers: with fewer, each
    // add waits on the last one (two registers were half as fast at 256 dimensions).
    let mut acc = [0f32; 16];
    let (ac, bc) = (a.chunks_exact(16), b.chunks_exact(16));
    let (ar, br) = (ac.remainder(), bc.remainder());
    for (x, y) in ac.zip(bc) {
        for l in 0..16 {
            let d = x[l] - y[l];
            acc[l] += d * d;
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ar.iter().zip(br) {
        let d = x - y;
        s += d * d;
    }
    s
}

// MARK: Half precision

/// Full-precision rows as stored: f32 (segments before version 7) or f16 (since). Half precision
/// halves a segment (its full-precision rows were 92% of it at 256 dimensions) and costs nothing
/// measurable: rows are only read to rerank candidates (0.4% of FiQA's vectors per query), and
/// f16's 11 significant bits put distances within 0.1% of exact.
#[derive(Clone, Copy)]
pub enum Rows<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
    /// Rows not yet written, as a segment's build has them: (document, vector) per row.
    Refs(&'a [(u32, VectorRef<'a>)]),
}

/// One full-precision vector where it's kept: f32, or f16 (a memtable's, a segment's since version
/// 7). A build reads them in place, so a flush or merge never holds every vector decoded.
#[derive(Clone, Copy, Debug)]
pub enum VectorRef<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
}

impl<'a> VectorRef<'a> {
    pub fn len(self) -> usize {
        match self {
            VectorRef::F32(v) => v.len(),
            VectorRef::F16(h) => h.len(),
        }
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn to_f32(self) -> std::borrow::Cow<'a, [f32]> {
        match self {
            VectorRef::F32(v) => std::borrow::Cow::Borrowed(v),
            VectorRef::F16(h) => std::borrow::Cow::Owned(h.iter().map(|&x| f16_to_f32(x)).collect()),
        }
    }

    pub fn to_f16(self) -> std::borrow::Cow<'a, [u16]> {
        match self {
            VectorRef::F16(h) => std::borrow::Cow::Borrowed(h),
            VectorRef::F32(v) => std::borrow::Cow::Owned(to_f16(v)),
        }
    }

    #[inline]
    pub fn dist2(self, q: &[f32]) -> f32 {
        match self {
            VectorRef::F32(v) => dist2(q, v),
            VectorRef::F16(h) => dist2_f16(q, h),
        }
    }

    /// ⟨q, v⟩.
    #[inline]
    pub fn dot(self, q: &[f32]) -> f32 {
        match self {
            VectorRef::F32(v) => dot(q, v),
            VectorRef::F16(h) => dot_f16(q, h),
        }
    }
}

impl Rows<'_> {
    /// Squared distance from `q` to row `row` (of `dims`).
    #[inline]
    pub fn dist2(&self, q: &[f32], row: usize, dims: usize) -> f32 {
        match self {
            Rows::F32(f) => dist2(q, &f[row * dims..(row + 1) * dims]),
            Rows::F16(h) => dist2_f16(q, &h[row * dims..(row + 1) * dims]),
            Rows::Refs(v) => v[row].1.dist2(q),
        }
    }

    /// ⟨q, row `row`⟩.
    #[inline]
    pub fn dot(&self, q: &[f32], row: usize, dims: usize) -> f32 {
        match self {
            Rows::F32(f) => dot(q, &f[row * dims..(row + 1) * dims]),
            Rows::F16(h) => dot_f16(q, &h[row * dims..(row + 1) * dims]),
            Rows::Refs(v) => v[row].1.dot(q),
        }
    }

    /// What `metric` orders row `row` by against `q`, exactly (`Metric`: the squared distance,
    /// or for dot product the product, negated).
    #[inline]
    pub fn key(&self, metric: Metric, q: &[f32], row: usize, dims: usize) -> f32 {
        match metric {
            Metric::Dot => -self.dot(q, row, dims),
            Metric::Cosine | Metric::Euclidean => self.dist2(q, row, dims),
        }
    }

    /// Asks the processor to start fetching row `row`: a search knows which rows it will read
    /// a few candidates ahead, and each is a miss in a file far larger than the caches (reading
    /// them cold had been half of a search over a million vectors).
    #[inline]
    pub fn prefetch(&self, row: usize, dims: usize) {
        let (start, bytes) = match self {
            Rows::F32(f) => (f[row * dims..].as_ptr() as *const u8, dims * 4),
            Rows::F16(h) => (h[row * dims..].as_ptr() as *const u8, dims * 2),
            Rows::Refs(_) => return,
        };
        // Every cache line of the row (64 bytes each; 128 on Apple's processors, where every
        // other request is for a line already asked for and costs nothing).
        for at in (0..bytes).step_by(64) {
            // SAFETY: a hint that reads nothing: the address is within the row, and a prefetch of
            // any address is allowed.
            unsafe { prefetch(start.add(at)) };
        }
    }

    pub fn row(&self, row: usize, dims: usize) -> Vec<f32> {
        match self {
            Rows::F32(f) => f[row * dims..(row + 1) * dims].to_vec(),
            Rows::F16(h) => h[row * dims..(row + 1) * dims].iter().map(|&x| f16_to_f32(x)).collect(),
            Rows::Refs(v) => v[row].1.to_f32().to_vec(),
        }
    }
}

/// A hint to bring the cache line at `p` into the nearest cache. No effect on what the program
/// computes; nothing where there's no such instruction to give.
#[inline(always)]
unsafe fn prefetch(p: *const u8) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `prfm` reads no memory and faults on no address.
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p, options(nostack, preserves_flags, readonly));
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch reads no memory and faults on no address (SSE, part of x86-64).
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p as *const i8);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = p;
}

/// `v` in half precision (rounded to nearest, ties to even).
pub fn to_f16(v: &[f32]) -> Vec<u16> {
    let mut out = vec![0u16; v.len()];
    #[cfg_attr(not(any(target_arch = "aarch64", target_arch = "x86_64")), allow(unused_mut))]
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        while i + 4 <= v.len() {
            // SAFETY: NEON is part of aarch64; four floats in, four halves out, within bounds.
            unsafe { vst1_u16(out.as_mut_ptr().add(i), vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(v.as_ptr().add(i))))) };
            i += 4;
        }
    }
    #[cfg(target_arch = "x86_64")]
    if x86::avx2() {
        // SAFETY: the processor has AVX2 and F16C; `out` is as long as `v`.
        i = unsafe { x86::to_f16(v, &mut out) };
    }
    for j in i..v.len() {
        out[j] = f32_to_f16(v[j]);
    }
    out
}

#[inline]
pub fn dist2_f16(a: &[f32], b: &[u16]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n = a.len().min(b.len()) / 16 * 16;
        // SAFETY: NEON is part of aarch64; every load is within the first `n` of both slices.
        let mut s = unsafe {
            let mut acc = [vdupq_n_f32(0.0); 4];
            for i in (0..n).step_by(16) {
                for j in 0..2 {
                    let h = vreinterpretq_f16_u16(vld1q_u16(b.as_ptr().add(i + 8 * j)));
                    let (lo, hi) = (vcvt_f32_f16(vget_low_f16(h)), vcvt_high_f32_f16(h));
                    let d0 = vsubq_f32(vld1q_f32(a.as_ptr().add(i + 8 * j)), lo);
                    let d1 = vsubq_f32(vld1q_f32(a.as_ptr().add(i + 8 * j + 4)), hi);
                    acc[2 * j] = vfmaq_f32(acc[2 * j], d0, d0);
                    acc[2 * j + 1] = vfmaq_f32(acc[2 * j + 1], d1, d1);
                }
            }
            vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])))
        };
        for i in n..a.len().min(b.len()) {
            let d = a[i] - f16_to_f32(b[i]);
            s += d * d;
        }
        s
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if x86::avx2() {
            // SAFETY: the processor has AVX2, FMA and F16C.
            return unsafe { x86::dist2_f16(a, b) };
        }
        let mut acc = [0f32; 8];
        let (ac, bc) = (a.chunks_exact(8), b.chunks_exact(8));
        let (ar, br) = (ac.remainder(), bc.remainder());
        for (x, y) in ac.zip(bc) {
            for l in 0..8 {
                let d = x[l] - f16_to_f32(y[l]);
                acc[l] += d * d;
            }
        }
        acc.iter().sum::<f32>() + ar.iter().zip(br).map(|(x, &y)| (x - f16_to_f32(y)).powi(2)).sum::<f32>()
    }
}

/// ⟨a, b⟩ against a row in half precision: what a rerank computes for dot product, as `dist2_f16`
/// is for the other metrics, and laid out as it is.
#[inline]
pub fn dot_f16(a: &[f32], b: &[u16]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n = a.len().min(b.len()) / 16 * 16;
        // SAFETY: NEON is part of aarch64; every load is within the first `n` of both slices.
        let mut s = unsafe {
            let mut acc = [vdupq_n_f32(0.0); 4];
            for i in (0..n).step_by(16) {
                for j in 0..2 {
                    let h = vreinterpretq_f16_u16(vld1q_u16(b.as_ptr().add(i + 8 * j)));
                    let (lo, hi) = (vcvt_f32_f16(vget_low_f16(h)), vcvt_high_f32_f16(h));
                    acc[2 * j] = vfmaq_f32(acc[2 * j], vld1q_f32(a.as_ptr().add(i + 8 * j)), lo);
                    acc[2 * j + 1] = vfmaq_f32(acc[2 * j + 1], vld1q_f32(a.as_ptr().add(i + 8 * j + 4)), hi);
                }
            }
            vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])))
        };
        for i in n..a.len().min(b.len()) {
            s += a[i] * f16_to_f32(b[i]);
        }
        s
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if x86::avx2() {
            // SAFETY: the processor has AVX2, FMA and F16C.
            return unsafe { x86::dot_f16(a, b) };
        }
        let mut acc = [0f32; 8];
        let (ac, bc) = (a.chunks_exact(8), b.chunks_exact(8));
        let (ar, br) = (ac.remainder(), bc.remainder());
        for (x, y) in ac.zip(bc) {
            for l in 0..8 {
                acc[l] += x[l] * f16_to_f32(y[l]);
            }
        }
        acc.iter().sum::<f32>() + ar.iter().zip(br).map(|(x, &y)| x * f16_to_f32(y)).sum::<f32>()
    }
}

/// IEEE 754 binary16 to binary32, exactly (every half is a float).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        // Subnormal: normalize the mantissa.
        (0, m) => {
            let shift = m.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((m << shift) & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7fc0_0000 | (m << 13),
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// binary32 to binary16, rounded to nearest with ties to even (as the hardware conversion).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Subnormal: the implicit bit joins the mantissa, shifted into place, then rounded.
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let up = rest > mid || (rest == mid && half & 1 == 1);
        return sign | (half + up as u32) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    let up = rest > 0x1000 || (rest == 0x1000 && half & 1 == 1);
    // A carry out of the mantissa moves to the next exponent, as it should (up to infinity).
    sign | (half + up as u32) as u16
}

// MARK: Rotation

#[derive(Clone, Debug)]
pub struct Rotation {
    pub dims: usize,
    pub padded: usize,
    pub seed: u64,
    /// Per round, each dimension's random sign times 1/√padded (so each round keeps lengths).
    /// Drawn dimension by dimension, three at a time: the order segments were built with.
    signs: [Vec<f32>; 3],
}

impl Rotation {
    pub fn new(dims: usize, seed: u64) -> Rotation {
        let padded = dims.next_power_of_two().max(64);
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let scale = 1.0 / (padded as f32).sqrt();
        let mut signs: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; padded]);
        for i in 0..padded {
            for round in &mut signs {
                round[i] = if next() & 1 == 0 { scale } else { -scale };
            }
        }
        Rotation { dims, padded, seed, signs }
    }

    /// The rotated vector (length `padded`).
    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        self.apply_ref(VectorRef::F32(v))
    }

    /// `apply`, decoding a half-precision vector straight into the transform's buffer.
    pub fn apply_ref(&self, v: VectorRef<'_>) -> Vec<f32> {
        #[cfg(target_arch = "x86_64")]
        if x86::avx2() {
            // SAFETY: the processor has AVX2 and FMA.
            return unsafe { x86::apply_ref(self, v) };
        }
        self.apply_ref_portable(v)
    }

    #[inline(always)]
    fn apply_ref_portable(&self, v: VectorRef<'_>) -> Vec<f32> {
        let mut x = vec![0f32; self.padded];
        let n = v.len().min(self.dims);
        match v {
            VectorRef::F32(f) => x[..n].copy_from_slice(&f[..n]),
            VectorRef::F16(h) => x[..n].iter_mut().zip(h).for_each(|(x, &h)| *x = f16_to_f32(h)),
        }
        for signs in &self.signs {
            for (xi, s) in x.iter_mut().zip(signs) {
                *xi *= s;
            }
            fwht(&mut x);
        }
        x
    }
}

/// The fast Walsh–Hadamard transform, in place (`x.len()` a power of two, at least 4). The first
/// two stages on groups of four, the rest on halves of blocks as slices, which vectorize: with an
/// index per element they didn't, and rotating was a fifth of building a segment.
#[inline(always)]
fn fwht(x: &mut [f32]) {
    for c in x.chunks_exact_mut(4) {
        let (s0, s1, s2, s3) = (c[0] + c[1], c[0] - c[1], c[2] + c[3], c[2] - c[3]);
        c[0] = s0 + s2;
        c[1] = s1 + s3;
        c[2] = s0 - s2;
        c[3] = s1 - s3;
    }
    let mut h = 4;
    while h < x.len() {
        for block in x.chunks_exact_mut(2 * h) {
            let (lo, hi) = block.split_at_mut(h);
            for (a, b) in lo.iter_mut().zip(hi) {
                let (p, q) = (*a, *b);
                *a = p + q;
                *b = p - q;
            }
        }
        h *= 2;
    }
}

/// A vector's 1-bit code and RaBitQ's factors: its residual's length from the centroid, and
/// ⟨ō, o⟩ (how well the code represents its direction).
#[derive(Clone, Copy, Debug, Default)]
pub struct Factors {
    pub norm: f32,
    pub ip: f32,
}

/// The 16 sums of a lookup table: entry `p` is Σ u[i] over the bits i set in p.
#[inline]
fn lut16(u: &[u8; 4], out: &mut [u8]) {
    out[0] = 0;
    for p in 1..16usize {
        // The entry without p's lowest bit, plus that bit's value.
        out[p] = out[p & (p - 1)] + u[p.trailing_zeros() as usize];
    }
}

/// A segment's 1-bit codes again, in the order FastScan reads them: each cluster's codes in blocks
/// of 32 (the last one padded with zero codes); a block holds, for each group of 4 dimensions,
/// 16 bytes whose low nibble is code k's 4 bits and high nibble code k + 16's. Made from the
/// stored codes when a segment is first searched, as big as they are.
#[derive(Debug, Default)]
pub struct Blocked {
    pub bytes: Vec<u8>,
    /// Cluster c's blocks are `starts[c]..starts[c + 1]`.
    pub starts: Vec<u32>,
    /// Bytes per block: 16 per group of 4 dimensions.
    pub block: usize,
}

impl Blocked {
    pub fn new(codes: &[u64], offsets: &[u32], words: usize) -> Blocked {
        let groups = words * 16;
        let block = groups * 16;
        let clusters = offsets.len().saturating_sub(1);
        let mut starts = vec![0u32; clusters + 1];
        for c in 0..clusters {
            let n = (offsets[c + 1] - offsets[c]) as usize;
            starts[c + 1] = starts[c] + n.div_ceil(32) as u32;
        }
        let mut bytes = vec![0u8; starts[clusters] as usize * block];
        for c in 0..clusters {
            let first = offsets[c] as usize;
            let n = offsets[c + 1] as usize - first;
            for k in 0..n {
                let code = &codes[(first + k) * words..(first + k + 1) * words];
                let at = (starts[c] as usize + k / 32) * block;
                let (lane, shift) = (k % 32 % 16, if k % 32 < 16 { 0 } else { 4 });
                for g in 0..groups {
                    let nibble = (code[g / 16] >> ((g % 16) * 4)) as u8 & 15;
                    bytes[at + g * 16 + lane] |= nibble << shift;
                }
            }
        }
        Blocked { bytes, starts, block }
    }
}

/// The code at place `k` of a cluster whose blocks start at `bytes` (`Blocked`'s layout, read
/// back): a merge copies codes from the segments it reads instead of encoding their vectors again.
pub fn unblock(bytes: &[u8], block: usize, k: usize, code: &mut [u64]) {
    let at = (k / 32) * block;
    let (lane, shift) = (k % 32 % 16, if k % 32 < 16 { 0 } else { 4 });
    code.iter_mut().for_each(|w| *w = 0);
    for g in 0..block / 16 {
        let nibble = (bytes[at + g * 16 + lane] >> shift) & 15;
        code[g / 16] |= (nibble as u64) << ((g % 16) * 4);
    }
}

/// What an estimate needs of one vector besides the query, worked out once per segment (when it's
/// first searched) instead of on every estimate: 1 / ⟨o, ō⟩, the error bound's √(1 − ⟨o, ō⟩²) /
/// ⟨o, ō⟩, the residual's length, and the code's set bits. That's a division, a square root and a
/// bit count fewer per vector scanned.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Pre {
    pub inv_ip: f32,
    pub err: f32,
    pub norm: f32,
    pub ones: f32,
}

impl Pre {
    /// From a vector's factors and its code.
    pub fn of(f: Factors, code: &[u64]) -> Pre {
        Pre { inv_ip: 1.0 / f.ip, err: ((1.0 - f.ip * f.ip).max(0.0)).sqrt() / f.ip, norm: f.norm, ones: code.iter().map(|w| w.count_ones()).sum::<u32>() as f32 }
    }

    pub fn all(norms: &[f32], ips: &[f32], codes: &[u64], words: usize) -> Vec<Pre> {
        norms
            .iter()
            .zip(ips)
            .zip(codes.chunks_exact(words))
            .map(|((&norm, &ip), code)| Pre { inv_ip: 1.0 / ip, err: ((1.0 - ip * ip).max(0.0)).sqrt() / ip, norm, ones: code.iter().map(|w| w.count_ones()).sum::<u32>() as f32 })
            .collect()
    }
}

/// Encodes `x − c` (both rotated) into `words` u64s of sign bits.
pub fn encode(x: &[f32], c: &[f32], code: &mut [u64]) -> Factors {
    #[cfg(target_arch = "x86_64")]
    if x86::avx2() {
        // SAFETY: the processor has AVX2 and FMA.
        return unsafe { x86::encode(x, c, code) };
    }
    encode_portable(x, c, code)
}

#[inline(always)]
fn encode_portable(x: &[f32], c: &[f32], code: &mut [u64]) -> Factors {
    let p = x.len();
    let mut r: Vec<f32> = x.iter().zip(c).map(|(a, b)| a - b).collect();
    let norm = r.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        r.iter_mut().for_each(|v| *v /= norm);
    }
    code.iter_mut().for_each(|w| *w = 0);
    let mut abs = 0f32;
    for (i, v) in r.iter().enumerate() {
        if *v > 0.0 {
            code[i / 64] |= 1 << (i % 64);
        }
        abs += v.abs();
    }
    let ip = if norm > 0.0 { (abs / (p as f32).sqrt()).max(1e-6) } else { 1.0 };
    Factors { norm, ip }
}

/// The query against one cluster: its residual quantized to 4 bits, and the lookup tables made of it.
pub struct QueryCode {
    /// The quantized query, a value 0–15 per dimension.
    u: Vec<u8>,
    /// For `Blocked` codes: per 4 dimensions, the quantized query summed over each of the 16
    /// patterns of 4 code bits (pattern bit i is dimension 4g + i), 16 bytes a group.
    lut: Vec<u8>,
    /// The residual, kept between clusters to spare an allocation.
    r: Vec<f32>,
    low: f32,
    delta: f32,
    sum: f32,
    pub norm: f32,
    sqrt_p: f32,
    bound_scale: f32,
}

impl QueryCode {
    pub fn new(q: &[f32], c: &[f32]) -> QueryCode {
        let p = q.len();
        let mut qc = QueryCode { u: vec![0; p], lut: vec![0; 4 * p], r: vec![0.0; p], low: 0.0, delta: 0.0, sum: 0.0, norm: 0.0, sqrt_p: (p as f32).sqrt(), bound_scale: EPSILON / ((p - 1) as f32).sqrt() };
        qc.set(q, c);
        qc
    }

    /// Requantizes for another cluster. Done for every cluster probed, so it's written to
    /// vectorize: one pass for the residual and its length, one for the range, one to quantize,
    /// and the lookup tables (with a division, rounding and a branch per bit, this had cost more
    /// than estimating the cluster's codes).
    pub fn set(&mut self, q: &[f32], c: &[f32]) {
        #[cfg(target_arch = "x86_64")]
        if x86::avx2() {
            // SAFETY: the processor has AVX2 and FMA.
            return unsafe { x86::set(self, q, c) };
        }
        self.set_portable(q, c)
    }

    #[inline(always)]
    fn set_portable(&mut self, q: &[f32], c: &[f32]) {
        let r = &mut self.r;
        let mut norm2 = 0f32;
        for ((r, a), b) in r.iter_mut().zip(q).zip(c) {
            *r = a - b;
            norm2 += *r * *r;
        }
        let norm = norm2.sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 1.0 };
        let (mut low, mut high) = (f32::MAX, f32::MIN);
        for v in r.iter_mut() {
            *v *= inv;
            low = low.min(*v);
            high = high.max(*v);
        }
        let delta = ((high - low) / 15.0).max(1e-12);
        let scale = 1.0 / delta;
        let mut total = 0u32;
        for (u, v) in self.u.iter_mut().zip(r.iter()) {
            // (v − low)/delta is at least 0, so adding a half and truncating rounds it.
            let q = (((v - low) * scale + 0.5) as u32).min(15);
            *u = q as u8;
            total += q;
        }
        self.low = low;
        self.delta = delta;
        self.sum = r.len() as f32 * low + delta * total as f32;
        self.norm = norm;
        self.build_lut();
    }

    /// Σ u[i] over the code's set bits, one code at a time: what `weighted_block` computes for 32.
    fn weighted_scalar(&self, code: &[u64]) -> u32 {
        let mut sum = 0u32;
        for (w, &bits) in code.iter().enumerate() {
            let mut b = bits;
            while b != 0 {
                sum += self.u[w * 64 + b.trailing_zeros() as usize] as u32;
                b &= b - 1;
            }
        }
        sum
    }

    /// The lookup tables from `u`: entry p of group g is Σ u[4g + i] over the bits i set in p. With
    /// NEON a group's 16 entries are four masked copies of its values added up (a scalar table,
    /// 15 dependent additions a group, cost more than the scan it was for).
    fn build_lut(&mut self) {
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            // SAFETY: NEON is part of aarch64; group g reads u[4g..4g + 4] and writes
            // lut[16g..16g + 16], within their lengths (p and 4p).
            unsafe {
                let m = LUT_MASKS.map(|row| vld1q_u8(row.as_ptr()));
                for g in 0..self.u.len() / 4 {
                    let u = self.u.as_ptr().add(4 * g);
                    let a = vaddq_u8(vandq_u8(m[0], vld1q_dup_u8(u)), vandq_u8(m[1], vld1q_dup_u8(u.add(1))));
                    let b = vaddq_u8(vandq_u8(m[2], vld1q_dup_u8(u.add(2))), vandq_u8(m[3], vld1q_dup_u8(u.add(3))));
                    vst1q_u8(self.lut.as_mut_ptr().add(16 * g), vaddq_u8(a, b));
                }
                return;
            }
        }
        #[cfg(target_arch = "x86_64")]
        if x86::avx2() {
            // SAFETY: the processor has AVX2; `lut` is four times as long as `u`.
            unsafe { x86::build_lut(&self.u, &mut self.lut) };
            return;
        }
        #[allow(unreachable_code)]
        for g in 0..self.u.len() / 4 {
            let u = [self.u[4 * g], self.u[4 * g + 1], self.u[4 * g + 2], self.u[4 * g + 3]];
            lut16(&u, &mut self.lut[16 * g..16 * g + 16]);
        }
    }

    /// `weighted` for the 32 codes of one block of `Blocked` codes at once: a table lookup scores
    /// 16 of them on 4 dimensions (FastScan, as RaBitQ's authors use it), against a bit count per
    /// code, plane and word.
    #[inline]
    pub fn weighted_block(&self, block: &[u8], out: &mut [u32; 32]) {
        let groups = self.lut.len() / 16;
        // The groups are a multiple of 4 (dimensions are padded to a power of two, 64 or more).
        assert!(block.len() >= groups * 16 && groups % 4 == 0);
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            // SAFETY: NEON is part of aarch64; loads are of 16 bytes at multiples of 16 within
            // `block` and `self.lut` (both at least `groups` × 16 long), stores within `out`.
            unsafe {
                let low4 = vdupq_n_u8(0x0f);
                let mut wide = [vdupq_n_u32(0); 8];
                // Sums in 16 bits for up to WIDEN groups (a group adds at most 60), then into 32:
                // past 4,368 dimensions 16 bits would wrap.
                let (mut b, mut t) = (block.as_ptr(), self.lut.as_ptr());
                let mut left = groups / 4;
                while left > 0 {
                    let mut acc = [vdupq_n_u16(0); 4];
                    // Four groups add up in bytes (at most 4 · 60) before widening: written out,
                    // so that nothing in the loop asks how many groups are left.
                    for _ in 0..left.min(WIDEN / 4) {
                        let (b0, b1, b2, b3) = (vld1q_u8(b), vld1q_u8(b.add(16)), vld1q_u8(b.add(32)), vld1q_u8(b.add(48)));
                        let (t0, t1, t2, t3) = (vld1q_u8(t), vld1q_u8(t.add(16)), vld1q_u8(t.add(32)), vld1q_u8(t.add(48)));
                        let lo = vaddq_u8(
                            vaddq_u8(vqtbl1q_u8(t0, vandq_u8(b0, low4)), vqtbl1q_u8(t1, vandq_u8(b1, low4))),
                            vaddq_u8(vqtbl1q_u8(t2, vandq_u8(b2, low4)), vqtbl1q_u8(t3, vandq_u8(b3, low4))),
                        );
                        let hi = vaddq_u8(
                            vaddq_u8(vqtbl1q_u8(t0, vshrq_n_u8::<4>(b0)), vqtbl1q_u8(t1, vshrq_n_u8::<4>(b1))),
                            vaddq_u8(vqtbl1q_u8(t2, vshrq_n_u8::<4>(b2)), vqtbl1q_u8(t3, vshrq_n_u8::<4>(b3))),
                        );
                        acc[0] = vaddw_u8(acc[0], vget_low_u8(lo));
                        acc[1] = vaddw_high_u8(acc[1], lo);
                        acc[2] = vaddw_u8(acc[2], vget_low_u8(hi));
                        acc[3] = vaddw_high_u8(acc[3], hi);
                        (b, t) = (b.add(64), t.add(64));
                    }
                    left -= left.min(WIDEN / 4);
                    for (i, a) in acc.into_iter().enumerate() {
                        wide[2 * i] = vaddw_u16(wide[2 * i], vget_low_u16(a));
                        wide[2 * i + 1] = vaddw_high_u16(wide[2 * i + 1], a);
                    }
                }
                for (i, w) in wide.into_iter().enumerate() {
                    vst1q_u32(out.as_mut_ptr().add(i * 4), w);
                }
                return;
            }
        }
        #[cfg(target_arch = "x86_64")]
        if x86::avx2() {
            // SAFETY: the processor has AVX2; `block` and `self.lut` hold `groups` × 16 bytes.
            unsafe { x86::weighted_block(&self.lut, block, out) };
            return;
        }
        #[allow(unreachable_code)]
        {
            *out = [0; 32];
            for g in 0..groups {
                let t = &self.lut[g * 16..(g + 1) * 16];
                for (k, &b) in block[g * 16..(g + 1) * 16].iter().enumerate() {
                    out[k] += t[(b & 15) as usize] as u32;
                    out[k + 16] += t[(b >> 4) as usize] as u32;
                }
            }
        }
    }

    /// `estimate_pre` from a weighted sum (`weighted_block`'s).
    #[inline]
    pub fn estimate_weighted(&self, weighted: u32, pre: &Pre) -> (f32, f32) {
        let s = self.delta * weighted as f32 + self.low * pre.ones;
        let oq = (2.0 * s - self.sum) / self.sqrt_p;
        let est = oq * pre.inv_ip;
        let err = self.bound_scale * pre.err;
        let base = pre.norm * pre.norm + self.norm * self.norm;
        let two = 2.0 * pre.norm * self.norm;
        ((base - two * est).max(0.0), (base - two * (est + err)).max(0.0))
    }

    /// Of a block's first `pre.len()` codes (at most 32; their sums in `weighted`), those that may
    /// be nearer than `threshold`: `each(place in the block, estimate, lower bound)` for every one
    /// whose lower bound is under it. Nearly none are, once there are k results, so the bounds are
    /// worked out four at a time (`estimate_weighted`'s arithmetic in its order, so exactly its
    /// bounds), and only those that pass are estimated one by one. A code at a time, this loop had
    /// been most of a scan: a table lookup scores 32 codes in less time than 32 bounds took.
    ///
    /// `among`: a bit per code of the block that may be a result at all (a filter's; all ones
    /// without one). The others aren't looked at.
    #[inline]
    pub fn below(&self, weighted: &[u32; 32], pre: &[Pre], threshold: f32, among: u32, mut each: impl FnMut(usize, f32, f32)) {
        let n = pre.len().min(32);
        #[cfg_attr(not(any(target_arch = "aarch64", target_arch = "x86_64")), allow(unused_mut))]
        let mut i = 0;
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        while i + 4 <= n {
            let four = among >> i & 0xF;
            if four == 0 {
                i += 4;
                continue;
            }
            // SAFETY: four sums and four `Pre`s (16 floats, `repr(C)`) from `i`, within both.
            let mut passed = unsafe { self.below4(weighted.as_ptr().add(i), pre.as_ptr().add(i), threshold) } & four;
            while passed != 0 {
                let j = i + passed.trailing_zeros() as usize;
                passed &= passed - 1;
                let (est, low) = self.estimate_weighted(weighted[j], &pre[j]);
                debug_assert!(low < threshold);
                each(j, est, low);
            }
            i += 4;
        }
        for j in i..n {
            if among >> j & 1 == 0 {
                continue;
            }
            let (est, low) = self.estimate_weighted(weighted[j], &pre[j]);
            if low < threshold {
                each(j, est, low);
            }
        }
    }

    /// A bit for each of four codes whose lower bound is under `threshold`.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn below4(&self, weighted: *const u32, pre: *const Pre, threshold: f32) -> u32 {
        use std::arch::aarch64::*;
        // SAFETY: NEON is part of aarch64; the caller vouches for the four sums and four `Pre`s.
        unsafe {
            // The fields of four `Pre`s, each in a register of its own.
            let f = vld4q_f32(pre as *const f32);
            let (inv_ip, err, norm, ones) = (f.0, f.1, f.2, f.3);
            let w = vcvtq_f32_u32(vld1q_u32(weighted));
            let s = vaddq_f32(vmulq_n_f32(w, self.delta), vmulq_n_f32(ones, self.low));
            let oq = vdivq_f32(vsubq_f32(vmulq_n_f32(s, 2.0), vdupq_n_f32(self.sum)), vdupq_n_f32(self.sqrt_p));
            let est = vmulq_f32(oq, inv_ip);
            let bound = vmulq_n_f32(err, self.bound_scale);
            let base = vaddq_f32(vmulq_f32(norm, norm), vdupq_n_f32(self.norm * self.norm));
            let two = vmulq_n_f32(vmulq_n_f32(norm, 2.0), self.norm);
            let low = vmaxq_f32(vsubq_f32(base, vmulq_f32(two, vaddq_f32(est, bound))), vdupq_n_f32(0.0));
            const BITS: [u32; 4] = [1, 2, 4, 8];
            vaddvq_u32(vandq_u32(vcltq_f32(low, vdupq_n_f32(threshold)), vld1q_u32(BITS.as_ptr())))
        }
    }

    /// A bit for each of four codes whose lower bound is under `threshold` (SSE2, which every
    /// x86-64 processor has).
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn below4(&self, weighted: *const u32, pre: *const Pre, threshold: f32) -> u32 {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is part of x86-64; the caller vouches for the four sums and four `Pre`s.
        unsafe {
            // The fields of four `Pre`s, each in a register of its own (a 4 × 4 transpose).
            let p = pre as *const f32;
            let (r0, r1, r2, r3) = (_mm_loadu_ps(p), _mm_loadu_ps(p.add(4)), _mm_loadu_ps(p.add(8)), _mm_loadu_ps(p.add(12)));
            let (t0, t1, t2, t3) = (_mm_unpacklo_ps(r0, r1), _mm_unpacklo_ps(r2, r3), _mm_unpackhi_ps(r0, r1), _mm_unpackhi_ps(r2, r3));
            let (inv_ip, err, norm, ones) = (_mm_movelh_ps(t0, t1), _mm_movehl_ps(t1, t0), _mm_movelh_ps(t2, t3), _mm_movehl_ps(t3, t2));
            let w = _mm_cvtepi32_ps(_mm_loadu_si128(weighted as *const __m128i));
            let s = _mm_add_ps(_mm_mul_ps(w, _mm_set1_ps(self.delta)), _mm_mul_ps(ones, _mm_set1_ps(self.low)));
            let oq = _mm_div_ps(_mm_sub_ps(_mm_mul_ps(s, _mm_set1_ps(2.0)), _mm_set1_ps(self.sum)), _mm_set1_ps(self.sqrt_p));
            let est = _mm_mul_ps(oq, inv_ip);
            let bound = _mm_mul_ps(err, _mm_set1_ps(self.bound_scale));
            let base = _mm_add_ps(_mm_mul_ps(norm, norm), _mm_set1_ps(self.norm * self.norm));
            let two = _mm_mul_ps(_mm_mul_ps(norm, _mm_set1_ps(2.0)), _mm_set1_ps(self.norm));
            let low = _mm_max_ps(_mm_sub_ps(base, _mm_mul_ps(two, _mm_add_ps(est, bound))), _mm_setzero_ps());
            _mm_movemask_ps(_mm_cmplt_ps(low, _mm_set1_ps(threshold))) as u32
        }
    }

    /// `below` for dot product, where a code's distance is from the query's part across its
    /// list's centroid to the vector's (`search_lists`), and what is compared is that less two
    /// terms of the vector's own: its remainder's squared length (in `pre`) and `times` its
    /// length along the centroid (`along`). `each` for every code whose lower bound less those
    /// is under `threshold`, with the estimate and the bound as distances still (the caller
    /// takes the terms off). The bounds are `estimate_weighted`'s, four at a time as in
    /// `below`, so they pick the same codes as one at a time.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn below_along(&self, weighted: &[u32; 32], pre: &[Pre], along: &[f32], times: f32, threshold: f32, among: u32, mut each: impl FnMut(usize, f32, f32)) {
        let n = pre.len().min(32).min(along.len());
        #[cfg_attr(not(any(target_arch = "aarch64", target_arch = "x86_64")), allow(unused_mut))]
        let mut i = 0;
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        while i + 4 <= n {
            let four = among >> i & 0xF;
            if four == 0 {
                i += 4;
                continue;
            }
            // SAFETY: four sums, four `Pre`s (16 floats, `repr(C)`) and four lengths from `i`,
            // within all three.
            let mut passed = unsafe { self.below4_along(weighted.as_ptr().add(i), pre.as_ptr().add(i), along.as_ptr().add(i), times, threshold) } & four;
            while passed != 0 {
                let j = i + passed.trailing_zeros() as usize;
                passed &= passed - 1;
                let (est, low) = self.estimate_weighted(weighted[j], &pre[j]);
                debug_assert!(low - pre[j].norm * pre[j].norm - along[j] * times < threshold);
                each(j, est, low);
            }
            i += 4;
        }
        for j in i..n {
            if among >> j & 1 == 0 {
                continue;
            }
            let (est, low) = self.estimate_weighted(weighted[j], &pre[j]);
            if low - pre[j].norm * pre[j].norm - along[j] * times < threshold {
                each(j, est, low);
            }
        }
    }

    /// A bit for each of four codes whose lower bound, less its vector's two terms, is under
    /// `threshold` (`below4`'s arithmetic, then the terms).
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn below4_along(&self, weighted: *const u32, pre: *const Pre, along: *const f32, times: f32, threshold: f32) -> u32 {
        use std::arch::aarch64::*;
        // SAFETY: NEON is part of aarch64; the caller vouches for the four sums, `Pre`s and lengths.
        unsafe {
            let f = vld4q_f32(pre as *const f32);
            let (inv_ip, err, norm, ones) = (f.0, f.1, f.2, f.3);
            let w = vcvtq_f32_u32(vld1q_u32(weighted));
            let s = vaddq_f32(vmulq_n_f32(w, self.delta), vmulq_n_f32(ones, self.low));
            let oq = vdivq_f32(vsubq_f32(vmulq_n_f32(s, 2.0), vdupq_n_f32(self.sum)), vdupq_n_f32(self.sqrt_p));
            let est = vmulq_f32(oq, inv_ip);
            let bound = vmulq_n_f32(err, self.bound_scale);
            let own = vmulq_f32(norm, norm);
            let base = vaddq_f32(own, vdupq_n_f32(self.norm * self.norm));
            let two = vmulq_n_f32(vmulq_n_f32(norm, 2.0), self.norm);
            let low = vmaxq_f32(vsubq_f32(base, vmulq_f32(two, vaddq_f32(est, bound))), vdupq_n_f32(0.0));
            let key = vsubq_f32(vsubq_f32(low, own), vmulq_n_f32(vld1q_f32(along), times));
            const BITS: [u32; 4] = [1, 2, 4, 8];
            vaddvq_u32(vandq_u32(vcltq_f32(key, vdupq_n_f32(threshold)), vld1q_u32(BITS.as_ptr())))
        }
    }

    /// A bit for each of four codes whose lower bound, less its vector's two terms, is under
    /// `threshold` (`below4`'s arithmetic, then the terms; SSE2).
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn below4_along(&self, weighted: *const u32, pre: *const Pre, along: *const f32, times: f32, threshold: f32) -> u32 {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is part of x86-64; the caller vouches for the four sums, `Pre`s and lengths.
        unsafe {
            let p = pre as *const f32;
            let (r0, r1, r2, r3) = (_mm_loadu_ps(p), _mm_loadu_ps(p.add(4)), _mm_loadu_ps(p.add(8)), _mm_loadu_ps(p.add(12)));
            let (t0, t1, t2, t3) = (_mm_unpacklo_ps(r0, r1), _mm_unpacklo_ps(r2, r3), _mm_unpackhi_ps(r0, r1), _mm_unpackhi_ps(r2, r3));
            let (inv_ip, err, norm, ones) = (_mm_movelh_ps(t0, t1), _mm_movehl_ps(t1, t0), _mm_movelh_ps(t2, t3), _mm_movehl_ps(t3, t2));
            let w = _mm_cvtepi32_ps(_mm_loadu_si128(weighted as *const __m128i));
            let s = _mm_add_ps(_mm_mul_ps(w, _mm_set1_ps(self.delta)), _mm_mul_ps(ones, _mm_set1_ps(self.low)));
            let oq = _mm_div_ps(_mm_sub_ps(_mm_mul_ps(s, _mm_set1_ps(2.0)), _mm_set1_ps(self.sum)), _mm_set1_ps(self.sqrt_p));
            let est = _mm_mul_ps(oq, inv_ip);
            let bound = _mm_mul_ps(err, _mm_set1_ps(self.bound_scale));
            let own = _mm_mul_ps(norm, norm);
            let base = _mm_add_ps(own, _mm_set1_ps(self.norm * self.norm));
            let two = _mm_mul_ps(_mm_mul_ps(norm, _mm_set1_ps(2.0)), _mm_set1_ps(self.norm));
            let low = _mm_max_ps(_mm_sub_ps(base, _mm_mul_ps(two, _mm_add_ps(est, bound))), _mm_setzero_ps());
            let key = _mm_sub_ps(_mm_sub_ps(low, own), _mm_mul_ps(_mm_loadu_ps(along), _mm_set1_ps(times)));
            _mm_movemask_ps(_mm_cmplt_ps(key, _mm_set1_ps(threshold))) as u32
        }
    }

    /// Estimated squared distance and its lower bound, for a code with these factors.
    #[inline]
    pub fn estimate(&self, code: &[u64], f: Factors) -> (f32, f32) {
        let ones: u32 = code.iter().map(|w| w.count_ones()).sum();
        let weighted = self.weighted_scalar(code);
        // Σ over set bits of the quantized query, then ⟨ō, q⟩ = (2S − Σq) / √P.
        let s = self.delta * weighted as f32 + self.low * ones as f32;
        let oq = (2.0 * s - self.sum) / self.sqrt_p;
        let est = oq / f.ip;
        let err = self.bound_scale * ((1.0 - f.ip * f.ip).max(0.0)).sqrt() / f.ip;
        let base = f.norm * f.norm + self.norm * self.norm;
        let two = 2.0 * f.norm * self.norm;
        ((base - two * est).max(0.0), (base - two * (est + err)).max(0.0))
    }
}

/// Groups `weighted_block` sums in 16 bits before widening: each adds at most 60 (4 · 15), so up to
/// 1,092 fit; a multiple of 4.
const WIDEN: usize = 1088;

/// For a lookup table's 16 entries at once: mask i has entry p's byte set where p has bit i, so a
/// group's table is its four values, each under its mask, added up.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const LUT_MASKS: [[u8; 16]; 4] = {
    let mut m = [[0u8; 16]; 4];
    let mut p = 0;
    while p < 16 {
        let mut i = 0;
        while i < 4 {
            if p >> i & 1 == 1 {
                m[i][p] = 0xff;
            }
            i += 1;
        }
        p += 1;
    }
    m
};

// MARK: x86-64

/// The kernels above for x86-64 with AVX2, FMA and F16C (Haswell, 2013, and every Zen), chosen
/// when the program runs: a binary built for plain x86-64 has only SSE2 to compile for, and must
/// still start on a processor without them (where the portable code above runs).
#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::{LUT_MASKS, WIDEN, f16_to_f32};

    /// Loops written to vectorize, compiled again for AVX2 and FMA (inlined into these, they are):
    /// twice the width of the SSE2 a plain x86-64 build gives them.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn apply_ref(r: &super::Rotation, v: super::VectorRef<'_>) -> Vec<f32> {
        r.apply_ref_portable(v)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn encode(x: &[f32], c: &[f32], code: &mut [u64]) -> super::Factors {
        super::encode_portable(x, c, code)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn set(qc: &mut super::QueryCode, q: &[f32], c: &[f32]) {
        qc.set_portable(q, c)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemm_nt(a: &[f32], b: &[f32], dims: usize, out: &mut [f32]) {
        super::gemm_nt_portable(a, b, dims, out)
    }

    /// Whether the processor has what these need (asked once).
    #[inline]
    pub fn avx2() -> bool {
        static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *HAS.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c"))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps::<1>(s, s)))
    }

    /// Four registers of eight, as `dist2` keeps four of four.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dist2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut i = 0;
        // SAFETY (of the loads): each reads 8 floats ending at or before `n` in both slices.
        let mut s = unsafe {
            let mut acc = [_mm256_setzero_ps(); 4];
            while i + 32 <= n {
                for (j, acc) in acc.iter_mut().enumerate() {
                    let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 8 * j)), _mm256_loadu_ps(pb.add(i + 8 * j)));
                    *acc = _mm256_fmadd_ps(d, d, *acc);
                }
                i += 32;
            }
            while i + 8 <= n {
                let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
                acc[0] = _mm256_fmadd_ps(d, d, acc[0]);
                i += 8;
            }
            sum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])))
        };
        for j in i..n {
            let d = a[j] - b[j];
            s += d * d;
        }
        s
    }

    /// `dist2` against a row in half precision: eight halves to floats at a time (F16C).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dist2_f16(a: &[f32], b: &[u16]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut i = 0;
        // SAFETY (of the loads): each reads 8 floats and 8 halves ending at or before `n`.
        let mut s = unsafe {
            let mut acc = [_mm256_setzero_ps(); 4];
            while i + 32 <= n {
                for (j, acc) in acc.iter_mut().enumerate() {
                    let h = _mm256_cvtph_ps(_mm_loadu_si128(pb.add(i + 8 * j) as *const __m128i));
                    let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 8 * j)), h);
                    *acc = _mm256_fmadd_ps(d, d, *acc);
                }
                i += 32;
            }
            while i + 8 <= n {
                let h = _mm256_cvtph_ps(_mm_loadu_si128(pb.add(i) as *const __m128i));
                let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), h);
                acc[0] = _mm256_fmadd_ps(d, d, acc[0]);
                i += 8;
            }
            sum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])))
        };
        for j in i..n {
            let d = a[j] - f16_to_f32(b[j]);
            s += d * d;
        }
        s
    }

    /// ⟨a, b⟩, four registers of eight (as `dist2`).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut i = 0;
        // SAFETY (of the loads): each reads 8 floats ending at or before `n` in both slices.
        let mut s = unsafe {
            let mut acc = [_mm256_setzero_ps(); 4];
            while i + 32 <= n {
                for (j, acc) in acc.iter_mut().enumerate() {
                    *acc = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8 * j)), _mm256_loadu_ps(pb.add(i + 8 * j)), *acc);
                }
                i += 32;
            }
            while i + 8 <= n {
                acc[0] = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc[0]);
                i += 8;
            }
            sum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])))
        };
        for j in i..n {
            s += a[j] * b[j];
        }
        s
    }

    /// `dot` against a row in half precision: eight halves to floats at a time (F16C).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16(a: &[f32], b: &[u16]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut i = 0;
        // SAFETY (of the loads): each reads 8 floats and 8 halves ending at or before `n`.
        let mut s = unsafe {
            let mut acc = [_mm256_setzero_ps(); 4];
            while i + 32 <= n {
                for (j, acc) in acc.iter_mut().enumerate() {
                    let h = _mm256_cvtph_ps(_mm_loadu_si128(pb.add(i + 8 * j) as *const __m128i));
                    *acc = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8 * j)), h, *acc);
                }
                i += 32;
            }
            while i + 8 <= n {
                let h = _mm256_cvtph_ps(_mm_loadu_si128(pb.add(i) as *const __m128i));
                acc[0] = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), h, acc[0]);
                i += 8;
            }
            sum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])))
        };
        for j in i..n {
            s += a[j] * f16_to_f32(b[j]);
        }
        s
    }

    /// Halves of `v` into `out` (as long), eight at a time, rounded to nearest, ties to even;
    /// how many were done (a multiple of eight: the rest is the caller's).
    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn to_f16(v: &[f32], out: &mut [u16]) -> usize {
        let n = v.len().min(out.len());
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: 8 floats read and 8 halves written, both ending at or before `n`.
            unsafe {
                let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(v.as_ptr().add(i)));
                _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, h);
            }
            i += 8;
        }
        i
    }

    /// `QueryCode::build_lut`: `lut` holds 16 bytes for each 4 of `u`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn build_lut(u: &[u8], lut: &mut [u8]) {
        let groups = (u.len() / 4).min(lut.len() / 16);
        // SAFETY: the masks are 16 bytes each; group g reads u[4g..4g + 4] and writes
        // lut[16g..16g + 16], within `groups`.
        unsafe {
            let m = LUT_MASKS.map(|row| _mm_loadu_si128(row.as_ptr() as *const __m128i));
            for g in 0..groups {
                let v = u.as_ptr().add(4 * g);
                let a = _mm_add_epi8(_mm_and_si128(m[0], _mm_set1_epi8(*v as i8)), _mm_and_si128(m[1], _mm_set1_epi8(*v.add(1) as i8)));
                let b = _mm_add_epi8(_mm_and_si128(m[2], _mm_set1_epi8(*v.add(2) as i8)), _mm_and_si128(m[3], _mm_set1_epi8(*v.add(3) as i8)));
                _mm_storeu_si128(lut.as_mut_ptr().add(16 * g) as *mut __m128i, _mm_add_epi8(a, b));
            }
        }
    }

    /// `QueryCode::weighted_block`: one register holds a group's sums for all 32 codes, the low
    /// nibbles' (codes 0–15) in its low half and the high nibbles' (16–31) in its high half, so a
    /// group is one shuffle. `lut` and `block` hold 16 bytes a group; the groups are a multiple of 4.
    #[target_feature(enable = "avx2")]
    pub unsafe fn weighted_block(lut: &[u8], block: &[u8], out: &mut [u32; 32]) {
        let groups = lut.len() / 16;
        // SAFETY: loads are of 16 bytes at multiples of 16 below `groups` × 16 in `lut` and
        // `block`; the stores are the 32 sums.
        unsafe {
            let low4 = _mm256_set1_epi8(0x0f);
            let zero = _mm256_setzero_si256();
            let mut wide = [zero; 4];
            let mut chunk = 0;
            while chunk < groups {
                let end = groups.min(chunk + WIDEN);
                // Sums in 16 bits: codes 0–7 and 16–23, codes 8–15 and 24–31.
                let (mut lo, mut hi) = (zero, zero);
                let mut g4 = chunk;
                while g4 < end {
                    // Four groups add up in bytes (at most 4 · 60) before widening.
                    let mut bytes = zero;
                    for g in g4..g4 + 4 {
                        let b = _mm_loadu_si128(block.as_ptr().add(g * 16) as *const __m128i);
                        let t = _mm256_broadcastsi128_si256(_mm_loadu_si128(lut.as_ptr().add(g * 16) as *const __m128i));
                        let nibbles = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16::<4>(b), b), low4);
                        bytes = _mm256_add_epi8(bytes, _mm256_shuffle_epi8(t, nibbles));
                    }
                    lo = _mm256_add_epi16(lo, _mm256_unpacklo_epi8(bytes, zero));
                    hi = _mm256_add_epi16(hi, _mm256_unpackhi_epi8(bytes, zero));
                    g4 += 4;
                }
                // Into 32 bits: codes 0–3 and 16–19, 4–7 and 20–23, 8–11 and 24–27, 12–15 and 28–31.
                wide[0] = _mm256_add_epi32(wide[0], _mm256_unpacklo_epi16(lo, zero));
                wide[1] = _mm256_add_epi32(wide[1], _mm256_unpackhi_epi16(lo, zero));
                wide[2] = _mm256_add_epi32(wide[2], _mm256_unpacklo_epi16(hi, zero));
                wide[3] = _mm256_add_epi32(wide[3], _mm256_unpackhi_epi16(hi, zero));
                chunk = end;
            }
            for (i, w) in wide.into_iter().enumerate() {
                _mm_storeu_si128(out.as_mut_ptr().add(i * 4) as *mut __m128i, _mm256_castsi256_si128(w));
                _mm_storeu_si128(out.as_mut_ptr().add(16 + i * 4) as *mut __m128i, _mm256_extracti128_si256::<1>(w));
            }
        }
    }
}

// MARK: Clustering

/// Vectors k-means clusters (a random sample of them when there are more).
const KMEANS_SAMPLE: usize = 60_000;

/// k-means centroids for `data` (rows of `dims`), k-means++ seeding on a sample, then Lloyd.
pub fn kmeans(data: &[f32], dims: usize, k: usize, seed: u64) -> Vec<f32> {
    let n = data.len() / dims;
    let row = |i: usize| &data[i * dims..(i + 1) * dims];
    let mut s = seed | 1;
    let mut rand = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    // (The rows are the sample: a build passes as many as its centroids need.)
    let sample: Vec<usize> = (0..n).collect();
    let k = k.min(sample.len()).max(1);
    // k-means++ seeds from 32 candidates per centroid: its k passes are sequential, and over every
    // vector they were most of the build; Lloyd's iterations then see the whole sample.
    let seeds: Vec<usize> = if sample.len() <= 32 * k { sample.clone() } else { (0..32 * k).map(|_| sample[(rand() % sample.len() as u64) as usize]).collect() };
    let mut centroids: Vec<f32> = row(seeds[(rand() % seeds.len() as u64) as usize]).to_vec();
    let mut best: Vec<f32> = seeds.iter().map(|&i| dist2(row(i), &centroids[..dims])).collect();
    // Each step's distances to the new centroid as one matrix–vector product.
    let seed_rows: Vec<f32> = seeds.iter().flat_map(|&i| row(i).iter().copied()).collect();
    let seed_sq: Vec<f32> = seed_rows.chunks_exact(dims).map(|r| r.iter().map(|x| x * x).sum()).collect();
    let mut dots = vec![0f32; seeds.len()];
    while centroids.len() / dims < k {
        let total: f64 = best.iter().map(|&d| d as f64).sum();
        let mut pick = (rand() as f64 / u64::MAX as f64) * total;
        let mut chosen = seeds[seeds.len() - 1];
        for (j, &d) in best.iter().enumerate() {
            pick -= d as f64;
            if pick <= 0.0 {
                chosen = seeds[j];
                break;
            }
        }
        let c = row(chosen).to_vec();
        // The new centroid against every seed: one row times many, in pieces across threads.
        const PIECE: usize = 2048;
        dots.par_chunks_mut(PIECE).zip(seed_rows.par_chunks(PIECE * dims)).for_each(|(dots, rows)| gemm_nt(&c, rows, dims, dots));
        let c2: f32 = c.iter().map(|x| x * x).sum();
        for ((b, &d), &x2) in best.iter_mut().zip(&dots).zip(&seed_sq) {
            *b = b.min((x2 + c2 - 2.0 * d).max(0.0));
        }
        centroids.extend_from_slice(&c);
    }
    let rows: std::borrow::Cow<[f32]> = if sample.len() == n { data.into() } else { sample.iter().flat_map(|&i| row(i).iter().copied()).collect::<Vec<f32>>().into() };
    for _ in 0..8 {
        let assign = nearest_all(&rows, &centroids, dims);
        let mut sums = vec![0f64; k * dims];
        let mut counts = vec![0usize; k];
        for (&i, &a) in sample.iter().zip(&assign) {
            counts[a] += 1;
            for (d, v) in row(i).iter().enumerate() {
                sums[a * dims + d] += *v as f64;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dims {
                    centroids[c * dims + d] = (sums[c * dims + d] / counts[c] as f64) as f32;
                }
            }
        }
    }
    centroids
}

/// The nearest centroid of every row of `rows`. As a matrix product: the nearest c minimizes
/// ‖c‖² − 2⟨x, c⟩, and the products of a block of rows with every centroid are one GEMM, which
/// reads each centroid once per block instead of once per row (on macOS Accelerate's, which runs
/// on the matrix units; k-means' assignments were most of a segment's build time).
pub fn nearest_all(rows: &[f32], centroids: &[f32], dims: usize) -> Vec<usize> {
    let k = centroids.len() / dims;
    if k <= 1 {
        return vec![0; rows.len() / dims];
    }
    let norms: Vec<f32> = centroids.chunks_exact(dims).map(|c| c.iter().map(|x| x * x).sum()).collect();
    const BLOCK: usize = 256;
    // A thread's products go where its last block's went: a megabyte asked of the system and
    // given back for every block had the threads waiting on each other (six of them did the
    // work of three).
    rows.par_chunks(BLOCK * dims)
        .map_init(
            || vec![0f32; BLOCK * k],
            |dots, block| {
                let m = block.len() / dims;
                gemm_nt(block, centroids, dims, &mut dots[..m * k]);
                (0..m).map(|i| least(&dots[i * k..(i + 1) * k], &norms)).collect::<Vec<usize>>()
            },
        )
        .flatten_iter()
        .collect()
}

/// The first c for which ‖c‖² − 2⟨x, c⟩ is least, given the products and the norms.
fn least(dots: &[f32], norms: &[f32]) -> usize {
    let k = dots.len().min(norms.len());
    let mut best = (f32::MAX, 0usize);
    #[cfg(not(target_arch = "aarch64"))]
    let done = 0;
    #[cfg(target_arch = "aarch64")]
    let done = k / 4 * 4;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // SAFETY: NEON is part of aarch64; every load reads four floats under `done` in both.
        unsafe {
            // Four at a time, each lane keeping its least and where it was.
            let (mut low, mut at) = (vdupq_n_f32(f32::MAX), vdupq_n_u32(0));
            let mut places = vld1q_u32([0u32, 1, 2, 3].as_ptr());
            let mut c = 0;
            while c < done {
                let d = vld1q_f32(dots.as_ptr().add(c));
                let v = vsubq_f32(vld1q_f32(norms.as_ptr().add(c)), vaddq_f32(d, d));
                let less = vcltq_f32(v, low);
                low = vbslq_f32(less, v, low);
                at = vbslq_u32(less, places, at);
                places = vaddq_u32(places, vdupq_n_u32(4));
                c += 4;
            }
            let (mut lows, mut ats) = ([0f32; 4], [0u32; 4]);
            vst1q_f32(lows.as_mut_ptr(), low);
            vst1q_u32(ats.as_mut_ptr(), at);
            for (&v, &c) in lows.iter().zip(&ats) {
                if v < best.0 || v == best.0 && (c as usize) < best.1 {
                    best = (v, c as usize);
                }
            }
        }
    }
    for c in done..k {
        let v = norms[c] - 2.0 * dots[c];
        if v < best.0 {
            best = (v, c);
        }
    }
    best.1
}

/// out (m × k) = a (m × dims) · bᵀ (b: k × dims), row-major.
fn gemm_nt(a: &[f32], b: &[f32], dims: usize, out: &mut [f32]) {
    #[cfg(target_os = "macos")]
    {
        let (m, k) = (a.len() / dims, b.len() / dims);
        #[link(name = "Accelerate", kind = "framework")]
        unsafe extern "C" {
            fn cblas_sgemm(order: i32, ta: i32, tb: i32, m: i32, n: i32, k: i32, alpha: f32, a: *const f32, lda: i32, b: *const f32, ldb: i32, beta: f32, c: *mut f32, ldc: i32);
        }
        const ROW_MAJOR: i32 = 101;
        const NO_TRANS: i32 = 111;
        const TRANS: i32 = 112;
        // SAFETY: the slices hold m × dims, k × dims and m × k floats, as the call is told.
        unsafe { cblas_sgemm(ROW_MAJOR, NO_TRANS, TRANS, m as i32, k as i32, dims as i32, 1.0, a.as_ptr(), dims as i32, b.as_ptr(), dims as i32, 0.0, out.as_mut_ptr(), k as i32) };
    }
    #[cfg(not(target_os = "macos"))]
    {
        #[cfg(target_arch = "x86_64")]
        if x86::avx2() {
            // SAFETY: the processor has AVX2 and FMA.
            return unsafe { x86::gemm_nt(a, b, dims, out) };
        }
        #[cfg(target_arch = "aarch64")]
        return gemm_nt_neon(a, b, dims, out);
        #[cfg(not(target_arch = "aarch64"))]
        gemm_nt_portable(a, b, dims, out)
    }
}

/// `gemm_nt` with NEON, for ARM where there's no BLAS to call (Linux): a row against four of
/// `b`'s at a time, each product in four registers of four, so sixteen multiply-adds are under
/// way at once: about two a cycle, which is as fast as their operands load. The loop the
/// compiler makes of `gemm_nt_portable` took three times as long there (assigning 60,000
/// vectors of 256 dimensions to 632 centroids on one core: 1.17 s, and 0.39 s with this, its
/// tiles and `least`), and that was most of a build: a million such vectors were indexed in
/// 39 s, 19 of them in one k-means, and are in 11.
#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
fn gemm_nt_neon(a: &[f32], b: &[f32], dims: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;
    let (m, k) = (a.len() / dims, b.len() / dims);
    assert!(a.len() >= m * dims && b.len() >= k * dims && out.len() >= m * k);
    let whole = dims / 16 * 16;
    let tile = tile(dims);
    // A tile of `b`'s rows against a tile of `a`'s (see `tile`).
    for c0 in (0..k).step_by(tile) {
        let c1 = (c0 + tile).min(k);
        for i0 in (0..m).step_by(tile) {
            for i in i0..(i0 + tile).min(m) {
                let x = &a[i * dims..(i + 1) * dims];
                let mut c = c0;
                while c + 4 <= c1 {
                    // SAFETY: every load reads four floats within row `i` of `a` or one of rows
                    // `c..c + 4` of `b`, each `dims` long, at an offset under `whole`.
                    unsafe {
                        let ys: [*const f32; 4] = std::array::from_fn(|j| b.as_ptr().add((c + j) * dims));
                        let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
                        let mut t = 0;
                        while t < whole {
                            let xs = [vld1q_f32(x.as_ptr().add(t)), vld1q_f32(x.as_ptr().add(t + 4)), vld1q_f32(x.as_ptr().add(t + 8)), vld1q_f32(x.as_ptr().add(t + 12))];
                            for j in 0..4 {
                                for l in 0..4 {
                                    acc[j][l] = vfmaq_f32(acc[j][l], xs[l], vld1q_f32(ys[j].add(t + 4 * l)));
                                }
                            }
                            t += 16;
                        }
                        for j in 0..4 {
                            let sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc[j][0], acc[j][1]), vaddq_f32(acc[j][2], acc[j][3])));
                            let y = std::slice::from_raw_parts(ys[j], dims);
                            out[i * k + c + j] = sum + (whole..dims).map(|d| x[d] * y[d]).sum::<f32>();
                        }
                    }
                    c += 4;
                }
                for c in c..c1 {
                    out[i * k + c] = dot(x, &b[c * dims..(c + 1) * dims]);
                }
            }
        }
    }
}

/// How many rows of each matrix `gemm_nt` takes at a time: 16 KB of each, so both stay in a
/// first-level cache while every row of one meets every row of the other. A row at a time
/// against all of the other matrix read all of it from the second-level cache for every row,
/// and that traffic was a third of the time (the pass above: 0.67 → 0.43 s). A multiple of
/// four: the other matrix's rows are taken four at a time.
#[cfg_attr(target_os = "macos", allow(dead_code))]
fn tile(dims: usize) -> usize {
    ((16 << 10) / (dims.max(1) * 4)).clamp(4, 64) / 4 * 4
}

/// ⟨a, b⟩, sixteen multiply-adds under way at once.
#[cfg(target_arch = "aarch64")]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len().min(b.len());
    let whole = n / 16 * 16;
    // SAFETY: every load reads four floats ending at or before `whole` in both slices.
    let sum = unsafe {
        let mut acc = [vdupq_n_f32(0.0); 4];
        let mut t = 0;
        while t < whole {
            for (l, acc) in acc.iter_mut().enumerate() {
                *acc = vfmaq_f32(*acc, vld1q_f32(a.as_ptr().add(t + 4 * l)), vld1q_f32(b.as_ptr().add(t + 4 * l)));
            }
            t += 16;
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])))
    };
    sum + (whole..n).map(|d| a[d] * b[d]).sum::<f32>()
}

/// ⟨a, b⟩, in sixteen sums the compiler keeps in registers (as `dist2` is).
#[cfg(not(target_arch = "aarch64"))]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if x86::avx2() {
        // SAFETY: the processor has AVX2 and FMA.
        return unsafe { x86::dot(a, b) };
    }
    let mut acc = [0f32; 16];
    let (ac, bc) = (a.chunks_exact(16), b.chunks_exact(16));
    let (ar, br) = (ac.remainder(), bc.remainder());
    for (x, y) in ac.zip(bc) {
        for l in 0..16 {
            acc[l] += x[l] * y[l];
        }
    }
    acc.iter().sum::<f32>() + ar.iter().zip(br).map(|(x, y)| x * y).sum::<f32>()
}

/// `gemm_nt` as loops that vectorize, where there's no BLAS to call.
#[cfg_attr(any(target_os = "macos", target_arch = "aarch64"), allow(dead_code))]
#[inline(always)]
fn gemm_nt_portable(a: &[f32], b: &[f32], dims: usize, out: &mut [f32]) {
    let (m, k) = (a.len() / dims, b.len() / dims);
    let tile = tile(dims);
    // A tile of `b`'s rows against a tile of `a`'s (see `tile`).
    for c0 in (0..k).step_by(tile) {
        let c1 = (c0 + tile).min(k);
        for i0 in (0..m).step_by(tile) {
            // Four of `b`'s rows against one of `a`'s at a time: each load of the row serves
            // four products.
            for i in i0..(i0 + tile).min(m) {
                let x = &a[i * dims..(i + 1) * dims];
                let mut c = c0;
                while c + 4 <= c1 {
                    let ys: [&[f32]; 4] = std::array::from_fn(|j| &b[(c + j) * dims..(c + j + 1) * dims]);
                    let mut acc = [[0f32; 8]; 4];
                    let chunks = dims / 8;
                    for t in 0..chunks {
                        let xs = &x[t * 8..t * 8 + 8];
                        for j in 0..4 {
                            let y = &ys[j][t * 8..t * 8 + 8];
                            for l in 0..8 {
                                acc[j][l] += xs[l] * y[l];
                            }
                        }
                    }
                    for j in 0..4 {
                        let tail: f32 = (chunks * 8..dims).map(|d| x[d] * ys[j][d]).sum();
                        out[i * k + c + j] = acc[j].iter().sum::<f32>() + tail;
                    }
                    c += 4;
                }
                for c in c..c1 {
                    out[i * k + c] = x.iter().zip(&b[c * dims..(c + 1) * dims]).map(|(p, q)| p * q).sum();
                }
            }
        }
    }
}

pub fn nearest(v: &[f32], centroids: &[f32], dims: usize) -> usize {
    let mut best = (f32::MAX, 0);
    for (c, cent) in centroids.chunks_exact(dims).enumerate() {
        let d = dist2(v, cent);
        if d < best.0 {
            best = (d, c);
        }
    }
    best.1
}

/// How many clusters for `n` vectors: about √n, between 16 and 4096 (one below `MIN_CLUSTERED`).
pub fn cluster_count(n: usize) -> usize {
    if n < MIN_CLUSTERED { 1 } else { ((n as f64).sqrt() as usize).clamp(16, 4096) }
}

// MARK: The index of one segment

/// Centroids that segments share: a generation. A segment's vectors are in lists by nearest
/// centroid, and segments of one generation have the same centroids, so a search ranks them once
/// and scans each one's lists as if they were one segment's, a flush assigns its vectors to them
/// instead of clustering, and a merge copies codes instead of encoding. (With clusters of its own
/// in every segment, a search of S segments scanned about √S times the codes of one merged
/// segment: 5.7% against 2.2% of a million in 8, and every merge clustered, encoded and calibrated
/// everything again. Lance and Elasticsearch's DiskBBQ share centroids across segments the same
/// way; SPFresh keeps one set for the whole index.)
#[derive(Clone, Copy)]
pub struct Generation<'a> {
    /// The id of the segment that trained it (never 0: that's a segment without clusters).
    pub id: u64,
    /// How many vectors the namespace held when it was trained: what its clusters are sized
    /// for. Once the namespace holds four times as many, a build with enough vectors of its own
    /// trains the next one.
    pub trained_on: u32,
    /// The probe budget calibrated when it was trained.
    pub nprobe: u32,
    /// And the budgets for deeper searches, measured then.
    pub budgets: Budgets,
    /// Its centroids, rotated: `padded` floats each.
    pub centroids: &'a [f32],
}

/// How many depths a segment has room to keep a probe budget for, and the deepest measured:
/// the 40,960 nearest (each of 64 samples keeps that many neighbours while they are measured).
pub const DEPTHS: usize = 16;
const DEEPEST: usize = 12;

/// The depths: the 10 nearest, the 20 nearest, and so on, doubling. (In steps of four, a budget
/// read between two depths was far off where the data has an edge: topics of a thousand
/// vectors need a few clusters for their own and hundreds for the next.)
pub const fn depth_of(i: usize) -> usize {
    10 << i
}

/// How many clusters to probe to find the true `depth_of(i)` nearest vectors `TARGET_RECALL` of
/// the time, as measured when the clusters were trained (`calibrate`).
///
/// A search for k results among a share s of the vectors (a filter's, less what's deleted) has
/// to look about as deep as a search for the k / s nearest of all, and how many more clusters
/// that takes depends on the data: where neighbours beyond the first ten are spread over many
/// clusters, a formula fitted on other data stops far too early (a million synthetic vectors
/// in a thousand topics, 100 results under a filter of a tenth: recall 0.976 with the fitted
/// exponent). So it is measured, like the budget for the first ten.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Budgets {
    /// By depth; 0 where it wasn't measured (a depth past an eighth of the vectors).
    pub by_depth: [u32; DEPTHS],
    /// How many vectors the lists held when it was measured.
    pub measured_on: u32,
}

impl Budgets {
    /// The clusters to probe, of `clusters` holding `vectors`, for the `depth` nearest; none if
    /// nothing was measured (a segment from before there were budgets).
    ///
    /// Between two measured depths, a line through them on logarithmic scales; past the last,
    /// a line to "every vector takes every cluster". The same lists holding more vectors than
    /// when they were measured hold the same neighbourhood at a proportionally greater depth.
    pub fn probes(&self, clusters: usize, vectors: usize, depth: f64) -> Option<usize> {
        if self.by_depth[0] == 0 || self.measured_on == 0 {
            return None;
        }
        let depth = depth * self.measured_on as f64 / vectors.max(1) as f64;
        let mut points: Vec<(f64, f64)> = (0..DEPTHS).filter(|&i| self.by_depth[i] > 0).map(|i| (depth_of(i) as f64, self.by_depth[i] as f64)).collect();
        let last = *points.last().unwrap();
        if self.measured_on as f64 > last.0 {
            points.push((self.measured_on as f64, (clusters as f64).max(last.1)));
        }
        if depth <= points[0].0 {
            return Some(points[0].1 as usize);
        }
        for w in points.windows(2) {
            if depth <= w[1].0 {
                let t = (depth.ln() - w[0].0.ln()) / (w[1].0.ln() - w[0].0.ln());
                // (Rounded up, but not for the rounding of the logarithms at a measured depth.)
                let probes = (w[0].1.ln() + t * (w[1].1.ln() - w[0].1.ln())).exp();
                return Some(((probes - 1e-6).ceil() as usize).min(clusters.max(1)));
            }
        }
        Some(clusters.max(1))
    }
}

/// How many times the vectors a generation was trained for a namespace must hold before the next
/// is trained. Its clusters are sized for twice what it was trained for, so at four times as many
/// a search does about 6% more work than with clusters sized for that, and at once as many as
/// much (the cost of being off by a factor x in the number of clusters is (x + 1/x) / 2).
pub const OUTGROWN: usize = 4;
/// The fewest vectors per centroid a build trains with (the RaBitQ library warns under 39).
const PER_CENTROID: usize = 40;

/// Whether a build of `n` vectors, in a namespace of `total`, goes into `generation` (assigning,
/// not clustering): while the namespace hasn't outgrown it, and after that for as long as a build
/// is too small to train the centroids the namespace now needs (a larger merge will).
pub fn builds_into(generation: Option<&Generation<'_>>, padded: usize, n: usize, total: usize) -> bool {
    generation.is_some_and(|g| {
        let whole = g.id != 0 && g.centroids.len() >= 2 * padded && g.centroids.len() % padded == 0;
        whole && (total.max(n) < OUTGROWN * g.trained_on as usize || n < PER_CENTROID * cluster_count(2 * total.max(n)))
    })
}

/// What a merge knows of its vectors from segments of the generation it builds into, by document:
/// each one's list (u32::MAX: not known, to be assigned), code and factors, copied as they are
/// (and for dot product its length along its centroid, `Built::along`).
#[derive(Clone, Copy)]
pub struct Known<'a> {
    pub lists: &'a [u32],
    pub codes: &'a [u64],
    pub pre: &'a [Pre],
    pub along: &'a [f32],
}

/// Everything a segment stores about its vectors, built from (document, vector) pairs: the
/// vectors as the metric stores them (`Metric::prepare`).
pub struct Built {
    pub dims: usize,
    pub seed: u64,
    pub metric: Metric,
    /// Clusters to probe for recall@10 of `TARGET_RECALL` (see `calibrate`; 1 with one cluster).
    pub nprobe: u32,
    /// And for deeper searches (nothing measured with one cluster).
    pub budgets: Budgets,
    /// The generation these lists are of (0: one cluster, compared exactly), and how many vectors
    /// it was trained on.
    pub generation: u64,
    pub trained_on: u32,
    pub centroids: Vec<f32>,
    pub radius: Vec<f32>,
    pub offsets: Vec<u32>,
    pub docs: Vec<u32>,
    pub codes: Vec<u64>,
    pub pre: Vec<Pre>,
    /// For dot product (empty for the other metrics): how many times its list's centroid each
    /// vector holds, in the codes' order (its code is of the rest, across the centroid:
    /// `search_lists`), and the length of the longest vector in each list.
    pub along: Vec<f32>,
    pub longest: Vec<f32>,
}

/// Builds a segment's vector index: into `generation` if there is one to build into
/// (`builds_into`: each vector goes to its nearest centroid; nothing is clustered or calibrated,
/// and what `known` holds is copied), else by training one for a namespace of `total` vectors,
/// which `id` (the segment's) then names.
///
/// By distance (cosine and Euclidean), lists are k-means' by Euclidean distance, each vector
/// with its nearest centroid, and a code is of the vector's residual from it. By dot product
/// they are of directions: k-means over the vectors at length 1, each vector with the centroid
/// nearest its direction, and a code is of what is left of the vector across that centroid
/// (`search_lists` says why). Either way the probe budgets are calibrated against the
/// metric's own nearest neighbours, in the order it probes centroids in.
#[allow(clippy::too_many_arguments)]
pub fn build(vectors: &[(u32, VectorRef<'_>)], dims: usize, seed: u64, id: u64, generation: Option<Generation<'_>>, known: Option<Known<'_>>, total: usize, metric: Metric) -> Built {
    let rot = Rotation::new(dims, seed);
    let p = rot.padded;
    let n = vectors.len();
    let total = total.max(n);
    let words = p / 64;
    let into = generation.filter(|g| builds_into(Some(g), p, n, total));
    // Clusters for twice the namespace's vectors (the generation serves until there are four
    // times as many), as far as these vectors can train: a first segment trains what it can.
    let wanted = cluster_count(2 * total);
    let clusters = wanted.min((n / PER_CENTROID).max(16));
    // Codes are relative to their centroids: another generation's don't carry over.
    let known = known.filter(|_| into.is_some());
    // Vectors are rotated where they're needed: a sample for k-means, then block by block to be
    // assigned and encoded. All of them rotated at once had been a gigabyte at a million.
    let centroids = match into {
        Some(g) => g.centroids.to_vec(),
        None if n < MIN_CLUSTERED => {
            let sums: Vec<f64> = vectors
                .par_chunks(4096)
                .map(|block| {
                    let mut sum = vec![0f64; p];
                    for (_, v) in block {
                        for (s, x) in sum.iter_mut().zip(rot.apply_ref(*v)) {
                            *s += x as f64;
                        }
                    }
                    sum
                })
                .reduce(|| vec![0f64; p], |a, b| a.iter().zip(&b).map(|(x, y)| x + y).collect());
            sums.iter().map(|s| (s / n.max(1) as f64) as f32).collect()
        }
        None => {
            let mut r = (seed ^ 0x51_7C_C1_B7) | 1;
            let mut rand = move || {
                r ^= r << 13;
                r ^= r >> 7;
                r ^= r << 17;
                r
            };
            let most = KMEANS_SAMPLE.max(PER_CENTROID * clusters);
            let sample: Vec<usize> = if n <= most { (0..n).collect() } else { (0..most).map(|_| (rand() % n as u64) as usize).collect() };
            let mut rotated: Vec<f32> = sample.par_iter().flat_map_iter(|&i| rot.apply_ref(vectors[i].1)).collect();
            if metric == Metric::Dot {
                rotated.par_chunks_mut(p).for_each(to_unit);
            }
            kmeans(&rotated, p, clusters, seed ^ 0x9E37_79B9)
        }
    };
    let k = centroids.len() / p;
    let dotted = metric == Metric::Dot;
    // (For dot product: each centroid's squared length, for a vector's length along it.)
    let centroid_sq: Vec<f32> = if dotted { centroids.chunks_exact(p).map(|c| dot(c, c)).collect() } else { Vec::new() };
    let blocks: Vec<(Vec<u32>, Vec<u64>, Vec<Pre>, Vec<f32>, Vec<f32>)> = vectors
        .par_chunks(4096)
        .map(|block| {
            let mut assign = vec![0u32; block.len()];
            let mut codes = vec![0u64; block.len() * words];
            let mut pre = vec![Pre::default(); block.len()];
            // (By dot product: each vector's share of its centroid, and its own length.)
            let mut along = vec![0f32; if dotted { block.len() } else { 0 }];
            let mut length = vec![0f32; if dotted { block.len() } else { 0 }];
            // Those to work out, rotated, assigned and encoded together; the rest copied.
            let mut fresh: Vec<usize> = Vec::new();
            for (i, (doc, _)) in block.iter().enumerate() {
                match known.filter(|kn| kn.lists[*doc as usize] != u32::MAX) {
                    Some(kn) => {
                        let d = *doc as usize;
                        assign[i] = kn.lists[d];
                        codes[i * words..(i + 1) * words].copy_from_slice(&kn.codes[d * words..(d + 1) * words]);
                        pre[i] = kn.pre[d];
                        if dotted {
                            // (Its length from its two parts, which are at right angles.)
                            along[i] = kn.along[d];
                            length[i] = (along[i] * along[i] * centroid_sq[assign[i] as usize] + pre[i].norm * pre[i].norm).sqrt();
                        }
                    }
                    None => fresh.push(i),
                }
            }
            if !fresh.is_empty() && dotted {
                let rotated: Vec<f32> = fresh.iter().flat_map(|&i| rot.apply_ref(block[i].1)).collect();
                // The centroid nearest each vector's direction.
                let mut unit = rotated.clone();
                unit.chunks_exact_mut(p).for_each(to_unit);
                let nearest = nearest_all(&unit, &centroids, p);
                let mut part = vec![0f32; p];
                for ((r, &a), &i) in rotated.chunks_exact(p).zip(&nearest).zip(&fresh) {
                    // The vector is so many times its centroid, and a rest across it: the
                    // code is of the rest.
                    let centroid = &centroids[a * p..(a + 1) * p];
                    let times = if centroid_sq[a] > 0.0 { dot(r, centroid) / centroid_sq[a] } else { 0.0 };
                    part.iter_mut().zip(centroid).for_each(|(x, &c)| *x = times * c);
                    let code = &mut codes[i * words..(i + 1) * words];
                    let f = encode(r, &part, code);
                    assign[i] = a as u32;
                    pre[i] = Pre::of(f, code);
                    along[i] = times;
                    length[i] = dot(r, r).sqrt();
                }
            } else if !fresh.is_empty() {
                let rotated: Vec<f32> = fresh.iter().flat_map(|&i| rot.apply_ref(block[i].1)).collect();
                let nearest = nearest_all(&rotated, &centroids, p);
                for ((r, &a), &i) in rotated.chunks_exact(p).zip(&nearest).zip(&fresh) {
                    let code = &mut codes[i * words..(i + 1) * words];
                    let f = encode(r, &centroids[a * p..(a + 1) * p], code);
                    assign[i] = a as u32;
                    pre[i] = Pre::of(f, code);
                }
            }
            (assign, codes, pre, along, length)
        })
        .collect();
    let assign: Vec<u32> = blocks.iter().flat_map(|b| b.0.iter().copied()).collect();
    let codes: Vec<u64> = blocks.iter().flat_map(|b| b.1.iter().copied()).collect();
    let along: Vec<f32> = blocks.iter().flat_map(|b| b.3.iter().copied()).collect();
    let length: Vec<f32> = blocks.iter().flat_map(|b| b.4.iter().copied()).collect();
    let pre: Vec<Pre> = blocks.into_iter().flat_map(|b| b.2).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (assign[i], vectors[i].0));
    let mut offsets = vec![0u32; k + 1];
    for &a in &assign {
        offsets[a as usize + 1] += 1;
    }
    for c in 0..k {
        offsets[c + 1] += offsets[c];
    }
    let mut radius = vec![0f32; k];
    for (&a, f) in assign.iter().zip(&pre) {
        radius[a as usize] = radius[a as usize].max(f.norm);
    }
    let mut longest = vec![0f32; if dotted { k } else { 0 }];
    for (&a, &length) in assign.iter().zip(&length) {
        longest[a as usize] = longest[a as usize].max(length);
    }
    let mut built = Built {
        dims,
        seed,
        metric,
        nprobe: into.map_or(1, |g| g.nprobe),
        budgets: into.map_or(Budgets::default(), |g| g.budgets),
        generation: into.map_or(if k > 1 { id } else { 0 }, |g| g.id),
        // What its clusters are sized for: the namespace, or these vectors if they couldn't
        // train that many.
        trained_on: into.map_or(if clusters == wanted { total } else { n } as u32, |g| g.trained_on),
        docs: order.iter().map(|&i| vectors[i].0).collect(),
        codes: order.iter().flat_map(|&i| codes[i * words..(i + 1) * words].iter().copied()).collect(),
        pre: order.iter().map(|&i| pre[i]).collect(),
        along: if dotted { order.iter().map(|&i| along[i]).collect() } else { Vec::new() },
        longest,
        centroids,
        radius,
        offsets,
    };
    if into.is_none() && k > 1 {
        (built.nprobe, built.budgets) = calibrate(&built, vectors);
    }
    built
}

/// The recall@10 a segment's default probe budget aims for.
pub const TARGET_RECALL: f64 = 0.997;
/// For segments without measured budgets (`Budgets`: written before version 8): how a probe
/// budget calibrated for the 10 nearest of all vectors grows with how deep a search must look.
/// For k results among a share s of the vectors (a filter's, less what's deleted), about k / s
/// neighbours of the unfiltered order, the budget times (k / (10 s))^this. Chosen on real
/// queries held apart from the test queries (`examples/ann_calibration.rs`, plan 011); on data
/// whose neighbours beyond the first ten are spread wider, it stops too early.
pub const DEPTH_EXPONENT: f64 = 0.1;

/// For calibrating `DEPTH_EXPONENT` (bench builds): when set, used instead (f64 bits).
#[cfg(feature = "bench")]
pub static DEPTH_EXPONENT_OVERRIDE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The probe budget for `k` results among a share `passing` of a segment's vectors, from `base`,
/// calibrated for 10 among all.
pub fn scaled_budget(base: usize, k: usize, passing: f64) -> usize {
    #[allow(unused_mut)]
    let mut exponent = DEPTH_EXPONENT;
    #[cfg(feature = "bench")]
    {
        let bits = DEPTH_EXPONENT_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
        if bits != 0 {
            exponent = f64::from_bits(bits);
        }
    }
    let depth = (k as f64 / passing.clamp(1e-4, 1.0) / 10.0).max(1.0);
    (base as f64 * depth.powf(exponent)).ceil() as usize
}

/// Vectors that stand in for queries when calibrating.
const SAMPLES: usize = 256;
/// Probe budgets tried, smallest first (then every cluster). From four, where it had been
/// eight: where vectors gather as tightly as their clusters, a few lists hold the ten nearest
/// as often as eight do (the million in a thousand topics: 0.998 at two, 0.999 at four), and
/// each list more is tables made and codes scanned for nothing. Not from two: the samples are
/// the segment's own vectors, and queries from elsewhere find a little less than they predict
/// (FiQA: half a point), which four lists cover for 3% of a search's time.
const BUDGETS: [usize; 14] = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384];

/// How many clusters this segment must probe to find the true 10 nearest neighbours of its own
/// vectors (each left out of its own results) `TARGET_RECALL` of the time: the smallest budget
/// that does, over up to 256 sample vectors (with 100, sampling noise of about half a point made
/// it stop a step early on one corpus). How clustered embeddings are varies a lot by data
/// (on one corpus 16 probes find 99% of neighbours, on another 91%), so a fixed default is either
/// slow on one or inexact on the other; and held-out vectors predict real queries' recall well
/// (BEIR FiQA: 0.919 against 0.913 at 16 probes, 0.991 against 0.986 at 64).
///
/// The same pass over every vector also finds, for 64 of the samples, their nearest 20, 40 and
/// so on (up to an eighth of the vectors, 40,960 at most), and with them the budgets for those
/// depths: a neighbour is found if its cluster is among those probed, so the budget for a depth
/// is the place, in a sample's order of centroids, that `TARGET_RECALL` of its neighbours'
/// clusters are within. No searches: a search for 40,960 results reads that many vectors.
fn calibrate(built: &Built, vectors: &[(u32, VectorRef<'_>)]) -> (u32, Budgets) {
    let n = vectors.len();
    let clusters = built.radius.len();
    let samples: Vec<usize> = (0..SAMPLES.min(n)).map(|i| i * n / SAMPLES.min(n)).collect();
    let max_doc = vectors.iter().map(|(d, _)| *d).max().unwrap_or(0) as usize;
    let mut rows = vec![u32::MAX; max_doc + 1];
    for (row, (d, _)) in vectors.iter().enumerate() {
        rows[*d as usize] = row as u32;
    }
    let words = built.codes.len() / built.pre.len().max(1);
    let pre = &built.pre;
    let blocked = Blocked::new(&built.codes, &built.offsets, words);
    let rotation = Rotation::new(built.dims, built.seed);
    let view = View {
        rotation: &rotation,
        centroids: &built.centroids,
        radius: &built.radius,
        offsets: &built.offsets,
        docs: &built.docs,
        pre,
        blocks: &blocked.bytes,
        starts: &blocked.starts,
        block: blocked.block,
        rows: &rows,
        by_code: false,
        full: Rows::Refs(vectors),
        nprobe: 0,
        budgets: Budgets::default(),
        metric: built.metric,
        along: &built.along,
        longest: &built.longest,
    };
    // The true neighbours, by scoring every vector: ‖q − v‖² = ‖q‖² + ‖v‖² − 2⟨q, v⟩, the
    // products one matrix product per block of 4,096 vectors, gathered as they're needed: copying
    // every vector, and a distance per sample and vector at once, had been gigabytes at a million.
    // For dot product the neighbours are those of the greatest product, which is the product
    // alone: what a search by that metric has to find is what its budgets are measured against.
    let dotted = built.metric == Metric::Dot;
    let dims = built.dims;
    let m = samples.len();
    let queries: Vec<f32> = samples.iter().flat_map(|&s| vectors[s].1.to_f32().into_owned()).collect();
    let qsq: Vec<f32> = queries.chunks_exact(dims).map(|v| v.iter().map(|x| x * x).sum()).collect();
    const BLOCK: usize = 4096;
    // The samples whose deeper neighbours are kept, and how deep: each one's nearest, behind a
    // lock (a block offers only what is nearer than the farthest kept so far: soon, little).
    const DEEP: usize = 64;
    let deepest = (1..=DEEPEST).rev().find(|&i| depth_of(i) * 8 <= n);
    let deep: Vec<usize> = match deepest {
        Some(_) => (0..DEEP.min(m)).map(|i| i * m / DEEP.min(m)).collect(),
        None => Vec::new(),
    };
    let deep_tops: Vec<std::sync::Mutex<Nearest>> = deep.iter().map(|_| std::sync::Mutex::new(Nearest::new(deepest.map_or(0, depth_of)))).collect();
    let partial: Vec<Vec<Vec<(f32, u64)>>> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| {
            let first = b * BLOCK;
            let block: Vec<f32> = vectors[first..(first + BLOCK).min(n)].iter().flat_map(|(_, v)| v.to_f32().into_owned()).collect();
            let k = block.len() / dims;
            let mut dots = vec![0f32; m * k];
            gemm_nt(&queries, &block, dims, &mut dots);
            let sq: Vec<f32> = block.chunks_exact(dims).map(|v| v.iter().map(|x| x * x).sum()).collect();
            for (top, &i) in deep_tops.iter().zip(&deep) {
                let me = vectors[samples[i]].0;
                let limit = top.lock().unwrap().threshold();
                let near: Vec<(f32, u64)> = dots[i * k..(i + 1) * k]
                    .iter()
                    .zip(&sq)
                    .enumerate()
                    .filter_map(|(j, (&dot, &v2))| {
                        let d2 = if dotted { -dot } else { qsq[i] + v2 - 2.0 * dot };
                        (d2 < limit && vectors[first + j].0 != me).then_some((d2, (first + j) as u64))
                    })
                    .collect();
                if !near.is_empty() {
                    let mut top = top.lock().unwrap();
                    for (d2, row) in near {
                        top.offer(d2, row);
                    }
                }
            }
            (0..m)
                .map(|i| {
                    let me = vectors[samples[i]].0;
                    let mut top = Nearest::new(10);
                    for (j, (&dot, &v2)) in dots[i * k..(i + 1) * k].iter().zip(&sq).enumerate() {
                        if vectors[first + j].0 != me {
                            top.offer(if dotted { -dot } else { qsq[i] + v2 - 2.0 * dot }, (first + j) as u64);
                        }
                    }
                    top.into_sorted()
                })
                .collect()
        })
        .collect();
    let truth: Vec<Vec<u32>> = (0..m)
        .map(|i| {
            let mut top = Nearest::new(10);
            for p in &partial {
                for &(d2, row) in &p[i] {
                    top.offer(d2, row);
                }
            }
            top.into_sorted().into_iter().map(|(_, row)| vectors[row as usize].0).collect()
        })
        .collect();
    let recall_at = |budget: usize| -> f64 {
        let found: usize = samples
            .par_iter()
            .zip(&truth)
            .map(|(&s, want)| {
                let (me, q) = (vectors[s].0, vectors[s].1.to_f32());
                let mut top = Nearest::new(10);
                view.search(&q, 0, &|d| d != me, &mut top, budget);
                let got: Vec<u32> = top.into_sorted().into_iter().map(|(_, id)| id as u32).collect();
                want.iter().filter(|d| got.contains(d)).count()
            })
            .sum();
        found as f64 / truth.iter().map(Vec::len).sum::<usize>().max(1) as f64
    };
    // Below the target, recall can stop growing with the budget: the codes' estimates, not the
    // probes, are what misses then (a million synthetic vectors: 0.996 from 8 probes on). Two
    // budgets in a row that add under a twentieth of a point keep the smaller one, instead of
    // falling back to every cluster (20× slower there for nothing).
    let base = 'budget: {
        let mut plateau: Option<(usize, f64)> = None;
        let mut flat = 0;
        for &b in BUDGETS.iter().filter(|&&b| b < clusters) {
            let r = recall_at(b);
            if r >= TARGET_RECALL {
                break 'budget b as u32;
            }
            match plateau {
                Some((_, best)) if r < best + 0.0005 => {
                    flat += 1;
                    if flat == 2 {
                        break 'budget plateau.unwrap().0 as u32;
                    }
                }
                _ => {
                    plateau = Some((b, r));
                    flat = 0;
                }
            }
        }
        clusters as u32
    };

    // Deeper: where each neighbour's cluster is in its sample's order of centroids.
    let mut budgets = Budgets { by_depth: [0; DEPTHS], measured_on: n as u32 };
    budgets.by_depth[0] = base;
    if let Some(deepest) = deepest {
        let mut cluster_of = vec![0u32; n];
        for c in 0..clusters {
            for &doc in &built.docs[built.offsets[c] as usize..built.offsets[c + 1] as usize] {
                cluster_of[rows[doc as usize] as usize] = c as u32;
            }
        }
        let p = rotation.padded;
        // Per depth, every (sample, neighbour)'s place.
        let mut places: Vec<Vec<u32>> = vec![Vec::new(); deepest + 1];
        for (top, &i) in deep_tops.into_iter().zip(&deep) {
            let qr = rotation.apply(&queries[i * dims..(i + 1) * dims]);
            // (In the order a search probes them: `search_lists`.)
            let mut order: Vec<(f32, u32)> = if dotted {
                let mut unit = qr.clone();
                to_unit(&mut unit);
                (0..clusters).map(|c| (promise(&unit, &built.centroids[c * p..(c + 1) * p], built.longest[c]), c as u32)).collect()
            } else {
                (0..clusters).map(|c| (dist2(&qr, &built.centroids[c * p..(c + 1) * p]), c as u32)).collect()
            };
            order.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut place = vec![0u32; clusters];
            for (at, &(_, c)) in order.iter().enumerate() {
                place[c as usize] = at as u32 + 1;
            }
            let nearest = top.into_inner().unwrap().into_sorted();
            for (d, places) in places.iter_mut().enumerate().skip(1) {
                places.extend(nearest.iter().take(depth_of(d)).map(|&(_, row)| place[cluster_of[row as usize] as usize]));
            }
        }
        for d in 1..=deepest {
            let places = &mut places[d];
            places.sort_unstable();
            let within = places.get(((places.len() as f64 * TARGET_RECALL).ceil() as usize).saturating_sub(1)).copied().unwrap_or(0);
            // Never fewer than for a shallower search.
            budgets.by_depth[d] = within.max(budgets.by_depth[d - 1]);
        }
    }
    (base, budgets)
}

/// A read-only view of one segment's vectors (over the mapped file).
#[derive(Clone, Copy)]
pub struct View<'a> {
    pub rotation: &'a Rotation,
    pub centroids: &'a [f32],
    pub radius: &'a [f32],
    pub offsets: &'a [u32],
    pub docs: &'a [u32],
    /// Per vector, in code order (see `Pre`).
    pub pre: &'a [Pre],
    /// The codes in FastScan's order (see `Blocked`): cluster c's blocks are
    /// `starts[c]..starts[c + 1]`, of `block` bytes each.
    pub blocks: &'a [u8],
    pub starts: &'a [u32],
    pub block: usize,
    /// Where a document's full-precision row is: `rows[doc]` (u32::MAX: none).
    pub rows: &'a [u32],
    /// Rows are in the codes' order (segments since version 8): code j's row is row j, so a
    /// cluster's rows are next to each other and a rerank needn't look a row up.
    pub by_code: bool,
    pub full: Rows<'a>,
    /// The calibrated probe budget (0: not calibrated, from before calibration existed).
    pub nprobe: usize,
    /// Probe budgets for deeper searches (nothing measured in older segments).
    pub budgets: Budgets,
    /// How its vectors are compared; and for dot product (empty otherwise) how many times its
    /// list's centroid each vector holds, in code order, and each list's longest vector's length.
    pub metric: Metric,
    pub along: &'a [f32],
    pub longest: &'a [f32],
}

/// The best k by the metric's key at full precision (`Metric`: a squared distance, or a dot
/// product negated), shared across the segments searched.
pub struct Nearest {
    pub k: usize,
    heap: BinaryHeap<(OrderedF32, u64)>,
    /// How many full-precision vectors were read (for the benchmark).
    pub reranked: usize,
    pub estimated: usize,
    /// When tracing (for finding why a neighbour was missed): each (segment ordinal, cluster)
    /// whose codes were scanned.
    pub probed: Option<Vec<(u32, u32)>>,
}

/// A distance ordered by `total_cmp`, in all comparisons alike.
#[derive(Clone, Copy)]
struct OrderedF32(f32);
impl PartialEq for OrderedF32 {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o).is_eq()
    }
}
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrderedF32 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

impl Nearest {
    pub fn new(k: usize) -> Nearest {
        Nearest { k, heap: BinaryHeap::with_capacity(k.min(1024) + 1), reranked: 0, estimated: 0, probed: None }
    }

    /// One that records the clusters it scans.
    pub fn traced(k: usize) -> Nearest {
        Nearest { probed: Some(Vec::new()), ..Nearest::new(k) }
    }

    /// The k-th best distance once there are k, else infinity.
    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.k { f32::INFINITY } else { self.heap.peek().unwrap().0.0 }
    }

    /// Keeps the k nearest, and of equally near ones those with the lowest ids, in whatever
    /// order they're offered (clusters, segments and exact scans each have their own).
    #[inline]
    pub fn offer(&mut self, d2: f32, id: u64) {
        if self.heap.len() < self.k {
            self.heap.push((OrderedF32(d2), id));
        } else if self.heap.peek().is_some_and(|worst| (OrderedF32(d2), id) < *worst) {
            self.heap.pop();
            self.heap.push((OrderedF32(d2), id));
        }
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// (squared distance, id), nearest first.
    pub fn into_sorted(self) -> Vec<(f32, u64)> {
        let mut v: Vec<(f32, u64)> = self.heap.into_iter().map(|(d, id)| (d.0, id)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        v
    }
}

/// How many candidates ahead of the one being read a search fetches rows (a million vectors on
/// an M1 Max: 0.195 ms a search without, 0.16 with, the same from 2 ahead to 32).
const AHEAD: usize = 4;

impl View<'_> {
    /// The full-precision row of the vector at place `j` in code order.
    #[inline]
    fn row(&self, j: usize) -> usize {
        if self.by_code { j } else { self.rows[self.docs[j] as usize] as usize }
    }

    /// Searches this segment's vectors for `q` (as its metric prepares a query: normalized for
    /// cosine), into `top`. `keep(doc)` says whether
    /// a document may be a result (live, and passing the filter). Clusters are probed nearest
    /// first, skipping any that can't hold a closer vector (centroid distance minus radius), and
    /// stop after `nprobe` (0: the segment's calibrated budget) once there are k results: with a
    /// selective filter, probing goes on until k pass it.
    pub fn search(&self, q: &[f32], ordinal: u32, keep: &dyn Fn(u32) -> bool, top: &mut Nearest, nprobe: usize) {
        self.search_rotated(q, &self.rotation.apply(q), ordinal, keep, None, top, nprobe, 1.0, &crate::stop::Stop::NEVER)
    }

    /// `search`, with the query already rotated (segments built with the same seed share it), giving
    /// up between clusters when `stop` says so.
    #[allow(clippy::too_many_arguments)]
    /// `passing`: the share of its vectors `keep` is expected to keep, which deepens the
    /// budget (`search_lists`); an explicit `nprobe` is used as given.
    #[allow(clippy::too_many_arguments)]
    pub fn search_rotated(&self, q: &[f32], qr: &[f32], ordinal: u32, keep: &dyn Fn(u32) -> bool, among: Option<Among<'_>>, top: &mut Nearest, nprobe: usize, passing: f64, stop: &crate::stop::Stop) {
        if self.radius.len() == 1 {
            // One cluster (a segment of under 4,096 vectors): every row at full precision is both
            // exact and faster than estimating from codes first (NFCorpus: 0.05 against 0.06 ms).
            let dims = self.rotation.dims;
            if let Some(p) = &mut top.probed {
                p.push((ordinal, 0));
            }
            if self.metric == Metric::Dot {
                for (j, &doc) in self.docs.iter().enumerate() {
                    if keep(doc) {
                        let d = -self.full.dot(q, self.row(j), dims);
                        top.reranked += 1;
                        top.offer(d, (ordinal as u64) << 32 | doc as u64);
                    }
                }
                return;
            }
            for (j, &doc) in self.docs.iter().enumerate() {
                if keep(doc) {
                    let d = self.full.dist2(q, self.row(j), dims);
                    top.reranked += 1;
                    top.offer(d, (ordinal as u64) << 32 | doc as u64);
                }
            }
            return;
        }
        search_lists(&[Member { view: *self, ordinal, keep, among }], q, qr, top, nprobe, passing, stop);
    }
}

/// One segment's part in a search of segments that share centroids.
pub struct Member<'a> {
    pub view: View<'a>,
    pub ordinal: u32,
    /// Whether a document may be a result (live, and passing the filter).
    pub keep: &'a dyn Fn(u32) -> bool,
    /// The filter's documents as bits, where they were made (`keep` still decides: it knows
    /// what is deleted too).
    pub among: Option<Among<'a>>,
}

/// What a search knows of its filter in one segment.
#[derive(Clone, Copy)]
pub struct Among<'a> {
    /// The filter's documents as bits, once they are made; and what makes them when they are
    /// first asked for, if that is cheap (from columns).
    pub bits: &'a std::cell::OnceCell<std::sync::Arc<crate::bitmap::Bitmap>>,
    pub make: Option<&'a dyn Fn() -> std::sync::Arc<crate::bitmap::Bitmap>>,
    /// The share of the segment's documents that pass the filter, and that `keep` keeps (the
    /// filter, less what is deleted).
    pub share: f64,
    pub kept: f64,
    /// Few pass: lists are masked by the bits from the first on. (Otherwise from when the
    /// threshold turns out to rule little out; see `search_lists`.)
    pub mask: bool,
}

/// The bits a member's lists are masked by, with the share they pass, if they are this round:
/// few pass, or the threshold is loose (`search_lists`).
fn masked<'a>(among: Option<Among<'a>>, loose: bool) -> Option<(&'a crate::bitmap::Bitmap, f64)> {
    let a = among.filter(|a| a.mask || loose)?;
    let bits = a.bits.get().or_else(|| a.make.map(|make| a.bits.get_or_init(make)))?;
    Some((&**bits, a.share))
}

/// How deep a search for `k` results looks when a share `passing` of the vectors may be one: the
/// depth, in the order of all vectors, within which `k` of them pass `TARGET_RECALL` of the time.
/// On average that is k / passing; but how many of the nearest D pass varies (a binomial), and
/// half the time fewer than k of the nearest k / passing do, so the k-th is deeper: the mean
/// plus 2.75 standard deviations (0.997 of a normal curve), which for 10 results under a filter
/// of a hundredth is 2,240 instead of 1,000. With everything passing, exactly k.
pub fn depth_for(k: usize, passing: f64) -> f64 {
    const Z: f64 = 2.75;
    let s = passing.clamp(1e-9, 1.0);
    let k = k.max(1) as f64;
    (k + Z * (k * (1.0 - s)).sqrt() + Z * Z * (1.0 - s) / 2.0) / s
}

/// Searches segments whose lists are of the same centroids (one generation; or one segment) as if
/// they were one: the centroids are ranked once, and for each one probed, nearest first, the
/// query's tables are made once and every member's list scanned, into one top k, so the threshold
/// that rules candidates out is the same for all of them. A member's list is skipped if it can't
/// hold anything closer (the centroid's distance minus that list's radius). Probing stops after
/// `nprobe` centroids once there are k results. 0: the budget measured for this depth, k results
/// among the share `passing` of the vectors that the members' `keep` is expected to keep
/// (`Budgets`), and more if less passes where the search looks than everywhere: a filter that
/// goes against the query (the documents near it are the ones that don't pass) is as deep as the
/// share that passes here says, not the share overall.
///
/// **Dot product** is this scan with one exact term per vector, over lists of another kind.
/// The plan had been the usual reduction to distances (⟨q, x⟩ = (‖q‖² + ‖x‖² − d²) / 2: the
/// same lists and codes, a code's estimated distance less its vector's squared length, lists
/// probed by ⟨q, c⟩). Measured on 200,000 clustered vectors of 256 dimensions, it needed 23%
/// of the codes for recall@10 of 0.986, and 93% with the vectors' lengths spread over a factor
/// of a hundred: lists by distance hold what is near, not what is long, so the vectors of the
/// greatest product are scattered over them, and no order of the centroids finds them soon.
/// What a product needs is the direction and the length apart:
/// - **Lists are of directions** (`build`): k-means over the vectors at length 1, and every
///   vector, whatever its length, in the list of the centroid nearest its direction. The same
///   data then takes 2% of the codes for 0.999, spread or not.
/// - **A vector is a multiple of its centroid plus a rest across it**, x = λc + r with r ⊥ c,
///   and the code is of r. So ⟨q, x⟩ = λ⟨q, c⟩ + ⟨q', r⟩, where q' is the query's own part
///   across c: λ is kept exactly (`along`, the one term per vector), ⟨q, c⟩ is one product per
///   list, and ⟨q', r⟩ is what the codes estimate, as they estimate ⟨q − c, x − c⟩ for a
///   distance, with the same tables, the same bound and the same scan. Neither part depends on
///   how long the vectors are, so it holds for lengths of any size and any spread.
/// - **Lists are probed by what they promise** (`promise`): how near the query's direction a
///   list is, as cosine orders them, times the length of the longest vector in it. A list is
///   skipped when the most a product in it can be is under the k-th: its longest vector's
///   length times the query's along the centroid, plus ‖q'‖ times the longest rest (its
///   radius).
/// - A rerank computes the product itself.
#[allow(clippy::too_many_arguments)]
pub fn search_lists(members: &[Member<'_>], q: &[f32], qr: &[f32], top: &mut Nearest, nprobe: usize, passing: f64, stop: &crate::stop::Stop) {
    // Compiled twice, so that the search by distance is the code it was before there was
    // another, with nothing in its loops to ask which one it is.
    match members.first().map(|m| m.view.metric) {
        Some(Metric::Dot) => lists::<true>(members, q, qr, top, nprobe, passing, stop),
        Some(Metric::Cosine | Metric::Euclidean) => lists::<false>(members, q, qr, top, nprobe, passing, stop),
        None => {}
    }
}

/// `v` at length 1 (as it is, if it has none).
fn to_unit(v: &mut [f32]) {
    let n = dot(v, v).sqrt();
    if n > 0.0 && n.is_finite() {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// For dot product: what a list is probed in the order of, least first. `unit` is the query
/// at length 1, `centroid` the list's (the mean of its vectors' directions) and `longest` the
/// length of the longest vector in it. 1 − ‖unit − centroid‖² / 2 is how near the
/// query's direction the list is: the cosine to its centroid, more for a list whose
/// directions are wide apart (a short centroid), which is the order cosine probes in. Times
/// the longest, that is about the product its best vector could have; negated, so the most
/// promising is the least.
#[inline]
fn promise(unit: &[f32], centroid: &[f32], longest: f32) -> f32 {
    -longest * (1.0 - dist2(unit, centroid) / 2.0)
}

/// `search_lists`, by distance (`DOT` false: cosine and Euclidean) or by dot product.
#[allow(clippy::too_many_arguments)]
fn lists<const DOT: bool>(members: &[Member<'_>], q: &[f32], qr: &[f32], top: &mut Nearest, nprobe: usize, passing: f64, stop: &crate::stop::Stop) {
    let Some(first) = members.first().map(|m| &m.view) else { return };
    let fixed = nprobe != 0;
    let k = top.k;
    // Segments written before calibration have no budget: the old default.
    let base = members.iter().map(|m| m.view.nprobe).max().filter(|n| *n > 0).unwrap_or(16);
    let clusters = first.radius.len();
    let vectors: usize = members.iter().map(|m| m.view.docs.len()).sum();
    let budgets = members.iter().map(|m| m.view.budgets).find(|b| b.by_depth[0] > 0);
    // Measured, or for segments from before that, the fitted formula.
    let budget_for = |passing: f64| match budgets.and_then(|b| b.probes(clusters, vectors, depth_for(k, passing))) {
        Some(probes) => probes.max(base),
        None => scaled_budget(base, k, passing),
    };
    let mut budget = if fixed { nprobe } else { budget_for(passing) };
    // Of the documents asked about so far, how many passed, and how many would have at the
    // share that passes overall.
    let (mut passed, mut expected) = (0usize, 0f64);
    // The threshold rules little out (the top k isn't full, or is deep: of the codes the last
    // round scored, a quarter were still candidates when their turn came), so each candidate
    // that doesn't pass costs more than looking its bit up first would: lists are masked by the
    // filter's bits, which are made now if they weren't.
    let mut loose = false;
    let (p, dims) = (first.rotation.padded, first.rotation.dims);
    // By dot product: the query's squared length, the query at length 1 (what lists are
    // ordered against), and against the list being scanned: its centroid's product with the
    // query, the multiple of the centroid that the query's part across it is taken from, and
    // the lengths of the query's two parts, along the centroid and across it.
    let qq = if DOT { dot(qr, qr) } else { 0.0 };
    let unit: Vec<f32> = if DOT {
        let mut unit = qr.to_vec();
        to_unit(&mut unit);
        unit
    } else {
        Vec::new()
    };
    let mut part: Vec<f32> = if DOT { vec![0.0; p] } else { Vec::new() };
    let (mut product, mut reach, mut across) = (0f32, 0f32, 0f32);
    // What centroids are probed in the order of: their distance, or what their lists promise
    // (the longest vector in a list being the longest in any member's).
    let mut order: Vec<(f32, usize)> = if DOT {
        (0..first.radius.len()).map(|c| (promise(&unit, &first.centroids[c * p..(c + 1) * p], members.iter().fold(0f32, |longest, m| longest.max(m.view.longest[c]))), c)).collect()
    } else {
        (0..first.radius.len()).map(|c| (dist2(qr, &first.centroids[c * p..(c + 1) * p]), c)).collect()
    };
    // The centroids nearest first, as far as the search goes: a search probes a few of
    // thousands, and putting all of them in order was a third of its time (a million vectors,
    // 1,400 centroids: 0.134 → 0.095 ms). So the nearest stretch is picked out and sorted, and
    // the next only if the search gets there. (By distance, then by number: one order, however
    // it is arrived at.)
    let nearer = |a: &(f32, usize), b: &(f32, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    let mut sorted = 0;
    // (estimate, lower bound, member, place in its codes).
    let mut candidates: Vec<(f32, f32, u32, u32)> = Vec::new();
    let mut qc: Option<QueryCode> = None;
    let mut weighted = [0u32; 32];
    for probed in 0..order.len() {
        if probed == sorted {
            let stretch = (2 * budget.min(order.len())).max(32).max(sorted).min(order.len() - sorted);
            if stretch < order.len() - sorted {
                order[sorted..].select_nth_unstable_by(stretch - 1, nearer);
            }
            order[sorted..sorted + stretch].sort_unstable_by(nearer);
            sorted += stretch;
        }
        let (cd2, c) = order[probed];
        // Fewer passed than chance explains (two standard deviations under what the overall
        // share would have): the budget for the share that passes here. Counted as if 4 more
        // had passed among as many as that takes overall, so a few documents don't decide; and
        // only ever deeper, so without the test every unlucky list would add to it.
        if !fixed && passed as f64 + 2.0 * expected.sqrt() < expected {
            const PRIOR: f64 = 4.0;
            let here = (passed as f64 + PRIOR) / (expected + PRIOR);
            budget = budget.max(budget_for(passing * here));
        }
        if probed >= budget && top.len() >= top.k || stop.stopped() {
            break;
        }
        candidates.clear();
        let considered = top.estimated;
        let mut asked = 0usize;
        if DOT {
            // The query is so many times this centroid, and a part across it.
            let centroid = &first.centroids[c * p..(c + 1) * p];
            let cc = dot(centroid, centroid);
            product = dot(qr, centroid);
            let times = if cc > 0.0 { product / cc } else { 0.0 };
            part.iter_mut().zip(centroid).for_each(|(x, &c)| *x = times * c);
            reach = if cc > 0.0 { product.abs() / cc.sqrt() } else { 0.0 };
            across = (qq - times * product).max(0.0).sqrt();
        }

        let mut tables = false;
        for (at, m) in members.iter().enumerate() {
            let v = &m.view;
            let (start, n) = (v.offsets[c] as usize, (v.offsets[c + 1] - v.offsets[c]) as usize);
            if DOT {
                // No product in a list is over its longest vector's length times the length
                // of the query's part along the centroid, plus the length of the query's part
                // across it times the longest rest (the list's radius).
                if n == 0 || top.len() >= top.k && -(v.longest[c] * reach + across * v.radius[c]) >= top.threshold() {
                    continue;
                }
            } else {
                // A list can't hold anything closer than its centroid's distance minus its radius.
                let lb = (cd2.sqrt() - v.radius[c]).max(0.0);
                if n == 0 || top.len() >= top.k && lb * lb >= top.threshold() {
                    continue;
                }
            }
            if let Some(p) = &mut top.probed {
                p.push((m.ordinal, c as u32));
            }
            if !tables {
                // The query's residual from the centroid; by dot product, from its own multiple
                // of the centroid, which leaves its part across it.
                let centroid = if DOT { &part[..] } else { &first.centroids[c * p..(c + 1) * p] };
                match &mut qc {
                    Some(qc) => qc.set(qr, centroid),
                    None => qc = Some(QueryCode::new(qr, centroid)),
                }
                tables = true;
            }
            let qc = qc.as_ref().unwrap();
            let mask = masked(m.among, loose);
            for (b, block) in (v.starts[c] as usize..v.starts[c + 1] as usize).enumerate() {
                let j = start + b * 32;
                let end = (j + 32).min(start + n);
                // With a filter few pass, most blocks hold no document that does (at 1%, 72% of
                // them): looking up 32 bits costs about what scoring the block does, and saves
                // estimating, sorting and asking about 32 candidates no threshold rules out yet.
                let among = match mask {
                    Some((bits, share)) => {
                        let among = v.docs[j..end].iter().enumerate().fold(0u32, |among, (i, &doc)| among | (bits.get(doc) as u32) << i);
                        passed += among.count_ones() as usize;
                        expected += (end - j) as f64 * share;
                        among
                    }
                    None => u32::MAX,
                };
                if among == 0 {
                    continue;
                }
                qc.weighted_block(&v.blocks[block * v.block..(block + 1) * v.block], &mut weighted);
                top.estimated += among.count_ones().min((end - j) as u32) as usize;
                if DOT {
                    // The codes estimate ‖q' − r‖² = ‖q'‖² + ‖r‖² − 2⟨q', r⟩, and ⟨q, x⟩ =
                    // λ⟨q, c⟩ + ⟨q', r⟩: a product, negated, is half that distance less ‖r‖²,
                    // 2⟨q, c⟩λ and ‖q'‖². The threshold in those terms for the scan, and what
                    // passes back in its own.
                    let (times, own) = (2.0 * product, qc.norm * qc.norm);
                    qc.below_along(&weighted, &v.pre[j..end], &v.along[j..end], times, 2.0 * top.threshold() + own, among, |i, est, low| {
                        let rest = v.pre[j + i].norm;
                        let off = rest * rest + v.along[j + i] * times + own;
                        candidates.push(((est - off) / 2.0, (low - off) / 2.0, at as u32, (j + i) as u32))
                    });
                } else {
                    qc.below(&weighted, &v.pre[j..end], top.threshold(), among, |i, est, low| candidates.push((est, low, at as u32, (j + i) as u32)));
                }
            }
        }
        // Most promising first, so the threshold tightens early and fewer need reading.
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Rows are fetched a few candidates ahead of being read (those the threshold as it is
        // now wouldn't already rule out: it only tightens).
        let fetch = |i: usize, threshold: f32| {
            if let Some(&(_, _, at, j)) = candidates.get(i).filter(|c| c.1 < threshold) {
                let v = &members[at as usize].view;
                v.full.prefetch(v.row(j as usize), dims);
            }
        };
        for i in 0..AHEAD.min(candidates.len()) {
            fetch(i, top.threshold());
        }
        for (i, &(_, low, at, j)) in candidates.iter().enumerate() {
            fetch(i + AHEAD, top.threshold());
            let m = &members[at as usize];
            let doc = m.view.docs[j as usize];
            if low >= top.threshold() {
                continue;
            }
            asked += 1;
            let kept = (m.keep)(doc);
            // (Where the list was masked, what passes was counted then.)
            if let Some(a) = m.among.filter(|a| masked(Some(*a), loose).is_none()) {
                passed += kept as usize;
                expected += a.kept;
            }
            if !kept {
                continue;
            }
            let d = if DOT { -m.view.full.dot(q, m.view.row(j as usize), dims) } else { m.view.full.dist2(q, m.view.row(j as usize), dims) };
            top.reranked += 1;
            top.offer(d, (m.ordinal as u64) << 32 | doc as u64);
        }
        // (Not by the first round alone: until it has found k, nothing is ruled out.)
        loose = top.len() < k || probed > 0 && asked * 4 > top.estimated - considered;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_preserves_lengths() {
        let r = Rotation::new(300, 7);
        let v: Vec<f32> = (0..300).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let x = r.apply(&v);
        let a: f32 = v.iter().map(|x| x * x).sum();
        let b: f32 = x.iter().map(|x| x * x).sum();
        assert!((a - b).abs() / a < 1e-4, "{a} {b}");
    }

    #[test]
    fn estimates_bound_the_true_distance() {
        let dims = 256;
        let rot = Rotation::new(dims, 3);
        let mut s = 99u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f32 / 1000.0 - 1.0
        };
        let mut violations = 0;
        let c = vec![0f32; rot.padded];
        for _ in 0..500 {
            let mut a: Vec<f32> = (0..dims).map(|_| rnd()).collect();
            let mut b: Vec<f32> = (0..dims).map(|_| rnd()).collect();
            normalize(&mut a);
            normalize(&mut b);
            let (ra, rb) = (rot.apply(&a), rot.apply(&b));
            let mut code = vec![0u64; rot.padded / 64];
            let f = encode(&ra, &c, &mut code);
            let (est, low) = QueryCode::new(&rb, &c).estimate(&code, f);
            let truth = dist2(&a, &b);
            assert!((est - truth).abs() < 0.5, "{est} vs {truth}");
            violations += (low > truth + 1e-3) as usize;
        }
        assert!(violations <= 2, "{violations} of 500 lower bounds above the truth");
    }

    #[test]
    fn the_least_is_the_first_of_the_least() {
        let mut s = 7u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for round in 0..2000 {
            let k = (rnd() % 70) as usize + 1;
            // Few values, so there are ties, in any place; and now and then a lone least.
            let dots: Vec<f32> = (0..k).map(|_| (rnd() % 5) as f32).collect();
            let norms: Vec<f32> = (0..k).map(|_| (rnd() % 4) as f32 * if round % 7 == 0 { 0.37 } else { 2.0 }).collect();
            let mut want = (f32::MAX, 0);
            for c in 0..k {
                let v = norms[c] - 2.0 * dots[c];
                if v < want.0 {
                    want = (v, c);
                }
            }
            assert_eq!(least(&dots, &norms), want.1, "round {round}: {dots:?} {norms:?}");
        }
        assert_eq!(least(&[], &[]), 0);
    }

    /// `gemm_nt`, whichever way it is done here, against products one by one: for rows and
    /// dimensions that aren't multiples of its tiles and registers.
    #[test]
    fn products_of_every_row_with_every_row() {
        let mut s = 3u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f32 / 1000.0 - 1.0
        };
        for (m, k, dims) in [(1, 1, 1), (3, 5, 7), (17, 9, 16), (5, 70, 100), (33, 130, 256), (70, 3, 384), (2, 6, 1030)] {
            let a: Vec<f32> = (0..m * dims).map(|_| rnd()).collect();
            let b: Vec<f32> = (0..k * dims).map(|_| rnd()).collect();
            let mut out = vec![f32::NAN; m * k];
            gemm_nt(&a, &b, dims, &mut out);
            for i in 0..m {
                for c in 0..k {
                    let want: f32 = a[i * dims..(i + 1) * dims].iter().zip(&b[c * dims..(c + 1) * dims]).map(|(x, y)| x * y).sum();
                    assert!((out[i * k + c] - want).abs() <= 1e-3 * (1.0 + want.abs()), "{m} × {k} of {dims}: row {i}, {c}: {} vs {want}", out[i * k + c]);
                }
            }
        }
    }

    #[test]
    fn nearest_all_agrees_with_nearest() {
        let mut s = 11u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f32 / 1000.0 - 1.0
        };
        for (dims, k) in [(64, 1), (64, 7), (256, 33), (100, 300)] {
            let rows: Vec<f32> = (0..700 * dims).map(|_| rnd()).collect();
            let cents: Vec<f32> = (0..k * dims).map(|_| rnd()).collect();
            let all = nearest_all(&rows, &cents, dims);
            for (i, r) in rows.chunks_exact(dims).enumerate() {
                let (a, b) = (all[i], nearest(r, &cents, dims));
                // The same, or as near to rounding.
                assert!(a == b || (dist2(r, &cents[a * dims..(a + 1) * dims]) - dist2(r, &cents[b * dims..(b + 1) * dims])).abs() < 1e-4, "{dims} {k} row {i}");
            }
        }
    }

    #[test]
    fn rotation_is_the_one_segments_were_built_with() {
        // The implementation segments before 0.3 were built with, which queries must match.
        fn old(dims: usize, seed: u64, v: &[f32]) -> Vec<f32> {
            let padded = dims.next_power_of_two().max(64);
            let mut s = seed | 1;
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            };
            let signs: Vec<[f32; 3]> = (0..padded).map(|_| std::array::from_fn(|_| if next() & 1 == 0 { 1.0 } else { -1.0 })).collect();
            let mut x = vec![0f32; padded];
            x[..dims].copy_from_slice(&v[..dims]);
            let scale = 1.0 / (padded as f32).sqrt();
            for round in 0..3 {
                for (xi, s) in x.iter_mut().zip(&signs) {
                    *xi *= s[round];
                }
                let mut h = 1;
                while h < padded {
                    for i in (0..padded).step_by(h * 2) {
                        for j in i..i + h {
                            let (a, b) = (x[j], x[j + h]);
                            x[j] = a + b;
                            x[j + h] = a - b;
                        }
                    }
                    h *= 2;
                }
                x.iter_mut().for_each(|xi| *xi *= scale);
            }
            x
        }
        for dims in [8, 100, 256, 768] {
            let seed = 0x5EED_F0F0 ^ dims as u64;
            let v: Vec<f32> = (0..dims).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
            let (a, b) = (Rotation::new(dims, seed).apply(&v), old(dims, seed, &v));
            assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5), "{dims}");
        }
    }

    #[test]
    fn halves_round_trip_and_round_to_even() {
        // Every half converts to a float and back to itself (NaNs aside).
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if !f.is_nan() {
                assert_eq!(f32_to_f16(f), h, "{h:#x}");
            }
        }
        // The scalar conversion is the hardware's, on values around every rounding case.
        let mut s = 3u64;
        let xs: Vec<f32> = (0..200_000)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                match i % 3 {
                    0 => f32::from_bits(s as u32),
                    1 => (s % 20_000) as f32 / 10_000.0 - 1.0,
                    _ => ((s % 2000) as f32 - 1000.0) * 1e-7,
                }
            })
            .filter(|x| !x.is_nan())
            .collect();
        let hw = to_f16(&xs);
        for (x, h) in xs.iter().zip(&hw) {
            assert_eq!(f32_to_f16(*x), *h, "{x:e}");
        }
        // Distances against half rows: as against the rows they stand for.
        let a: Vec<f32> = xs[..1000].iter().map(|x| x.clamp(-1.0, 1.0)).collect();
        let b: Vec<f32> = xs[1000..2000].iter().map(|x| x.clamp(-1.0, 1.0)).collect();
        let hb = to_f16(&b);
        let back: Vec<f32> = hb.iter().map(|&h| f16_to_f32(h)).collect();
        for n in [16, 100, 256, 1000] {
            let (x, y) = (dist2_f16(&a[..n], &hb[..n]), dist2(&a[..n], &back[..n]));
            assert!((x - y).abs() <= 1e-4 * y.max(1.0), "{n}: {x} vs {y}");
        }
    }

    /// The AVX2 kernels against plain arithmetic, at lengths with every kind of tail.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_kernels_match_plain_arithmetic() {
        if !x86::avx2() {
            eprintln!("no AVX2, FMA and F16C here: the portable code is what runs");
            return;
        }
        eprintln!("AVX2 kernels in use");
        let mut s = 7u64;
        let mut unit = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 20_001) as f32 / 10_000.0 - 1.0
        };
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 255, 256, 257, 1000, 1027] {
            let a: Vec<f32> = (0..n).map(|_| unit()).collect();
            let b: Vec<f32> = (0..n).map(|_| unit()).collect();
            let want: f64 = a.iter().zip(&b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum();
            // SAFETY: checked above.
            let got = unsafe { x86::dist2(&a, &b) } as f64;
            assert!((got - want).abs() <= 1e-5 * want.max(1.0), "dist2 of {n}: {got} vs {want}");
            let halves: Vec<u16> = b.iter().map(|&x| f32_to_f16(x)).collect();
            assert_eq!(to_f16(&b), halves, "to_f16 of {n}");
            let want: f64 = a.iter().zip(&halves).map(|(x, &h)| (*x as f64 - f16_to_f32(h) as f64).powi(2)).sum();
            let got = unsafe { x86::dist2_f16(&a, &halves) } as f64;
            assert!((got - want).abs() <= 1e-5 * want.max(1.0), "dist2_f16 of {n}: {got} vs {want}");
        }
        // The loops compiled twice give what they give compiled once.
        let (dims, m, k) = (100, 7, 9);
        let rows: Vec<f32> = (0..m * dims).map(|_| unit()).collect();
        let cents: Vec<f32> = (0..k * dims).map(|_| unit()).collect();
        let (mut wide, mut plain) = (vec![0f32; m * k], vec![0f32; m * k]);
        unsafe { x86::gemm_nt(&rows, &cents, dims, &mut wide) };
        gemm_nt_portable(&rows, &cents, dims, &mut plain);
        for (w, p) in wide.iter().zip(&plain) {
            assert!((w - p).abs() <= 1e-4 * p.abs().max(1.0), "gemm_nt: {w} vs {p}");
        }
        let rot = Rotation::new(dims, 11);
        let (x, c) = (rot.apply(&rows[..dims]), rot.apply(&cents[..dims]));
        for (w, p) in x.iter().zip(rot.apply_ref_portable(VectorRef::F32(&rows[..dims]))) {
            assert!((w - p).abs() <= 1e-5, "rotation: {w} vs {p}");
        }
        let words = rot.padded / 64;
        let (mut wide, mut plain) = (vec![0u64; words], vec![0u64; words]);
        let (fw, fp) = (encode(&x, &c, &mut wide), encode_portable(&x, &c, &mut plain));
        assert_eq!(wide, plain, "codes");
        assert!((fw.norm - fp.norm).abs() <= 1e-5 && (fw.ip - fp.ip).abs() <= 1e-5, "factors");
    }

    /// Four bounds at a time pick exactly the codes one bound at a time does, with their numbers.
    #[test]
    fn bounds_four_at_a_time_are_the_bounds_one_at_a_time() {
        let mut s = 99u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let p = 256;
        let unit = |x: u64| (x % 20_001) as f32 / 10_000.0 - 1.0;
        let q: Vec<f32> = (0..p).map(|_| unit(next())).collect();
        let c: Vec<f32> = (0..p).map(|_| 0.3 * unit(next())).collect();
        let qc = QueryCode::new(&q, &c);
        for round in 0..2000 {
            let n = (next() % 33) as usize;
            let weighted: [u32; 32] = std::array::from_fn(|_| (next() % (15 * p as u64)) as u32);
            let pre: Vec<Pre> = (0..n)
                .map(|_| {
                    let ip = 0.5 + (next() % 500) as f32 / 1000.0;
                    Pre { inv_ip: 1.0 / ip, err: (1.0 - ip * ip).sqrt() / ip, norm: (next() % 2000) as f32 / 1000.0, ones: (next() % p as u64) as f32 }
                })
                .collect();
            let one_by_one: Vec<(f32, f32)> = (0..n).map(|i| qc.estimate_weighted(weighted[i], &pre[i])).collect();
            // Thresholds: none pass, all pass, and bounds themselves (the comparison is strict).
            let mut thresholds = vec![0.0, f32::INFINITY, 1.0, 2.5];
            thresholds.extend(one_by_one.iter().map(|b| b.1));
            for t in thresholds {
                // Among all of them, a few, one in two, none.
                for among in [u32::MAX, next() as u32 & next() as u32 & next() as u32, next() as u32, 0] {
                    let want: Vec<(usize, f32, f32)> = one_by_one.iter().enumerate().filter(|(i, b)| b.1 < t && among >> i & 1 == 1).map(|(i, b)| (i, b.0, b.1)).collect();
                    let mut got = Vec::new();
                    qc.below(&weighted, &pre, t, among, |i, est, low| got.push((i, est, low)));
                    assert_eq!(got, want, "round {round}, {n} codes, threshold {t}, among {among:032b}");
                }
            }
        }
    }

    /// `below_along` picks exactly the codes its condition picks one at a time, with their
    /// numbers: four at a time and in the tail, among all codes and among some.
    #[test]
    fn bounds_less_a_vectors_own_terms_four_at_a_time_are_those_one_at_a_time() {
        let mut s = 77u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let p = 256;
        let unit = |x: u64| (x % 20_001) as f32 / 10_000.0 - 1.0;
        let q: Vec<f32> = (0..p).map(|_| unit(next())).collect();
        let c: Vec<f32> = (0..p).map(|_| 0.3 * unit(next())).collect();
        let qc = QueryCode::new(&q, &c);
        for round in 0..2000 {
            let n = (next() % 33) as usize;
            let weighted: [u32; 32] = std::array::from_fn(|_| (next() % (15 * p as u64)) as u32);
            let pre: Vec<Pre> = (0..n)
                .map(|_| {
                    let ip = 0.5 + (next() % 500) as f32 / 1000.0;
                    Pre { inv_ip: 1.0 / ip, err: (1.0 - ip * ip).sqrt() / ip, norm: (next() % 2000) as f32 / 1000.0, ones: (next() % p as u64) as f32 }
                })
                .collect();
            // Lengths along a centroid of either sign, and a product with it of either sign.
            let along: Vec<f32> = (0..n).map(|_| 3.0 * unit(next())).collect();
            let times = 5.0 * unit(next());
            let one_by_one: Vec<(f32, f32, f32)> = (0..n)
                .map(|i| {
                    let (est, low) = qc.estimate_weighted(weighted[i], &pre[i]);
                    (est, low, low - pre[i].norm * pre[i].norm - along[i] * times)
                })
                .collect();
            // Thresholds: none pass, all pass, some, and the compared values themselves (the
            // comparison is strict).
            let mut thresholds = vec![f32::NEG_INFINITY, f32::INFINITY, 0.0, -2.5, 4.0];
            thresholds.extend(one_by_one.iter().map(|b| b.2));
            for t in thresholds {
                for among in [u32::MAX, next() as u32 & next() as u32 & next() as u32, next() as u32, 0] {
                    let want: Vec<(usize, f32, f32)> = one_by_one.iter().enumerate().filter(|(i, b)| b.2 < t && among >> i & 1 == 1).map(|(i, b)| (i, b.0, b.1)).collect();
                    let mut got = Vec::new();
                    qc.below_along(&weighted, &pre, &along, times, t, among, |i, est, low| got.push((i, est, low)));
                    assert_eq!(got, want, "round {round}, {n} codes, threshold {t}, among {among:032b}");
                }
            }
        }
    }

    /// `dot` and `dot_f16`, whichever way they are done here, against plain arithmetic, at
    /// lengths with every kind of tail.
    #[test]
    fn products_match_plain_arithmetic() {
        let mut s = 5u64;
        let mut unit = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 20_001) as f32 / 10_000.0 - 1.0
        };
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 255, 256, 257, 1000, 1027] {
            // (Lengths of every size: the vectors these are for aren't normalized.)
            for size in [1.0f32, 1e-3, 300.0] {
                let a: Vec<f32> = (0..n).map(|_| size * unit()).collect();
                let b: Vec<f32> = (0..n).map(|_| size * unit()).collect();
                let scale: f64 = a.iter().zip(&b).map(|(x, y)| (*x as f64 * *y as f64).abs()).sum::<f64>().max(1e-30);
                let want: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
                assert!((dot(&a, &b) as f64 - want).abs() <= 1e-5 * scale, "dot of {n} at {size}: {} vs {want}", dot(&a, &b));
                let halves = to_f16(&b);
                let want: f64 = a.iter().zip(&halves).map(|(x, &h)| *x as f64 * f16_to_f32(h) as f64).sum();
                assert!((dot_f16(&a, &halves) as f64 - want).abs() <= 1e-5 * scale, "dot_f16 of {n} at {size}: {} vs {want}", dot_f16(&a, &halves));
                assert_eq!(VectorRef::F16(&halves).dot(&a), dot_f16(&a, &halves));
                assert_eq!(Rows::F16(&halves).key(Metric::Dot, &a, 0, n), -dot_f16(&a, &halves));
                assert_eq!(Rows::F16(&halves).key(Metric::Euclidean, &a, 0, n), dist2_f16(&a, &halves));
            }
        }
    }

    /// The AVX2 products against plain arithmetic (the portable ones are checked above, where
    /// they are what runs).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_products_match_plain_arithmetic() {
        if !x86::avx2() {
            eprintln!("no AVX2, FMA and F16C here: the portable code is what runs");
            return;
        }
        let mut s = 9u64;
        let mut unit = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 20_001) as f32 / 10_000.0 - 1.0
        };
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 255, 256, 257, 1000, 1027] {
            let a: Vec<f32> = (0..n).map(|_| unit()).collect();
            let b: Vec<f32> = (0..n).map(|_| unit()).collect();
            let want: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            // SAFETY: checked above.
            let got = unsafe { x86::dot(&a, &b) } as f64;
            assert!((got - want).abs() <= 1e-5 * (n as f64).max(1.0), "dot of {n}: {got} vs {want}");
            let halves: Vec<u16> = b.iter().map(|&x| f32_to_f16(x)).collect();
            let want: f64 = a.iter().zip(&halves).map(|(x, &h)| *x as f64 * f16_to_f32(h) as f64).sum();
            let got = unsafe { x86::dot_f16(&a, &halves) } as f64;
            assert!((got - want).abs() <= 1e-5 * (n as f64).max(1.0), "dot_f16 of {n}: {got} vs {want}");
        }
    }

    /// Dot product as a search works it out: a vector is a multiple of its centroid and a rest
    /// across it, the query likewise, and ⟨q, x⟩ is the multiple times ⟨q, c⟩ plus the product
    /// of the two rests, which the code estimates and bounds. For vectors of every length, the
    /// estimate is near the product and the bound is above it, nearly always.
    #[test]
    fn a_product_is_bounded_by_its_estimate_across_the_centroid() {
        let dims = 256;
        let rot = Rotation::new(dims, 3);
        let mut s = 31u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f32 / 1000.0 - 1.0
        };
        let (mut violations, mut far) = (0, 0);
        for round in 0..500 {
            // A centroid as lists by direction have them (shorter than 1), a vector and a
            // query somewhere near it, each of a length of its own over four orders of
            // magnitude.
            let centre: Vec<f32> = (0..dims).map(|_| rnd()).collect();
            let length = |i: usize| 10f32.powi((round + i) as i32 % 5 - 2);
            let mut c: Vec<f32> = centre.clone();
            normalize(&mut c);
            c.iter_mut().for_each(|x| *x *= 0.7);
            let x: Vec<f32> = centre.iter().map(|t| length(0) * (t + 0.8 * rnd())).collect();
            let q: Vec<f32> = centre.iter().map(|t| length(2) * (t + 0.8 * rnd())).collect();
            let (rc, rx, rq) = (rot.apply(&c), rot.apply(&x), rot.apply(&q));
            let cc = dot(&rc, &rc);
            // The vector's part along the centroid, and the code of the rest.
            let along = dot(&rx, &rc) / cc;
            let part: Vec<f32> = rc.iter().map(|v| along * v).collect();
            let mut code = vec![0u64; rot.padded / 64];
            let f = encode(&rx, &part, &mut code);
            // The query's, and the tables of its rest.
            let product = dot(&rq, &rc);
            let mine: Vec<f32> = rc.iter().map(|v| product / cc * v).collect();
            let qc = QueryCode::new(&rq, &mine);
            let (est, low) = qc.estimate(&code, f);
            // (As `search_lists` turns a distance between the rests into a product, negated.)
            let off = f.norm * f.norm + along * 2.0 * product + qc.norm * qc.norm;
            let (estimate, at_most) = (-(est - off) / 2.0, -(low - off) / 2.0);
            let truth = dot(&q, &x);
            let scale = dot(&q, &q).sqrt() * dot(&x, &x).sqrt();
            far += ((estimate - truth).abs() > 0.15 * scale) as usize;
            violations += (at_most < truth - 1e-3 * scale) as usize;
        }
        assert!(far <= 2, "{far} of 500 estimates far from the product");
        assert!(violations <= 2, "{violations} of 500 upper bounds below the product");
    }

    #[test]
    fn metrics_say_what_they_are_and_what_a_key_scores() {
        for metric in [Metric::Cosine, Metric::Euclidean, Metric::Dot] {
            assert_eq!(Metric::from_name(metric.name()), Some(metric));
            assert_eq!(Metric::from_code(metric.code()), Some(metric));
            for key in [0.0f32, 0.5, 2.0, -3.0] {
                assert!((metric.key(metric.score(key)) - key).abs() < 1e-6);
            }
        }
        assert_eq!((Metric::default(), Metric::Cosine.code()), (Metric::Cosine, 0), "no metric named is cosine");
        assert_eq!((Metric::from_name("manhattan"), Metric::from_code(3)), (None, None));
        // A nearer vector scores higher under each: a key is a distance, or a product negated.
        assert!(Metric::Cosine.score(0.2) > Metric::Cosine.score(0.4));
        assert_eq!((Metric::Euclidean.score(4.0), Metric::Dot.score(-7.0)), (-4.0, 7.0));
        // Cosine normalizes and refuses what has no direction; the others take vectors as
        // they come, and none takes one that isn't finite.
        let mut v = [3.0f32, 4.0];
        assert!(Metric::Cosine.prepare(&mut v) && (v[0] - 0.6).abs() < 1e-6);
        let mut v = [3.0f32, 4.0];
        assert!(Metric::Dot.prepare(&mut v) && Metric::Euclidean.prepare(&mut v) && v == [3.0, 4.0]);
        assert!(!Metric::Cosine.prepare(&mut [0.0, 0.0]) && Metric::Euclidean.prepare(&mut [0.0, 0.0]) && Metric::Dot.prepare(&mut [0.0, 0.0]));
        assert!([Metric::Cosine, Metric::Euclidean, Metric::Dot].iter().all(|m| !m.prepare(&mut [f32::NAN, 1.0]) && !m.prepare(&mut [f32::INFINITY, 1.0])));
    }

    /// Lists by direction are probed as cosine probes them while their longest vectors are
    /// as long as each other, and a list of longer vectors before one as near of shorter.
    #[test]
    fn lists_are_probed_by_what_they_promise() {
        let mut s = 13u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f32 / 1000.0 - 1.0
        };
        let dims = 64;
        let mut q: Vec<f32> = (0..dims).map(|_| rnd()).collect();
        to_unit(&mut q);
        let centroids: Vec<Vec<f32>> = (0..50).map(|_| (0..dims).map(|_| 0.1 * rnd()).collect()).collect();
        let by = |key: &dyn Fn(&Vec<f32>) -> f32| {
            let mut order: Vec<usize> = (0..centroids.len()).collect();
            order.sort_by(|a, b| key(&centroids[*a]).total_cmp(&key(&centroids[*b])));
            order
        };
        assert_eq!(by(&|c| promise(&q, c, 3.0)), by(&|c| dist2(&q, c)));
        let near = &centroids[by(&|c| dist2(&q, c))[0]];
        assert!(promise(&q, near, 2.0) < promise(&q, near, 1.0));
        assert_eq!(promise(&q, near, 0.0), 0.0, "an empty list promises nothing");
    }

    #[test]
    fn lookup_tables_match_the_scalar_ones() {
        let q: Vec<f32> = (0..256).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let qc = QueryCode::new(&q, &vec![0f32; 256]);
        for g in 0..64 {
            let mut want = [0u8; 16];
            lut16(&[qc.u[4 * g], qc.u[4 * g + 1], qc.u[4 * g + 2], qc.u[4 * g + 3]], &mut want);
            assert_eq!(&qc.lut[16 * g..16 * g + 16], &want, "group {g}");
        }
    }

    #[test]
    fn measured_budgets_are_read_between_depths() {
        let mut by_depth = [0; DEPTHS];
        by_depth[..4].copy_from_slice(&[8, 12, 30, 90]);
        let b = Budgets { by_depth, measured_on: 100_000 };
        let at = |depth: f64| b.probes(400, 100_000, depth).unwrap();
        // At and under the shallowest, its budget; at a measured depth, that depth's.
        assert_eq!([at(1.0), at(10.0), at(20.0), at(40.0), at(80.0)], [8, 8, 12, 30, 90]);
        // Between two, between theirs; never fewer for deeper.
        let deeper: Vec<usize> = (0..70).map(|i| at(10.0 * 1.2f64.powi(i))).collect();
        assert!(deeper.windows(2).all(|w| w[0] <= w[1]), "{deeper:?}");
        assert!((13..30).contains(&at(30.0)), "{}", at(30.0));
        // Past the last measured: towards every cluster for every vector, and no more than that.
        assert!((91..400).contains(&at(5000.0)), "{}", at(5000.0));
        assert_eq!([at(100_000.0), at(1e12)], [400, 400]);
        // Twice the vectors in the same lists: the same neighbourhood is twice as deep.
        assert_eq!(b.probes(400, 200_000, 80.0), Some(30));
        // Nothing measured: nothing to say.
        assert_eq!(Budgets::default().probes(400, 1000, 10.0), None);
    }

    #[test]
    fn a_filter_is_as_deep_as_its_kth_passing_neighbour_mostly_is() {
        // Everything passes: the k nearest.
        assert_eq!([depth_for(10, 1.0), depth_for(1000, 1.0)], [10.0, 1000.0]);
        // A share passes: deeper than k / share, by less the more results are asked for.
        for (k, share) in [(10, 0.5), (10, 0.01), (100, 0.01), (1000, 0.1)] {
            let (depth, mean) = (depth_for(k, share), k as f64 / share);
            assert!(depth > mean, "{k} among {share}");
            // How many of the nearest `depth` pass: that many trials of `share`. `k` is 2.75
            // standard deviations under its mean, within the rounding of the formula.
            let (passes, deviation) = (depth * share, (depth * share * (1.0 - share)).sqrt());
            assert!(((passes - k as f64) / deviation - 2.75).abs() < 0.15, "{k} among {share}: {depth}");
        }
        assert!(depth_for(100, 0.01) / 10_000.0 < depth_for(10, 0.01) / 1000.0);
        assert!(depth_for(10, 0.0).is_finite());
    }

    #[test]
    fn budgets_deepen_with_depth_only() {
        // The calibrated case keeps its budget; deeper searches get more, never less.
        assert_eq!(scaled_budget(64, 10, 1.0), 64);
        assert_eq!(scaled_budget(64, 1, 1.0), 64);
        let deeper = [scaled_budget(64, 100, 1.0), scaled_budget(64, 1000, 1.0), scaled_budget(64, 1000, 0.1)];
        assert!(deeper.windows(2).all(|w| w[0] <= w[1]) && deeper[0] > 64, "{deeper:?}");
        assert_eq!(scaled_budget(64, 100, 1.0), scaled_budget(64, 10, 0.1), "k results among a share s look as deep as k / s among all");
        assert!(scaled_budget(64, 10, 0.0) < usize::MAX, "a share of 0 is bounded");
    }

    #[test]
    fn block_scans_match_code_by_code() {
        let mut s = 5u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // Past 4,096 too, where sums no longer fit in 16 bits.
        for p in [64, 128, 256, 1024, 4096, 8192, 16384] {
            let words = p / 64;
            let random: Vec<f32> = (0..p).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect();
            // And nearly every value at its maximum (all-ones codes, the first, sum the most).
            let high: Vec<f32> = (0..p).map(|i| if i == 0 { -1.0 } else { 1.0 }).collect();
            for q in [random, high] {
                let qc = QueryCode::new(&q, &vec![0f32; p]);
                // Two clusters, of 45 and 7 codes: a full block, a partial one, and a lone partial one.
                let offsets = [0u32, 45, 52];
                let codes: Vec<u64> = (0..52 * words).map(|i| if i < words { u64::MAX } else if i < 2 * words { 0 } else { rnd() }).collect();
                let blocked = Blocked::new(&codes, &offsets, words);
                let mut weighted = [0u32; 32];
                for c in 0..2 {
                    for (b, at) in (blocked.starts[c] as usize..blocked.starts[c + 1] as usize).enumerate() {
                        qc.weighted_block(&blocked.bytes[at * blocked.block..(at + 1) * blocked.block], &mut weighted);
                        let n = (offsets[c + 1] - offsets[c]) as usize;
                        for (i, &w) in weighted.iter().enumerate() {
                            let j = offsets[c] as usize + b * 32 + i;
                            let want = if b * 32 + i < n { qc.weighted_scalar(&codes[j * words..(j + 1) * words]) } else { 0 };
                            assert_eq!(w, want, "{p} dimensions, cluster {c}, code {}", b * 32 + i);
                        }
                    }
                }
            }
        }
    }
}
