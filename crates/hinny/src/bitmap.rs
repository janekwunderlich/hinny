//! A fixed-size set of document numbers (deleted passages of a segment).

#[derive(Clone, Debug, Default)]
pub struct Bitmap {
    words: Vec<u64>,
    count: usize,
}

impl Bitmap {
    pub fn new(len: usize) -> Bitmap {
        Bitmap { words: vec![0; len.div_ceil(64)], count: 0 }
    }

    #[inline]
    pub fn get(&self, i: u32) -> bool {
        self.words.get(i as usize / 64).is_some_and(|w| w >> (i % 64) & 1 == 1)
    }

    /// Sets bit `i`; whether it was newly set.
    pub fn set(&mut self, i: u32) -> bool {
        let (w, b) = (i as usize / 64, i % 64);
        if w >= self.words.len() {
            self.words.resize(w + 1, 0);
        }
        let new = self.words[w] >> b & 1 == 0;
        self.words[w] |= 1 << b;
        self.count += new as usize;
        new
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// The set bits, in order.
    pub fn ones(&self) -> impl Iterator<Item = u32> + '_ {
        self.words.iter().enumerate().flat_map(|(w, &word)| {
            let mut bits = word;
            std::iter::from_fn(move || {
                (bits != 0).then(|| {
                    let b = bits.trailing_zeros();
                    bits &= bits - 1;
                    (w * 64) as u32 + b
                })
            })
        })
    }

    /// A bitmap of `len` from one bit per word of `f(word index)`'s 64 bits.
    pub fn from_words(len: usize, f: impl Fn(usize) -> u64) -> Bitmap {
        let mut words: Vec<u64> = (0..len.div_ceil(64)).map(f).collect();
        if len % 64 != 0 {
            *words.last_mut().unwrap() &= (1u64 << (len % 64)) - 1;
        }
        let count = words.iter().map(|w| w.count_ones() as usize).sum();
        Bitmap { words, count }
    }

    /// A bitmap of `len` whose words are `words` (one per 64 bits; bits past `len` are dropped).
    pub fn of_words(len: usize, mut words: Vec<u64>) -> Bitmap {
        words.resize(len.div_ceil(64), 0);
        if len % 64 != 0 {
            *words.last_mut().unwrap() &= (1u64 << (len % 64)) - 1;
        }
        let count = words.iter().map(|w| w.count_ones() as usize).sum();
        Bitmap { words, count }
    }

    /// Bits set in both, either, or not in `self` (of `len`).
    pub fn and(&self, o: &Bitmap) -> Bitmap {
        Bitmap::from_words(self.words.len() * 64, |i| self.words[i] & o.words.get(i).copied().unwrap_or(0))
    }

    pub fn or(&self, o: &Bitmap) -> Bitmap {
        let n = self.words.len().max(o.words.len());
        Bitmap::from_words(n * 64, |i| self.words.get(i).copied().unwrap_or(0) | o.words.get(i).copied().unwrap_or(0))
    }

    pub fn not(&self, len: usize) -> Bitmap {
        Bitmap::from_words(len, |i| !self.words.get(i).copied().unwrap_or(0))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    pub fn from_bytes(bytes: &[u8], len: usize) -> Bitmap {
        let mut words: Vec<u64> = bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        words.resize(len.div_ceil(64), 0);
        let count = words.iter().map(|w| w.count_ones() as usize).sum();
        Bitmap { words, count }
    }
}

/// Deletions with the change that made them. A view at sequence `seq` sees a passage unless it was
/// deleted at or before `seq` (Hekaton's end timestamps), so the writer marks deletions while
/// readers read, with no copying and no lock, and views from before a delete still see the passage.
///
/// About one bit per passage: deletions from before the segment or chunk was published (all a view
/// could see is after them) are a plain bitmap, and only deletions after it carry their sequence
/// number, in pages of 1,024 stamps allocated where such a delete lands.
pub struct Deletes {
    base: Bitmap,
    pages: Vec<std::sync::OnceLock<Box<[std::sync::atomic::AtomicU64]>>>,
    len: usize,
    /// How many stamps there are: `count` is asked on every write (the merge policy, the view
    /// published) and must not look at every passage.
    stamped: std::sync::atomic::AtomicUsize,
}

const PAGE: usize = 1024;

impl Deletes {
    pub fn new(len: usize) -> Deletes {
        Deletes { base: Bitmap::new(len), pages: (0..len.div_ceil(PAGE)).map(|_| std::sync::OnceLock::new()).collect(), len, stamped: std::sync::atomic::AtomicUsize::new(0) }
    }

    /// From saved deletions (deleted before any view).
    pub fn from_bitmap(b: &Bitmap, len: usize) -> Deletes {
        let mut d = Deletes::new(len);
        for i in b.ones().take_while(|&i| (i as usize) < len) {
            d.base.set(i);
        }
        d
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Marks `doc` deleted for every view (before it's published: carrying deletions over).
    pub fn mark(&mut self, doc: u32) {
        self.base.set(doc);
    }

    /// Whether `doc` is deleted as of the newest change.
    #[inline]
    pub fn deleted(&self, doc: u32) -> bool {
        self.base.get(doc) || self.stamp(doc) != 0
    }

    /// The sequence number a published passage was deleted at (0: not since publication).
    #[inline]
    fn stamp(&self, doc: u32) -> u64 {
        match self.pages.get(doc as usize / PAGE).and_then(|p| p.get()) {
            Some(page) => page[doc as usize % PAGE].load(std::sync::atomic::Ordering::Acquire),
            None => 0,
        }
    }

    /// Whether a view at `seq` sees `doc`.
    #[inline]
    pub fn visible(&self, doc: u32, seq: u64) -> bool {
        if self.base.get(doc) {
            return false;
        }
        let s = self.stamp(doc);
        s == 0 || s > seq
    }

    /// Deletes a published `doc` as of `seq`; false if it already was.
    pub fn delete(&self, doc: u32, seq: u64) -> bool {
        if self.base.get(doc) {
            return false;
        }
        let page = self.pages[doc as usize / PAGE].get_or_init(|| (0..PAGE).map(|_| std::sync::atomic::AtomicU64::new(0)).collect());
        let new = page[doc as usize % PAGE].compare_exchange(0, seq, std::sync::atomic::Ordering::Release, std::sync::atomic::Ordering::Relaxed).is_ok();
        if new {
            self.stamped.fetch_add(1, std::sync::atomic::Ordering::Release);
        }
        new
    }

    /// Deleted now (as of the newest change), for saving.
    pub fn to_bitmap(&self) -> Bitmap {
        let mut b = self.base.clone();
        for (p, page) in self.pages.iter().enumerate() {
            let Some(page) = page.get() else { continue };
            for (i, s) in page.iter().enumerate() {
                if s.load(std::sync::atomic::Ordering::Acquire) != 0 {
                    b.set((p * PAGE + i) as u32);
                }
            }
        }
        b
    }

    /// How many are deleted now: kept, not counted. (A view's count is this when the view is made;
    /// see `View::deleted`.)
    pub fn count(&self) -> usize {
        self.base.count() + self.stamped.load(std::sync::atomic::Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The count kept agrees with looking at every passage: with deletions from before
    /// publication, stamps on several pages, and passages deleted twice.
    #[test]
    fn the_count_kept_matches_a_scan() {
        let len = 5 * PAGE + 17;
        let mut d = Deletes::new(len);
        for doc in [0u32, 1, 700, 1023, 1024, 4000, (len - 1) as u32] {
            d.mark(doc);
        }
        let scan = |d: &Deletes| (0..len as u32).filter(|&i| d.deleted(i)).count();
        assert_eq!(d.count(), scan(&d));
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for seq in 2..3000u64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let doc = (x % len as u64) as u32;
            let was = d.deleted(doc);
            assert_eq!(d.delete(doc, seq), !was);
            if seq % 500 == 0 {
                assert_eq!(d.count(), scan(&d));
            }
        }
        assert_eq!(d.count(), scan(&d));
        assert_eq!(d.count(), d.to_bitmap().count());
        let reopened = Deletes::from_bitmap(&d.to_bitmap(), len);
        assert_eq!(reopened.count(), d.count());
        assert_eq!((0..len as u32).filter(|&i| !reopened.visible(i, 0)).count(), d.count());
    }
}
