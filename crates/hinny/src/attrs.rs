//! Attributes: typed values stored with each document (a date, a sender, a path, a kind), and the
//! filters a query can put on them.
//!
//! A segment stores each attribute as a column of codes, one per document, as narrow as the
//! column's values allow (`Layout`): an integer as how far it is above the column's smallest, in
//! steps of what they all share; a string as its place in the column's sorted dictionary. Zero
//! is "no value" (and an empty string is none). With them, the smallest and largest code of
//! every 1,024 documents. An attribute of several strings (tags) is a dictionary too, with each
//! document's codes one after another and, per string, the documents that have it. A condition on a column becomes runs of codes, checked a column at a
//! time on the bytes as stored (`codes`): a zone whose codes are all inside a run or outside all
//! isn't read. A condition on strings is answered by the dictionary (one string by binary
//! search, a prefix as the ranges that start with it, a part by one pass over its bytes), not
//! by each document's string. (Before segment version 8: integers as 8 bytes per document, `i64::MIN` for none,
//! and strings as an offset per document into a byte arena; still read.)
//!
//! A filter is bound to a segment's columns once per query, then checked per candidate document
//! or turned into the set of documents it matches. Search applies it when a document would enter
//! the top k, so a filtered-out document never raises the threshold and pruning stays exact.

/// An attribute's value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Str(String),
    /// A number with a fraction (a score, a price, a coordinate). Never NaN in an index.
    Float(f64),
    /// Several strings (labels, owners, the groups that may read it): a set, so each once and in
    /// byte order when read back, and an empty string is none.
    Tags(Vec<String>),
}

impl Value {
    pub fn kind(&self) -> Kind {
        match self {
            Value::Int(_) => Kind::Int,
            Value::Str(_) => Kind::Str,
            Value::Float(_) => Kind::Float,
            Value::Tags(_) => Kind::Tags,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Int = 1,
    Str = 2,
    Float = 3,
    Tags = 4,
}

impl Kind {
    pub fn from_u8(b: u8) -> Option<Kind> {
        match b {
            1 => Some(Kind::Int),
            2 => Some(Kind::Str),
            3 => Some(Kind::Float),
            4 => Some(Kind::Tags),
            _ => None,
        }
    }
}

/// A float as the integer a column keeps for it: in the same order, so a column of floats is a
/// column of integers to everything that compares (ranges, zones, the order of results), and
/// only what hands a value out turns it back. The bits of a float are in order as an integer
/// once those of negative ones are flipped below the sign; −0.0 is kept as 0.0, which it equals.
/// NaN has no place in an order: it is `NO_INT`, no value.
pub fn float_code(x: f64) -> i64 {
    if x.is_nan() {
        return NO_INT;
    }
    let bits = (if x == 0.0 { 0.0 } else { x }).to_bits() as i64;
    bits ^ (((bits >> 63) as u64) >> 1) as i64
}

/// The float `float_code` made `code` of.
pub fn code_float(code: i64) -> f64 {
    f64::from_bits((code ^ (((code >> 63) as u64) >> 1) as i64) as u64)
}

/// The integer that means "no value" in a column.
pub const NO_INT: i64 = i64::MIN;

/// A condition on attributes. String comparisons are exact except `Contains` and `Prefix`, which
/// ignore case (ASCII and Unicode lowercase).
///
/// On an attribute of several strings (tags), a condition on a string holds if it holds for one
/// of them: `Eq` is "has this tag", `In` "has one of these", `Prefix` "has one that starts so",
/// and `Not(Eq)` "doesn't have it".
#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    Eq(String, Value),
    In(String, Vec<Value>),
    /// Integers from `min` to `max`, inclusive; either end open. (On a column of floats, the
    /// floats between them.)
    Range(String, Option<i64>, Option<i64>),
    /// Numbers from `min` to `max`, inclusive; either end open. (On a column of integers, the
    /// integers between them.)
    FloatRange(String, Option<f64>, Option<f64>),
    Prefix(String, String),
    Contains(String, String),
    /// The whole string matches a pattern: `*` is any run of characters (none too), `?` any one
    /// character, `\` takes the next character as itself. With `true`, whatever the case.
    Glob(String, String, bool),
    /// The attribute has a value.
    Exists(String),
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

/// Attribute values of one document, by column number.
pub trait Row {
    /// An integer column's value; a float column's as `float_code` has it.
    fn int(&self, column: usize) -> Option<i64>;
    fn str(&self, column: usize) -> Option<&str>;
    /// Whether `is` holds for one of a tags column's strings.
    fn some_tag(&self, column: usize, is: &mut dyn FnMut(&str) -> bool) -> bool;
}

/// A filter with attribute names resolved to columns (None: the source has no such column, so
/// no document has a value for it).
#[derive(Clone, Debug, PartialEq)]
pub enum Bound {
    Eq(Option<(usize, Kind)>, Value),
    In(Option<(usize, Kind)>, Vec<Value>),
    Range(Option<usize>, Option<i64>, Option<i64>),
    Prefix(Option<usize>, String),
    Contains(Option<usize>, String),
    /// The pattern (lowercased, when case is ignored), and whether case is ignored.
    Glob(Option<usize>, String, bool),
    Exists(Option<(usize, Kind)>),
    And(Vec<Bound>),
    Or(Vec<Bound>),
    Not(Box<Bound>),
}

impl Filter {
    /// Matches nothing (an empty `Or`): what a range past the integers' ends is.
    pub fn nothing() -> Filter {
        Filter::Or(Vec::new())
    }

    /// Integers above `x` (none, past the largest).
    pub fn above(name: String, x: i64) -> Filter {
        x.checked_add(1).map_or_else(Filter::nothing, |low| Filter::Range(name, Some(low), None))
    }

    /// Integers below `x` (none, past the smallest).
    pub fn below(name: String, x: i64) -> Filter {
        x.checked_sub(1).map_or_else(Filter::nothing, |high| Filter::Range(name, None, Some(high)))
    }

    /// Resolves names against a source's columns (name, kind).
    ///
    /// A column of floats is bound as the column of integers it is stored as (`float_code`),
    /// with the condition's numbers turned into those: nothing past here knows of floats. A
    /// number that a column's kind can't hold (a fraction asked of integers, NaN) is a value no
    /// document has.
    pub fn bind(&self, columns: &[(String, Kind)]) -> Bound {
        let find = |name: &str| columns.iter().position(|(n, _)| n == name).map(|i| (i, columns[i].1));
        let typed = |name: &str, kind: Kind| find(name).filter(|&(_, k)| k == kind).map(|(i, _)| i);
        // A column of strings, one a document or several.
        let text = |name: &str| find(name).filter(|&(_, k)| matches!(k, Kind::Str | Kind::Tags)).map(|(i, _)| i);
        // The column as what it is stored as, and a value as what that holds (none: no document's).
        let stored = |column: Option<(usize, Kind)>| column.map(|(i, k)| (i, if k == Kind::Float { Kind::Int } else { k }));
        let held = |column: Option<(usize, Kind)>, v: &Value| -> Option<Value> {
            match (column.map(|c| c.1), v) {
                // Several strings aren't what an attribute is compared with: one of them is.
                (_, Value::Tags(_)) => None,
                (Some(Kind::Float), Value::Float(x)) => Some(float_code(*x)).filter(|c| *c != NO_INT).map(Value::Int),
                (Some(Kind::Float), Value::Int(x)) => Some(Value::Int(float_code(*x as f64))),
                (Some(Kind::Float), Value::Str(_)) => None,
                // An integer written as a float: itself, if it is one.
                (Some(Kind::Int), Value::Float(x)) => (x.fract() == 0.0 && *x >= -9.2e18 && *x <= 9.2e18).then(|| Value::Int(*x as i64)),
                _ => Some(v.clone()),
            }
        };
        match self {
            Filter::Eq(n, v) => match held(find(n), v) {
                Some(v) => Bound::Eq(stored(find(n)), v),
                None => Bound::Or(Vec::new()),
            },
            Filter::In(n, vs) => Bound::In(stored(find(n)), vs.iter().filter_map(|v| held(find(n), v)).collect()),
            Filter::Range(n, lo, hi) => match find(n) {
                Some((i, Kind::Float)) => Bound::Range(Some(i), lo.map(|x| float_code(x as f64)), hi.map(|x| float_code(x as f64))),
                _ => Bound::Range(typed(n, Kind::Int), *lo, *hi),
            },
            Filter::FloatRange(n, lo, hi) => {
                if lo.is_some_and(f64::is_nan) || hi.is_some_and(f64::is_nan) {
                    return Bound::Or(Vec::new());
                }
                match find(n) {
                    Some((i, Kind::Float)) => Bound::Range(Some(i), lo.map(float_code), hi.map(float_code)),
                    // The integers between them: from the first at or above the lower end to
                    // the last at or below the upper (as casts round: towards the ends of i64).
                    Some((i, Kind::Int)) => {
                        let (lo, hi) = (lo.map(|x| x.ceil()), hi.map(|x| x.floor()));
                        if lo.is_some_and(|x| x > 9.2e18) || hi.is_some_and(|x| x < -9.2e18) {
                            return Bound::Or(Vec::new());
                        }
                        Bound::Range(Some(i), lo.map(|x| x as i64), hi.map(|x| x as i64))
                    }
                    _ => Bound::Range(None, None, None),
                }
            }
            Filter::Prefix(n, p) => Bound::Prefix(text(n), p.to_lowercase()),
            Filter::Contains(n, p) => Bound::Contains(text(n), p.to_lowercase()),
            Filter::Glob(n, p, without_case) => Bound::Glob(text(n), if *without_case { p.to_lowercase() } else { p.clone() }, *without_case),
            Filter::Exists(n) => Bound::Exists(stored(find(n))),
            Filter::And(fs) => Bound::And(fs.iter().map(|f| f.bind(columns)).collect()),
            Filter::Or(fs) => Bound::Or(fs.iter().map(|f| f.bind(columns)).collect()),
            Filter::Not(f) => Bound::Not(Box::new(f.bind(columns))),
        }
    }
}

fn equals(row: &dyn Row, column: Option<(usize, Kind)>, v: &Value) -> bool {
    match (column, v) {
        (Some((c, Kind::Int)), Value::Int(x)) => row.int(c) == Some(*x),
        (Some((c, Kind::Str)), Value::Str(x)) => row.str(c) == Some(x.as_str()),
        (Some((c, Kind::Tags)), Value::Str(x)) => row.some_tag(c, &mut |s| s == x),
        _ => false,
    }
}

/// Whether the whole of `text` matches `pattern`: `*` is any run of characters (none too), `?`
/// any one character, `\` takes the next character as itself. On bytes, a character at a time
/// where it matters (`?`, and where a `*` tries again), so UTF-8 stays whole.
pub fn glob(text: &str, pattern: &str) -> bool {
    let (t, p) = (text.as_bytes(), pattern.as_bytes());
    // The bytes of the character that starts with `b`.
    let wide = |b: u8| match b {
        0..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    };
    let (mut ti, mut pi) = (0, 0);
    // The last `*` and where in the text it is trying from.
    let mut star: Option<(usize, usize)> = None;
    while ti < t.len() {
        let literal = match p.get(pi) {
            Some(b'*') => {
                star = Some((pi, ti));
                pi += 1;
                continue;
            }
            Some(b'?') => {
                ti += wide(t[ti]);
                pi += 1;
                continue;
            }
            Some(b'\\') if pi + 1 < p.len() => Some((p[pi + 1], 2)),
            Some(&b) => Some((b, 1)),
            None => None,
        };
        match literal {
            Some((b, n)) if b == t[ti] => {
                ti += 1;
                pi += n;
            }
            // Not this way: the `*` before takes one character more.
            _ => match &mut star {
                Some((at, from)) => {
                    *from += wide(t[*from]);
                    (pi, ti) = (*at + 1, *from);
                }
                None => return false,
            },
        }
    }
    p[pi.min(p.len())..].iter().all(|&b| b == b'*')
}

/// The start of `pattern` that is the same in every string it matches: up to its first `*`,
/// `?` or `\`.
pub(crate) fn glob_prefix(pattern: &str) -> &str {
    &pattern[..pattern.find(['*', '?', '\\']).unwrap_or(pattern.len())]
}

/// Lowercase only when needed (ASCII names are the common case).
pub(crate) fn lower(s: &str) -> std::borrow::Cow<'_, str> {
    if s.bytes().any(|b| b.is_ascii_uppercase() || b >= 0x80) { std::borrow::Cow::Owned(s.to_lowercase()) } else { std::borrow::Cow::Borrowed(s) }
}

impl Bound {
    pub fn matches(&self, row: &dyn Row) -> bool {
        match self {
            Bound::Eq(c, v) => equals(row, *c, v),
            Bound::In(c, vs) => vs.iter().any(|v| equals(row, *c, v)),
            Bound::Range(c, lo, hi) => c.and_then(|c| row.int(c)).is_some_and(|x| lo.is_none_or(|lo| x >= lo) && hi.is_none_or(|hi| x <= hi)),
            // (A column is of one string a document or of several: one of the two has any.)
            Bound::Prefix(c, p) => c.is_some_and(|c| row.str(c).is_some_and(|s| lower(s).starts_with(p.as_str())) || row.some_tag(c, &mut |s| lower(s).starts_with(p.as_str()))),
            Bound::Contains(c, p) => c.is_some_and(|c| row.str(c).is_some_and(|s| lower(s).contains(p.as_str())) || row.some_tag(c, &mut |s| lower(s).contains(p.as_str()))),
            Bound::Glob(c, p, without_case) => {
                let is = |s: &str| if *without_case { glob(&lower(s), p) } else { glob(s, p) };
                c.is_some_and(|c| row.str(c).is_some_and(is) || row.some_tag(c, &mut |s| is(s)))
            }
            Bound::Exists(c) => match c {
                Some((c, Kind::Int | Kind::Float)) => row.int(*c).is_some(),
                Some((c, Kind::Str)) => row.str(*c).is_some(),
                Some((c, Kind::Tags)) => row.some_tag(*c, &mut |_| true),
                None => false,
            },
            Bound::And(bs) => bs.iter().all(|b| b.matches(row)),
            Bound::Or(bs) => bs.iter().any(|b| b.matches(row)),
            Bound::Not(b) => !b.matches(row),
        }
    }
}

/// An attribute column being built (in memory, or while writing a segment).
#[derive(Clone, Debug)]
pub enum Column {
    Int(Vec<i64>),
    Str(Vec<Option<String>>),
    /// Floats, each as `float_code` has it (`NO_INT`: none).
    Float(Vec<i64>),
    Tags(Tags),
}

/// A column of several strings a document, being built: each distinct string kept once and
/// numbered as it is first seen, and a document's as their numbers.
#[derive(Clone, Debug, Default)]
pub struct Tags {
    /// Where each document's numbers end in `ids` (they start where the one before's end).
    ends: Vec<usize>,
    ids: Vec<u32>,
    names: Vec<std::sync::Arc<str>>,
    numbers: std::collections::HashMap<std::sync::Arc<str>, u32>,
}

impl Tags {
    fn len(&self) -> usize {
        self.ends.len()
    }

    fn ids_of(&self, doc: usize) -> &[u32] {
        let start = if doc == 0 { 0 } else { self.ends[doc - 1] };
        &self.ids[start..self.ends[doc]]
    }

    /// A document's strings, in the order they were given.
    pub fn of(&self, doc: usize) -> impl Iterator<Item = &str> {
        let ids = if doc < self.len() { self.ids_of(doc) } else { &[] };
        ids.iter().map(|&n| &*self.names[n as usize])
    }

    /// Another document, with these strings (each once; empty ones are none).
    fn push<'a>(&mut self, tags: impl Iterator<Item = &'a str>) {
        let from = self.ids.len();
        for t in tags.filter(|t| !t.is_empty()) {
            let n = match self.numbers.get(t) {
                Some(&n) => n,
                None => {
                    let name: std::sync::Arc<str> = t.into();
                    self.names.push(name.clone());
                    self.numbers.insert(name, self.names.len() as u32 - 1);
                    self.names.len() as u32 - 1
                }
            };
            if !self.ids[from..].contains(&n) {
                self.ids.push(n);
            }
        }
        self.ends.push(self.ids.len());
    }
}

impl Column {
    pub fn new(kind: Kind) -> Column {
        match kind {
            Kind::Int => Column::Int(Vec::new()),
            Kind::Str => Column::Str(Vec::new()),
            Kind::Float => Column::Float(Vec::new()),
            Kind::Tags => Column::Tags(Tags::default()),
        }
    }

    pub fn kind(&self) -> Kind {
        match self {
            Column::Int(_) => Kind::Int,
            Column::Str(_) => Kind::Str,
            Column::Float(_) => Kind::Float,
            Column::Tags(_) => Kind::Tags,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Column::Int(v) | Column::Float(v) => v.len(),
            Column::Str(v) => v.len(),
            Column::Tags(t) => t.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Pads with "no value" up to `len` documents.
    pub fn pad(&mut self, len: usize) {
        match self {
            Column::Int(v) | Column::Float(v) => v.resize(len, NO_INT),
            Column::Str(v) => v.resize(len, None),
            Column::Tags(t) => {
                if t.len() < len {
                    t.ends.resize(len, t.ids.len());
                }
            }
        }
    }

    /// Appends another column's values (of this kind; another kind's are none).
    pub fn extend_from(&mut self, other: &Column) {
        match (self, other) {
            (Column::Int(c), Column::Int(o)) | (Column::Float(c), Column::Float(o)) => c.extend_from_slice(o),
            (Column::Str(c), Column::Str(o)) => c.extend_from_slice(o),
            (Column::Tags(c), Column::Tags(o)) => {
                for d in 0..o.len() {
                    c.push(o.of(d));
                }
            }
            (c, o) => {
                let len = c.len() + o.len();
                c.pad(len);
            }
        }
    }

    /// Appends a value of this column's kind (or none, for another kind).
    pub fn push(&mut self, v: Option<&Value>) {
        match (self, v) {
            (Column::Int(c), Some(Value::Int(x))) => c.push(*x),
            (Column::Float(c), Some(Value::Float(x))) => c.push(float_code(*x)),
            (Column::Float(c), Some(Value::Int(x))) => c.push(float_code(*x as f64)),
            (Column::Int(c) | Column::Float(c), _) => c.push(NO_INT),
            (Column::Str(c), Some(Value::Str(x))) => c.push(Some(x.clone())),
            (Column::Str(c), _) => c.push(None),
            (Column::Tags(c), Some(Value::Tags(x))) => c.push(x.iter().map(String::as_str)),
            // One string is a set of one.
            (Column::Tags(c), Some(Value::Str(x))) => c.push(std::iter::once(x.as_str())),
            (Column::Tags(c), _) => c.push(std::iter::empty()),
        }
    }

    pub fn get(&self, doc: usize) -> Option<Value> {
        match self {
            Column::Int(v) => v.get(doc).filter(|&&x| x != NO_INT).map(|&x| Value::Int(x)),
            Column::Float(v) => v.get(doc).filter(|&&x| x != NO_INT).map(|&x| Value::Float(code_float(x))),
            Column::Str(v) => v.get(doc).and_then(|s| s.clone()).map(Value::Str),
            Column::Tags(t) => {
                let mut tags: Vec<String> = t.of(doc).map(str::to_string).collect();
                tags.sort_unstable();
                (!tags.is_empty()).then_some(Value::Tags(tags))
            }
        }
    }

    /// The bytes a segment stores (see `Layout`).
    pub fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Column::Int(v) | Column::Float(v) => {
                let present = v.iter().filter(|&&x| x != NO_INT).count();
                let (min, max) = v.iter().filter(|&&x| x != NO_INT).fold((i64::MAX, i64::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
                // What every value's distance from the smallest shares: timestamps in whole
                // seconds or days, prices in cents.
                let gcd = v.iter().filter(|&&x| x != NO_INT).fold(0u64, |g, &x| greatest_common(g, x.wrapping_sub(min) as u64)).max(1);
                let code = |x: i64| if x == NO_INT { 0 } else { (x.wrapping_sub(min) as u64) / gcd + 1 };
                let top = if present == 0 { 0 } else { code(max) };
                let width = width_for(top, present == v.len());
                out.push(width as u8);
                out.extend_from_slice(&[0; 7]);
                out.extend_from_slice(&(if present == 0 { 0 } else { min }).to_le_bytes());
                out.extend_from_slice(&gcd.to_le_bytes());
                out.extend_from_slice(&top.to_le_bytes());
                out.extend_from_slice(&(present as u64).to_le_bytes());
                put_codes(out, width, v.len(), |d| code(v[d]));
            }
            Column::Str(v) => {
                let mut distinct: Vec<&str> = v.iter().flatten().map(String::as_str).filter(|s| !s.is_empty()).collect();
                distinct.sort_unstable();
                distinct.dedup();
                let code = |s: &Option<String>| s.as_deref().filter(|s| !s.is_empty()).map_or(0, |s| distinct.binary_search(&s).unwrap() as u64 + 1);
                let codes: Vec<u64> = v.iter().map(code).collect();
                let mut counts = vec![0u32; distinct.len()];
                for &c in codes.iter().filter(|&&c| c != 0) {
                    counts[c as usize - 1] += 1;
                }
                let present = codes.iter().filter(|&&c| c != 0).count();
                let top = distinct.len() as u64;
                let width = width_for(top, present == v.len());
                out.push(width as u8);
                out.push(distinct.iter().any(|s| !s.is_ascii() && s.contains(INTO_ASCII)) as u8);
                out.extend_from_slice(&[0; 6]);
                out.extend_from_slice(&top.to_le_bytes());
                out.extend_from_slice(&top.to_le_bytes());
                out.extend_from_slice(&(present as u64).to_le_bytes());
                // The dictionary: where each string ends, how many documents have it, the strings.
                let mut at = 0u64;
                out.extend_from_slice(&at.to_le_bytes());
                for s in &distinct {
                    at += s.len() as u64;
                    out.extend_from_slice(&at.to_le_bytes());
                }
                for c in &counts {
                    out.extend_from_slice(&c.to_le_bytes());
                }
                for s in &distinct {
                    out.extend_from_slice(s.as_bytes());
                }
                put_codes(out, width, v.len(), |d| codes[d]);
            }
            Column::Tags(t) => {
                let docs = t.len();
                // The distinct strings in byte order, and each one's code there.
                let mut order: Vec<u32> = (0..t.names.len() as u32).collect();
                order.sort_unstable_by(|a, b| t.names[*a as usize].cmp(&t.names[*b as usize]));
                let mut code = vec![0u64; t.names.len()];
                for (i, &n) in order.iter().enumerate() {
                    code[n as usize] = i as u64 + 1;
                }
                let top = t.names.len() as u64;
                // Each document's codes in order, where they start, and how many documents have each.
                let mut values: Vec<u64> = Vec::with_capacity(t.ids.len());
                let mut starts: Vec<u64> = Vec::with_capacity(docs + 1);
                let mut counts = vec![0u32; t.names.len()];
                let mut present = 0u64;
                for d in 0..docs {
                    let from = values.len();
                    starts.push(from as u64);
                    values.extend(t.ids_of(d).iter().map(|&n| code[n as usize]));
                    values[from..].sort_unstable();
                    for &c in &values[from..] {
                        counts[c as usize - 1] += 1;
                    }
                    present += (values.len() > from) as u64;
                }
                starts.push(values.len() as u64);
                let width = width_for(top, false).max(1);
                let each = if values.len() <= u32::MAX as usize { 4 } else { 8 };
                out.push(width as u8);
                out.push(t.names.iter().any(|s| !s.is_ascii() && s.contains(INTO_ASCII)) as u8);
                out.push(each as u8);
                out.extend_from_slice(&[0; 5]);
                out.extend_from_slice(&top.to_le_bytes());
                out.extend_from_slice(&(values.len() as u64).to_le_bytes());
                out.extend_from_slice(&present.to_le_bytes());
                // The dictionary, as a column of one string a document has it.
                let mut at = 0u64;
                out.extend_from_slice(&at.to_le_bytes());
                for &n in &order {
                    at += t.names[n as usize].len() as u64;
                    out.extend_from_slice(&at.to_le_bytes());
                }
                for c in &counts {
                    out.extend_from_slice(&c.to_le_bytes());
                }
                for &n in &order {
                    out.extend_from_slice(t.names[n as usize].as_bytes());
                }
                for s in &starts {
                    out.extend_from_slice(&s.to_le_bytes()[..each]);
                }
                for v in &values {
                    out.extend_from_slice(&v.to_le_bytes()[..width]);
                }
                // Per string, the documents that have it: where each one's begin, then theirs.
                let mut begin = vec![0usize; t.names.len() + 1];
                for (i, &c) in counts.iter().enumerate() {
                    begin[i + 1] = begin[i] + c as usize;
                }
                let mut next = begin.clone();
                let mut having = vec![0u32; values.len()];
                for d in 0..docs {
                    for &c in &values[starts[d] as usize..starts[d + 1] as usize] {
                        having[next[c as usize - 1]] = d as u32;
                        next[c as usize - 1] += 1;
                    }
                }
                let mut at = 0u64;
                out.extend_from_slice(&at.to_le_bytes());
                for &c in &counts {
                    at += list_bytes(c as u64, docs) as u64;
                    out.extend_from_slice(&at.to_le_bytes());
                }
                for (i, &c) in counts.iter().enumerate() {
                    let list = &having[begin[i]..begin[i + 1]];
                    if as_bits(c as u64, docs) {
                        let mut words = vec![0u64; docs.div_ceil(64)];
                        for &d in list {
                            words[d as usize / 64] |= 1 << (d % 64);
                        }
                        for w in &words {
                            out.extend_from_slice(&w.to_le_bytes());
                        }
                    } else {
                        for d in list {
                            out.extend_from_slice(&d.to_le_bytes());
                        }
                    }
                }
            }
        }
    }
}

/// Whether the documents that have a tag are kept as a bit per document of the segment: when
/// their numbers (four bytes each) would take as much.
fn as_bits(count: u64, docs: usize) -> bool {
    count * 4 >= docs.div_ceil(64) as u64 * 8
}

/// The bytes the documents that have a tag take.
fn list_bytes(count: u64, docs: usize) -> usize {
    if as_bits(count, docs) { docs.div_ceil(64) * 8 } else { count as usize * 4 }
}

/// The characters outside ASCII whose lowercase is inside it, or starts there: İ (i, then a dot
/// above) and the Kelvin sign (k). A search for ASCII, without case, finds every other string by
/// its bytes; a string with one of these is lowercased first, as `Bound::matches` does.
pub(crate) const INTO_ASCII: [char; 2] = ['\u{130}', '\u{212A}'];

fn greatest_common(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The bytes a code up to `top` needs: 0 when every document has the same one, else 1, 2, 4 or 8.
fn width_for(top: u64, all_present: bool) -> usize {
    match top {
        0 => 0,
        1 if all_present => 0,
        t if t <= u8::MAX as u64 => 1,
        t if t <= u16::MAX as u64 => 2,
        t if t <= u32::MAX as u64 => 4,
        _ => 8,
    }
}

/// Documents per pair of smallest and largest code (a zone).
pub const ZONE: usize = 1024;

/// The codes of `docs` documents at `width` bytes each, then each zone's smallest and largest.
fn put_codes(out: &mut Vec<u8>, width: usize, docs: usize, code: impl Fn(usize) -> u64) {
    for d in 0..docs {
        out.extend_from_slice(&code(d).to_le_bytes()[..width]);
    }
    if width == 0 {
        return;
    }
    for zone in (0..docs).step_by(ZONE) {
        let (lo, hi) = (zone..(zone + ZONE).min(docs)).map(&code).fold((u64::MAX, 0), |(lo, hi), c| (lo.min(c), hi.max(c)));
        out.extend_from_slice(&lo.to_le_bytes());
        out.extend_from_slice(&hi.to_le_bytes());
    }
}

/// Where a column's parts are in a segment's bytes (version 8 on), and what its codes mean.
///
/// An integer column: a 40-byte header (the width of a code in bytes, the smallest value, the
/// step, the largest code, how many documents have a value), a code per document (0: none, else
/// `(value − smallest) / step + 1`), and per `ZONE` documents the smallest and largest code.
/// A string column: a 32-byte header (the width, whether a string has one of `INTO_ASCII`, the
/// number of distinct strings twice, how many documents have one), the dictionary (the distinct strings in byte order: where each ends, how
/// many documents have it, the bytes), a code per document (0: none, else its string's place in
/// the dictionary plus one), and the zones. A width of 0 means no codes are stored: every
/// document has the one value there is, or none has any.
/// A tags column: a 32-byte header (the width of a code, whether a string has one of
/// `INTO_ASCII`, the width of a start, the number of distinct strings, of strings in all
/// documents together, of documents with any), the dictionary as a string column's (a string's
/// count: the documents that have it), where each document's codes start, the codes (a
/// document's in order), then per string the documents that have it: where each one's bytes
/// begin, and they, as numbers in order or (`as_bits`) a bit per document of the segment.
/// No zones: a condition is answered from the documents kept per string, in time by how many
/// it matches, not by how many there are.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub kind: Kind,
    pub docs: usize,
    pub width: usize,
    pub min: i64,
    pub step: u64,
    /// The largest code (0: no document has a value).
    pub top: u64,
    pub present: u64,
    /// A string has one of `INTO_ASCII`.
    pub into_ascii: bool,
    /// Tags: the bytes of a start, where the starts are, where the lists' beginnings are, and
    /// where the lists are.
    each: usize,
    starts: usize,
    lists: usize,
    having: usize,
    codes: usize,
    zones: usize,
    ends: usize,
    counts: usize,
    arena: usize,
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

impl Layout {
    /// The layout of the column of `kind` whose bytes are `bytes[at..at + len]`, for `docs`
    /// documents; none if they aren't one.
    pub fn parse(kind: Kind, bytes: &[u8], at: usize, len: usize, docs: usize) -> Option<Layout> {
        // Floats are stored as the integers `float_code` makes of them.
        let kind = if kind == Kind::Float { Kind::Int } else { kind };
        if kind == Kind::Tags {
            return Layout::parse_tags(bytes, at, len, docs);
        }
        let head = if kind == Kind::Int { 40 } else { 32 };
        if len < head || at.checked_add(len)? > bytes.len() {
            return None;
        }
        let width = bytes[at] as usize;
        if ![0, 1, 2, 4, 8].contains(&width) {
            return None;
        }
        let zones = if width == 0 { 0 } else { docs.div_ceil(ZONE) * 16 };
        let mut l = Layout { kind, docs, width, min: 0, step: 1, top: 0, present: 0, into_ascii: kind == Kind::Str && bytes[at + 1] != 0, each: 0, starts: 0, lists: 0, having: 0, codes: 0, zones: 0, ends: 0, counts: 0, arena: 0 };
        let dictionary = match kind {
            Kind::Int | Kind::Float => {
                (l.min, l.step, l.top, l.present) = (u64_at(bytes, at + 8) as i64, u64_at(bytes, at + 16).max(1), u64_at(bytes, at + 24), u64_at(bytes, at + 32));
                0
            }
            Kind::Tags => return None,
            Kind::Str => {
                (l.top, l.present) = (u64_at(bytes, at + 8), u64_at(bytes, at + 24));
                let distinct = usize::try_from(l.top).ok()?;
                l.ends = at + head;
                l.counts = l.ends + distinct.checked_add(1)?.checked_mul(8)?;
                l.arena = l.counts.checked_add(distinct.checked_mul(4)?)?;
                if l.arena > at + len {
                    return None;
                }
                let text = usize::try_from(u64_at(bytes, l.ends + distinct * 8)).ok()?;
                (distinct + 1) * 8 + distinct * 4 + text
            }
        };
        l.codes = at + head + dictionary;
        l.zones = l.codes.checked_add(docs.checked_mul(width)?)?;
        (l.zones.checked_add(zones)? == at + len).then_some(l)
    }

    fn parse_tags(bytes: &[u8], at: usize, len: usize, docs: usize) -> Option<Layout> {
        let end = at.checked_add(len).filter(|&e| e <= bytes.len())?;
        if len < 32 {
            return None;
        }
        let (width, each) = (bytes[at] as usize, bytes[at + 2] as usize);
        if ![1, 2, 4, 8].contains(&width) || ![4, 8].contains(&each) {
            return None;
        }
        let (top, values, present) = (u64_at(bytes, at + 8), u64_at(bytes, at + 16), u64_at(bytes, at + 24));
        let (distinct, values) = (usize::try_from(top).ok()?, usize::try_from(values).ok()?);
        let mut l = Layout { kind: Kind::Tags, docs, width, min: 0, step: 1, top, present, into_ascii: bytes[at + 1] != 0, each, starts: 0, lists: 0, having: 0, codes: 0, zones: 0, ends: 0, counts: 0, arena: 0 };
        l.ends = at + 32;
        l.counts = l.ends.checked_add(distinct.checked_add(1)?.checked_mul(8)?)?;
        l.arena = l.counts.checked_add(distinct.checked_mul(4)?)?;
        if l.arena > end {
            return None;
        }
        let text = usize::try_from(u64_at(bytes, l.ends + distinct * 8)).ok()?;
        l.starts = l.arena.checked_add(text)?;
        l.codes = l.starts.checked_add(docs.checked_add(1)?.checked_mul(each)?)?;
        l.lists = l.codes.checked_add(values.checked_mul(width)?)?;
        l.having = l.lists.checked_add(distinct.checked_add(1)?.checked_mul(8)?)?;
        if l.having > end {
            return None;
        }
        // The codes end where the last document's do, and the lists where the segment's column does.
        let lists = usize::try_from(u64_at(bytes, l.lists + distinct * 8)).ok()?;
        (l.start(bytes, docs) == values && l.having.checked_add(lists)? == end).then_some(l)
    }

    /// Where a document's codes start among a tags column's (for `docs`: where they all end).
    #[inline]
    fn start(&self, bytes: &[u8], doc: usize) -> usize {
        let at = self.starts + doc * self.each;
        match self.each {
            4 => u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize,
            _ => u64_at(bytes, at) as usize,
        }
    }

    /// The codes of a document's tags, in order.
    pub fn tags<'a>(&self, bytes: &'a [u8], doc: usize) -> impl Iterator<Item = u64> + 'a {
        let (from, to) = if self.kind == Kind::Tags && doc < self.docs { (self.start(bytes, doc), self.start(bytes, doc + 1)) } else { (0, 0) };
        // (Never past the column's codes, whatever a damaged file says.)
        let to = to.min((self.lists - self.codes) / self.width);
        let from = from.min(to);
        let (codes, width) = (self.codes, self.width);
        (from..to).map(move |i| {
            let at = codes + i * width;
            match width {
                1 => bytes[at] as u64,
                2 => u16::from_le_bytes([bytes[at], bytes[at + 1]]) as u64,
                4 => u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as u64,
                _ => u64_at(bytes, at),
            }
        })
    }

    /// A bit per document: whether it has a tag whose code is in one of `runs` (first and last
    /// code of each, 1 and up). Read from the documents kept per tag, so in time by how many
    /// documents have them.
    pub fn having(&self, bytes: &[u8], runs: &[(u64, u64)]) -> Vec<u64> {
        let mut words = vec![0u64; self.docs.div_ceil(64)];
        if self.kind != Kind::Tags {
            return words;
        }
        for &(lo, hi) in runs {
            for code in lo.max(1)..=hi.min(self.top) {
                let i = code as usize - 1;
                let (a, b) = (u64_at(bytes, self.lists + i * 8) as usize, u64_at(bytes, self.lists + i * 8 + 8) as usize);
                let Some(list) = bytes.get(self.having + a..self.having + b) else { continue };
                if as_bits(self.count(bytes, code), self.docs) {
                    for (w, b) in words.iter_mut().zip(list.chunks_exact(8)) {
                        *w |= u64::from_le_bytes(b.try_into().unwrap());
                    }
                } else {
                    for d in list.chunks_exact(4) {
                        let d = u32::from_le_bytes(d.try_into().unwrap()) as usize;
                        if d < self.docs {
                            words[d / 64] |= 1 << (d % 64);
                        }
                    }
                }
            }
        }
        words
    }

    /// A bit per document: whether it has any tag.
    pub fn tagged(&self, bytes: &[u8]) -> Vec<u64> {
        let mut words = vec![0u64; self.docs.div_ceil(64)];
        if self.kind != Kind::Tags || self.present == 0 {
            return words;
        }
        if self.present as usize == self.docs {
            fill(&mut words, 0, self.docs);
            return words;
        }
        let mut start = self.start(bytes, 0);
        for d in 0..self.docs {
            let next = self.start(bytes, d + 1);
            words[d / 64] |= ((next > start) as u64) << (d % 64);
            start = next;
        }
        words
    }

    /// A document's code (0: no value).
    #[inline]
    pub fn code(&self, bytes: &[u8], doc: usize) -> u64 {
        let at = self.codes + doc * self.width;
        match self.width {
            0 => (self.present as usize == self.docs && self.top > 0) as u64,
            1 => bytes[at] as u64,
            2 => u16::from_le_bytes([bytes[at], bytes[at + 1]]) as u64,
            4 => u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as u64,
            _ => u64_at(bytes, at),
        }
    }

    /// A document's integer.
    #[inline]
    pub fn int(&self, bytes: &[u8], doc: usize) -> Option<i64> {
        match self.code(bytes, doc) {
            0 => None,
            c => Some(self.min.wrapping_add((c - 1).wrapping_mul(self.step) as i64)),
        }
    }

    /// The integer a code (1 and up) stands for.
    #[inline]
    pub fn value(&self, code: u64) -> i64 {
        self.min.wrapping_add((code - 1).wrapping_mul(self.step) as i64)
    }

    /// The smallest and largest code among zone `z`'s documents (`ZONE` of them, the last fewer).
    pub fn zone(&self, bytes: &[u8], z: usize) -> (u64, u64) {
        (u64_at(bytes, self.zones + z * 16), u64_at(bytes, self.zones + z * 16 + 8))
    }

    /// The string with a code (1 and up).
    #[inline]
    pub fn string<'a>(&self, bytes: &'a [u8], code: u64) -> Option<&'a str> {
        let i = code.checked_sub(1).filter(|_| code <= self.top)? as usize;
        let (a, b) = (u64_at(bytes, self.ends + i * 8) as usize, u64_at(bytes, self.ends + i * 8 + 8) as usize);
        std::str::from_utf8(bytes.get(self.arena + a..self.arena + b)?).ok()
    }

    /// How many documents have the string with a code.
    pub fn count(&self, bytes: &[u8], code: u64) -> u64 {
        let i = code as usize - 1;
        u32::from_le_bytes(bytes[self.counts + i * 4..self.counts + i * 4 + 4].try_into().unwrap()) as u64
    }

    /// The code of a string, if the column has it.
    pub fn find(&self, bytes: &[u8], s: &str) -> Option<u64> {
        let (mut lo, mut hi) = (1, self.top + 1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.string(bytes, mid)?.cmp(s) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// The codes of the integers in [lo, hi], as a range (none: no integer there).
    pub fn codes_between(&self, lo: i64, hi: i64) -> Option<(u64, u64)> {
        if self.top == 0 || lo > hi {
            return None;
        }
        let max = self.min.wrapping_add((self.top - 1).wrapping_mul(self.step) as i64);
        if hi < self.min || lo > max {
            return None;
        }
        // The first code at or above lo, the last at or below hi.
        let first = if lo <= self.min { 1 } else { (lo.wrapping_sub(self.min) as u64).div_ceil(self.step) + 1 };
        let last = if hi >= max { self.top } else { (hi.wrapping_sub(self.min) as u64) / self.step + 1 };
        (first <= last).then_some((first, last))
    }

    /// The bytes of the string with a code (1 and up); none past the last.
    fn raw<'a>(&self, bytes: &'a [u8], code: u64) -> &'a [u8] {
        let i = code as usize - 1;
        let (a, b) = (u64_at(bytes, self.ends + i * 8) as usize, u64_at(bytes, self.ends + i * 8 + 8) as usize);
        bytes.get(self.arena..).and_then(|arena| arena.get(a..b)).unwrap_or(&[])
    }

    /// Every distinct string with its code, in order.
    pub fn each_string<'a>(&self, bytes: &'a [u8], mut each: impl FnMut(u64, &'a str)) {
        for code in 1..=self.top {
            if let Ok(s) = std::str::from_utf8(self.raw(bytes, code)) {
                each(code, s);
            }
        }
    }

    /// The codes of the strings that start with exactly `prefix`: the first, and one past the
    /// last (the dictionary is in byte order, so they are next to each other).
    pub fn with_prefix(&self, bytes: &[u8], prefix: &str) -> (u64, u64) {
        let first = first_where(1, self.top + 1, |c| self.raw(bytes, c) >= prefix.as_bytes());
        (first, first_where(first, self.top + 1, |c| !self.raw(bytes, c).starts_with(prefix.as_bytes())))
    }

    /// The codes of the strings that start with `prefix` whatever their case, as ranges in order.
    /// `prefix` is ASCII without capitals.
    ///
    /// The dictionary is in byte order, so the strings that start with given bytes are next to
    /// each other: two binary searches a byte. Without case a byte has two forms (three with
    /// `INTO_ASCII`), and each form present is followed separately: as many ranges as the
    /// dictionary has ways of writing the prefix, not one test per string.
    pub fn starting_with(&self, bytes: &[u8], prefix: &str) -> Vec<(u64, u64)> {
        // The first code, one past the last, and the bytes they share.
        let mut live = vec![(1u64, self.top + 1, 0usize)];
        for (i, &b) in prefix.as_bytes().iter().enumerate() {
            let mut forms: Vec<&[u8]> = vec![std::slice::from_ref(&b)];
            let capital = [b.to_ascii_uppercase()];
            if capital[0] != b {
                forms.push(&capital);
            }
            if b == b'k' {
                forms.push("\u{212A}".as_bytes());
            }
            // İ is "i" and a dot above: a prefix that goes on has another letter where the dot is.
            if b == b'i' && i + 1 == prefix.len() {
                forms.push("\u{130}".as_bytes());
            }
            let mut next = Vec::new();
            for &(lo, hi, shared) in &live {
                for form in &forms {
                    let rest = |code: u64| self.raw(bytes, code).get(shared..).unwrap_or(&[]);
                    let first = first_where(lo, hi, |c| rest(c) >= *form);
                    let end = first_where(first, hi, |c| !rest(c).starts_with(form));
                    if first < end {
                        next.push((first, end, shared + form.len()));
                    }
                }
            }
            live = next;
            if live.is_empty() {
                break;
            }
        }
        let mut runs: Vec<(u64, u64)> = live.iter().filter(|r| r.0 < r.1).map(|&(lo, hi, _)| (lo, hi - 1)).collect();
        runs.sort_unstable();
        runs
    }

    /// The codes of the strings `needle` is in, whatever the case of their ASCII letters, in
    /// order. `needle` is ASCII without capitals and not empty. One pass over the dictionary's
    /// bytes (`substring`), not a test per string.
    pub fn containing(&self, bytes: &[u8], needle: &str) -> Vec<u64> {
        let strings = self.top as usize;
        let end = |i: usize| u64_at(bytes, self.ends + i * 8) as usize;
        let Some(arena) = bytes.get(self.arena..).and_then(|b| b.get(..end(strings))) else { return Vec::new() };
        let mut codes = Vec::new();
        // The string an occurrence is in is at or after the last one's, and occurrences in a
        // string already found don't count again.
        let (mut string, mut past) = (0, 0);
        crate::substring::find_without_case(arena, needle.as_bytes(), |at| {
            if at < past {
                return;
            }
            // The first string that ends after `at`.
            string = first_where(string as u64, strings as u64, |i| end(i as usize + 1) > at) as usize;
            // Not the end of one string and the start of the next.
            if string < strings && at + needle.len() <= end(string + 1) {
                codes.push(string as u64 + 1);
                past = end(string + 1);
            }
        });
        codes
    }

    /// A bit per document: whether its code is in [lo, hi] (1 ≤ lo).
    pub fn in_range(&self, bytes: &[u8], lo: u64, hi: u64) -> Vec<u64> {
        self.in_runs(bytes, &[(lo, hi)])
    }

    /// A bit per document: whether its code is in one of `runs` (first and last code of each, 1
    /// and up, in order, apart from each other). Zones whose codes are all inside one run or
    /// outside all aren't read; the rest are compared a word of bits at a time (`codes`).
    pub fn in_runs(&self, bytes: &[u8], runs: &[(u64, u64)]) -> Vec<u64> {
        let mut words = vec![0u64; self.docs.div_ceil(64)];
        debug_assert!(runs.iter().all(|r| 1 <= r.0 && r.0 <= r.1) && runs.windows(2).all(|w| w[0].1 < w[1].0), "{runs:?}");
        if runs.is_empty() {
            return words;
        }
        if self.width == 0 {
            let code = self.code(bytes, 0);
            if runs.iter().any(|&(lo, hi)| (lo..=hi).contains(&code)) {
                fill(&mut words, 0, self.docs);
            }
            return words;
        }
        // Up to `FEW` runs reach into a zone: a comparison per run. More: a bit per code,
        // looked up (made the first time it's needed), if that is no more than a bit per code a
        // document could have eight times over; else the run is searched for.
        use crate::codes::FEW;
        let mut near: Vec<(u64, u64)> = Vec::with_capacity(FEW);
        let mut wanted: Vec<u64> = Vec::new();
        let small = self.top / 64 <= self.docs as u64;
        for (z, first) in (0..self.docs).step_by(ZONE).enumerate() {
            let n = (self.docs - first).min(ZONE);
            let (zlo, zhi) = (u64_at(bytes, self.zones + z * 16), u64_at(bytes, self.zones + z * 16 + 8));
            let from = runs.partition_point(|r| r.1 < zlo);
            let to = from + runs[from..].partition_point(|r| r.0 <= zhi);
            if from == to {
                continue;
            }
            if runs[from].0 <= zlo && runs[from].1 >= zhi {
                fill(&mut words, first, n);
                continue;
            }
            let codes = &bytes[self.codes + first * self.width..self.codes + (first + n) * self.width];
            let out = &mut words[first / 64..(first + n).div_ceil(64)];
            if to - from <= FEW {
                // A run that reaches into the zone starts at or under the zone's largest code,
                // so its first code fits the codes' width.
                near.clear();
                near.extend(runs[from..to].iter().map(|&(lo, hi)| (lo, hi - lo)));
                crate::codes::in_runs(codes, self.width, &near, out);
                continue;
            }
            if small && wanted.is_empty() {
                wanted = vec![0u64; self.top as usize / 64 + 1];
                for &(lo, hi) in runs {
                    for c in lo..=hi.min(self.top) {
                        wanted[c as usize / 64] |= 1 << (c % 64);
                    }
                }
            }
            let reach = &runs[from..to];
            let has = |c: u64| match small {
                true => wanted.get(c as usize / 64).is_some_and(|w| w >> (c % 64) & 1 == 1),
                false => reach.get(reach.partition_point(|r| r.1 < c)).is_some_and(|r| r.0 <= c),
            };
            macro_rules! look_up {
                ($t:ty) => {{
                    const W: usize = std::mem::size_of::<$t>();
                    for (w, chunk) in out.iter_mut().zip(codes.chunks(64 * W)) {
                        *w = chunk.chunks_exact(W).enumerate().fold(0u64, |bits, (i, c)| bits | (has(<$t>::from_le_bytes(c.try_into().unwrap()) as u64) as u64) << i);
                    }
                }};
            }
            match self.width {
                1 => look_up!(u8),
                2 => look_up!(u16),
                4 => look_up!(u32),
                _ => look_up!(u64),
            }
        }
        words
    }
}

/// The first of `lo..hi` for which `is` holds (it holds from there on); `hi` if none.
fn first_where(mut lo: u64, mut hi: u64, is: impl Fn(u64) -> bool) -> u64 {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if is(mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// Codes in order, each once, as runs of codes next to each other: the first and last of each.
pub fn runs_of(codes: &[u64]) -> Vec<(u64, u64)> {
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for &c in codes {
        match runs.last_mut() {
            Some(r) if r.1 + 1 == c => r.1 = c,
            _ => runs.push((c, c)),
        }
    }
    runs
}

/// Sets `n` bits of `words` from bit `first`.
fn fill(words: &mut [u64], first: usize, n: usize) {
    let (mut d, end) = (first, first + n);
    while d < end {
        if d % 64 == 0 && d + 64 <= end {
            words[d / 64] = u64::MAX;
            d += 64;
        } else {
            words[d / 64] |= 1 << (d % 64);
            d += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct R(Vec<Column>);
    impl Row for R {
        fn int(&self, c: usize) -> Option<i64> {
            match &self.0[c] {
                Column::Int(v) | Column::Float(v) => v.first().copied().filter(|&x| x != NO_INT),
                _ => None,
            }
        }
        fn str(&self, c: usize) -> Option<&str> {
            match &self.0[c] {
                Column::Str(v) => v.first().and_then(|s| s.as_deref()),
                _ => None,
            }
        }
        fn some_tag(&self, c: usize, is: &mut dyn FnMut(&str) -> bool) -> bool {
            match &self.0[c] {
                Column::Tags(t) => t.of(0).any(is),
                _ => false,
            }
        }
    }

    #[test]
    fn patterns_match_whole_strings() {
        for (text, pattern, is) in [
            ("report.pdf", "*.pdf", true),
            ("report.pdf", "*.PDF", false),
            ("report.pdf", "report.pdf", true),
            ("report.pdf", "report", false),
            ("report.pdf", "r*t.p?f", true),
            ("report.pdf", "*", true),
            ("", "*", true),
            ("", "", true),
            ("", "?", false),
            ("a", "", false),
            ("abc", "a*b*c*", true),
            ("abc", "*a*b*c*d", false),
            ("aaab", "*ab", true),
            ("aaab", "a*a*a*a*b", false),
            ("mississippi", "m*iss*ip?i", true),
            ("mississippi", "m*iss*ip?", false),
            // A character is a character, however many bytes.
            ("über", "?ber", true),
            ("über", "??ber", false),
            ("日本語", "??語", true),
            ("日本語", "*本*", true),
            ("naïve", "na?ve", true),
            // A backslash takes the next character as itself.
            ("a*b", "a\\*b", true),
            ("axb", "a\\*b", false),
            ("a?b", "a\\?b", true),
            ("a\\b", "a\\\\b", true),
            ("ab\\", "ab\\", true),
        ] {
            assert_eq!(glob(text, pattern), is, "{text:?} against {pattern:?}");
        }
        assert_eq!((glob_prefix("/home/u*"), glob_prefix("abc"), glob_prefix("?x"), glob_prefix("a\\*")), ("/home/u", "abc", "", "a"));
    }

    #[test]
    fn floats_are_in_order_as_their_codes() {
        let floats = [f64::NEG_INFINITY, -1e300, -2.5, -1.0, -5e-324, 0.0, 5e-324, 0.1, 1.0, 1.0000000000000002, 2.5, 1e300, f64::INFINITY];
        let codes: Vec<i64> = floats.iter().map(|&x| float_code(x)).collect();
        assert!(codes.windows(2).all(|w| w[0] < w[1]), "{codes:?}");
        for (&x, &c) in floats.iter().zip(&codes) {
            assert_eq!(code_float(c).to_bits(), x.to_bits(), "{x}");
            assert_ne!(c, NO_INT);
        }
        // −0.0 is 0.0 (they are equal), and NaN is no value.
        assert_eq!(float_code(-0.0), float_code(0.0));
        assert_eq!(code_float(float_code(-0.0)).to_bits(), 0.0f64.to_bits());
        assert_eq!(float_code(f64::NAN), NO_INT);
    }

    #[test]
    fn conditions_on_floats_are_bound_to_their_codes() {
        let cols = vec![("score".to_string(), Kind::Float), ("n".to_string(), Kind::Int)];
        let row = |score: f64, n: i64| R(vec![Column::Float(vec![float_code(score)]), Column::Int(vec![n])]);
        let t = |f: Filter, score: f64, n: i64| f.bind(&cols).matches(&row(score, n));
        let score = || "score".to_string();
        // Floats, and integers asked of a float column.
        assert!(t(Filter::FloatRange(score(), Some(0.5), Some(1.5)), 1.0, 0));
        assert!(t(Filter::FloatRange(score(), Some(1.0), Some(1.0)), 1.0, 0));
        assert!(!t(Filter::FloatRange(score(), Some(1.0000000000000002), None), 1.0, 0));
        assert!(t(Filter::FloatRange(score(), None, Some(-0.0)), 0.0, 0));
        assert!(t(Filter::Range(score(), Some(-3), Some(-2)), -2.5, 0));
        assert!(!t(Filter::Range(score(), Some(-2), None), -2.5, 0));
        assert!(t(Filter::Eq(score(), Value::Float(2.5)), 2.5, 0) && !t(Filter::Eq(score(), Value::Float(2.5)), 2.6, 0));
        assert!(t(Filter::Eq(score(), Value::Int(3)), 3.0, 0));
        assert!(t(Filter::In(score(), vec![Value::Str("x".into()), Value::Float(f64::NAN), Value::Int(3)]), 3.0, 0));
        assert!(t(Filter::Exists(score()), f64::INFINITY, 0));
        // NaN is no document's, on either side.
        assert!(!t(Filter::Eq(score(), Value::Float(f64::NAN)), 1.0, 0));
        assert!(!t(Filter::FloatRange(score(), Some(f64::NAN), None), 1.0, 0));
        assert!(!t(Filter::Exists(score()), f64::NAN, 0));
        // Floats asked of an integer column: the integers between them.
        let n = || "n".to_string();
        assert!(t(Filter::FloatRange(n(), Some(1.5), Some(2.5)), 0.0, 2));
        assert!(!t(Filter::FloatRange(n(), Some(1.5), Some(1.9)), 0.0, 2) && !t(Filter::FloatRange(n(), Some(1.5), Some(1.9)), 0.0, 1));
        assert!(t(Filter::FloatRange(n(), Some(f64::NEG_INFINITY), Some(1e300)), 0.0, i64::MAX));
        assert!(!t(Filter::FloatRange(n(), Some(1e300), None), 0.0, i64::MAX));
        assert!(t(Filter::Eq(n(), Value::Float(2.0)), 0.0, 2) && !t(Filter::Eq(n(), Value::Float(2.5)), 0.0, 2));
    }

    #[test]
    fn filters() {
        let cols = vec![("date".to_string(), Kind::Int), ("from".to_string(), Kind::Str)];
        let row = R(vec![Column::Int(vec![100]), Column::Str(vec![Some("Mia Müller <mia@x.org>".into())])]);
        let t = |f: Filter| f.bind(&cols).matches(&row);
        assert!(t(Filter::Range("date".into(), Some(50), None)));
        assert!(!t(Filter::Range("date".into(), Some(101), None)));
        assert!(t(Filter::Contains("from".into(), "MÜLLER".into())));
        assert!(t(Filter::Prefix("from".into(), "mia".into())));
        assert!(!t(Filter::Eq("from".into(), Value::Int(3))));
        assert!(!t(Filter::Exists("nope".into())));
        assert!(t(Filter::Not(Box::new(Filter::Exists("nope".into())))));
        assert!(t(Filter::And(vec![Filter::Exists("date".into()), Filter::Or(vec![Filter::Eq("date".into(), Value::Int(1)), Filter::Eq("date".into(), Value::Int(100))])])));
    }
}
