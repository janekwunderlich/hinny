//! Node.js bindings for hinny (the `hinny` package's native module), through Node-API.
//!
//! Calls are synchronous, as the engine's are: a query takes a fraction of a millisecond, less
//! than a round trip through the thread pool would add. `optimizeAsync`, `flushAsync` and
//! `verifyAsync` run the three calls that can take seconds on a large namespace on libuv's
//! thread pool instead, so a server's event loop keeps running.
//!
//! JavaScript has one number type, so conversions are explicit: keys and versions (unsigned
//! 64 bits) are taken as a safe integer or a bigint and given back as bigints; an integer-valued
//! number in an attribute or condition is an integer (`Value::Int`), any other a float; integer
//! attributes come back as numbers when they are safe integers and as bigints when they aren't.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use hinny::{Filter as HFilter, Value};
use napi::bindgen_prelude::*;
use napi::{Env, JsValue, Unknown, ValueType};
use napi_derive::napi;

// MARK: Errors

/// The `code` of every error the module throws itself.
pub enum Code {
    /// The input is wrong: a field or attribute not in the schema, a vector of other dimensions,
    /// NaN in a float attribute, an option of the wrong type, a batch over the limit.
    Invalid,
    /// A query ran past its `timeout`.
    Timeout,
    /// Another process has the namespace open for writing.
    Locked,
    /// The namespace, snapshot or database was closed.
    Closed,
    /// Reading or writing files failed (the message names the file).
    Io,
}

impl AsRef<str> for Code {
    fn as_ref(&self) -> &str {
        match self {
            Code::Invalid => "ERR_HINNY_INVALID",
            Code::Timeout => "ERR_HINNY_TIMEOUT",
            Code::Locked => "ERR_HINNY_LOCKED",
            Code::Closed => "ERR_HINNY_CLOSED",
            Code::Io => "ERR_HINNY_IO",
        }
    }
}

type Result<T> = std::result::Result<T, napi::Error<Code>>;

fn invalid(msg: impl ToString) -> napi::Error<Code> {
    napi::Error::new(Code::Invalid, msg)
}

fn code(e: &std::io::Error) -> Code {
    match e.kind() {
        // (A name that is taken, for a fork, is a wrong input too.)
        std::io::ErrorKind::InvalidInput | std::io::ErrorKind::AlreadyExists => Code::Invalid,
        std::io::ErrorKind::TimedOut => Code::Timeout,
        std::io::ErrorKind::WouldBlock => Code::Locked,
        _ => Code::Io,
    }
}

/// Errors from the engine and from Node-API, as errors with a `code`.
trait Js<T> {
    fn js(self) -> Result<T>;
}

impl<T> Js<T> for std::io::Result<T> {
    fn js(self) -> Result<T> {
        self.map_err(|e| napi::Error::new(code(&e), e.to_string()))
    }
}

impl<T> Js<T> for napi::Result<T> {
    fn js(self) -> Result<T> {
        self.map_err(|e| invalid(e.reason))
    }
}

// MARK: Values in

fn type_of(v: &Unknown<'_>) -> Result<ValueType> {
    v.get_type().js()
}

/// A property that is set: undefined and null are as if it weren't there.
fn present(v: Option<Unknown<'_>>) -> Result<Option<Unknown<'_>>> {
    match v {
        Some(v) if !matches!(type_of(&v)?, ValueType::Undefined | ValueType::Null) => Ok(Some(v)),
        _ => Ok(None),
    }
}

fn object<'e>(v: &Unknown<'e>, what: &str) -> Result<Object<'e>> {
    if type_of(v)? != ValueType::Object || v.is_array().js()? {
        return Err(invalid(format!("{what} is an object")));
    }
    unsafe { v.cast::<Object<'e>>() }.js()
}

fn array<'e>(v: &Unknown<'e>, what: &str) -> Result<Array<'e>> {
    if !v.is_array().js()? {
        return Err(invalid(format!("{what} is an array")));
    }
    unsafe { v.cast::<Array<'e>>() }.js()
}

/// The elements of an array.
fn elements<'e>(v: &Unknown<'e>, what: &str) -> Result<Vec<Unknown<'e>>> {
    let a = array(v, what)?;
    (0..a.len()).map(|i| a.get::<Unknown<'e>>(i).js()?.ok_or_else(|| invalid(format!("{what} has a hole at {i}")))).collect()
}

/// An object's own (enumerable) properties, in order, with undefined ones left out.
fn entries<'e>(o: &Object<'e>) -> Result<Vec<(String, Unknown<'e>)>> {
    let mut out = Vec::new();
    for k in Object::keys(o).js()? {
        if let Some(v) = present(o.get::<Unknown<'e>>(&k).js()?)? {
            out.push((k, v));
        }
    }
    Ok(out)
}

/// Options from an object, refusing names it doesn't know (a misspelt option would otherwise
/// be silently ignored).
fn options<'e>(v: Option<Unknown<'e>>, what: &str, known: &[&str]) -> Result<Vec<(String, Unknown<'e>)>> {
    let Some(v) = present(v)? else { return Ok(Vec::new()) };
    let e = entries(&object(&v, what)?)?;
    if let Some((k, _)) = e.iter().find(|(k, _)| !known.contains(&k.as_str())) {
        return Err(invalid(format!("{what} has no option {k:?} (it has {})", known.join(", "))));
    }
    Ok(e)
}

fn number(v: &Unknown<'_>, what: &str) -> Result<f64> {
    if type_of(v)? != ValueType::Number {
        return Err(invalid(format!("{what} is a number")));
    }
    unsafe { v.cast::<f64>() }.js()
}

fn count(v: &Unknown<'_>, what: &str) -> Result<usize> {
    let x = number(v, what)?;
    if !(x >= 0.0 && x.fract() == 0.0 && x <= MAX_SAFE) {
        return Err(invalid(format!("{what} is an integer, 0 or more")));
    }
    Ok(x as usize)
}

fn boolean(v: &Unknown<'_>, what: &str) -> Result<bool> {
    if type_of(v)? != ValueType::Boolean {
        return Err(invalid(format!("{what} is true or false")));
    }
    unsafe { v.cast::<bool>() }.js()
}

fn string(v: &Unknown<'_>, what: &str) -> Result<String> {
    if type_of(v)? != ValueType::String {
        return Err(invalid(format!("{what} is a string")));
    }
    unsafe { v.cast::<String>() }.js()
}

fn strings(v: &Unknown<'_>, what: &str) -> Result<Vec<String>> {
    elements(v, what)?.iter().map(|s| string(s, &format!("each of {what}"))).collect()
}

const MAX_SAFE: f64 = 9_007_199_254_740_991.0;

/// A bigint's value, if it fits in 64 bits: (negative, magnitude).
fn bigint(v: &Unknown<'_>) -> Result<Option<(bool, u64)>> {
    let b = unsafe { v.cast::<BigInt>() }.js()?;
    let magnitude = match b.words.iter().rposition(|w| *w != 0) {
        None => 0,
        Some(0) => b.words[0],
        Some(_) => return Ok(None),
    };
    Ok(Some((b.sign_bit && magnitude != 0, magnitude)))
}

/// An unsigned 64-bit integer (a key, a version): a safe integer or a bigint.
fn u64_of(v: &Unknown<'_>, what: &str) -> Result<u64> {
    match type_of(v)? {
        ValueType::Number => {
            let x = unsafe { v.cast::<f64>() }.js()?;
            if x >= 0.0 && x.fract() == 0.0 && x <= MAX_SAFE {
                Ok(x as u64)
            } else if x.fract() == 0.0 && x > MAX_SAFE {
                Err(invalid(format!("{what} {x} is above 2^53 - 1, where numbers lose precision: pass it as a bigint")))
            } else {
                Err(invalid(format!("{what} is an integer from 0 to 2^64 - 1, not {x}")))
            }
        }
        ValueType::BigInt => match bigint(v)? {
            Some((false, x)) => Ok(x),
            _ => Err(invalid(format!("{what} is an integer from 0 to 2^64 - 1"))),
        },
        _ => Err(invalid(format!("{what} is a number or a bigint"))),
    }
}

/// An attribute value: an integer-valued number (or a bigint, or a boolean as 0 or 1) is an
/// integer, any other number a float, a string a string.
fn value(v: &Unknown<'_>, what: &str) -> Result<Value> {
    match type_of(v)? {
        ValueType::Number => {
            let x = unsafe { v.cast::<f64>() }.js()?;
            // Integral and in i64's range (2^63 itself is out).
            const TWO_63: f64 = 9_223_372_036_854_775_808.0;
            if x.fract() == 0.0 && (-TWO_63..TWO_63).contains(&x) {
                Ok(Value::Int(x as i64))
            } else {
                Ok(Value::Float(x))
            }
        }
        ValueType::BigInt => match bigint(v)? {
            Some((false, x)) if x <= i64::MAX as u64 => Ok(Value::Int(x as i64)),
            Some((true, x)) if x <= 1 << 63 => Ok(Value::Int((x as i64).wrapping_neg())),
            _ => Err(invalid(format!("{what} is out of the range of 64-bit integers"))),
        },
        ValueType::String => Ok(Value::Str(unsafe { v.cast::<String>() }.js()?)),
        ValueType::Boolean => Ok(Value::Int(unsafe { v.cast::<bool>() }.js()? as i64)),
        t => Err(invalid(format!("{what} is a number, bigint or string, not {}", js_type(t)))),
    }
}

/// A number in a condition: an integer or a float (either finds a column of the other kind).
fn condition_number(v: &Unknown<'_>, what: &str) -> Result<hinny::Number> {
    match value(v, what)? {
        Value::Int(x) => Ok(hinny::Number::Int(x)),
        Value::Float(x) => Ok(hinny::Number::Float(x)),
        Value::Str(_) | Value::Tags(_) => Err(invalid(format!("{what} is a number, not a string"))),
    }
}

fn js_type(t: ValueType) -> &'static str {
    match t {
        ValueType::Undefined => "undefined",
        ValueType::Null => "null",
        ValueType::Boolean => "a boolean",
        ValueType::Number => "a number",
        ValueType::String => "a string",
        ValueType::Symbol => "a symbol",
        ValueType::Object => "an object",
        ValueType::Function => "a function",
        ValueType::BigInt => "a bigint",
        _ => "another type",
    }
}

/// A vector: a Float32Array (copied once), a Float64Array, or an array of numbers.
fn vector(v: &Unknown<'_>) -> Result<Vec<f32>> {
    if v.is_typedarray().js()? {
        if let Ok(a) = unsafe { v.cast::<&[f32]>() } {
            return Ok(a.to_vec());
        }
        if let Ok(a) = unsafe { v.cast::<&[f64]>() } {
            return Ok(a.iter().map(|x| *x as f32).collect());
        }
        return Err(invalid("a vector is a Float32Array, a Float64Array or an array of numbers"));
    }
    if !v.is_array().js()? {
        return Err(invalid("a vector is a Float32Array, a Float64Array or an array of numbers"));
    }
    elements(v, "a vector")?.iter().map(|x| number(x, "each element of a vector").map(|x| x as f32)).collect()
}

/// Terms with weights: an object of term → weight, or an array of [term, weight].
fn weighted(v: &Unknown<'_>, what: &str) -> Result<Vec<(String, f32)>> {
    if v.is_array().js()? {
        return elements(v, what)?
            .iter()
            .map(|pair| {
                let p = elements(pair, &format!("each of {what}"))?;
                if p.len() != 2 {
                    return Err(invalid(format!("each of {what} is [term, weight]")));
                }
                Ok((string(&p[0], "a term")?, number(&p[1], "a weight")? as f32))
            })
            .collect();
    }
    entries(&object(v, what)?)?.into_iter().map(|(t, w)| Ok((t, number(&w, "a weight")? as f32))).collect()
}

/// Field → terms with weights.
type SparseTerms = Vec<(String, Vec<(String, f32)>)>;

fn sparse(v: &Unknown<'_>) -> Result<SparseTerms> {
    entries(&object(v, "sparse")?)?.into_iter().map(|(field, terms)| Ok((field.clone(), weighted(&terms, &format!("sparse.{field}"))?))).collect()
}

fn filter_of(v: &Unknown<'_>) -> Result<HFilter> {
    let not_a_filter = || invalid("filter is a Filter, built with attr(): attr(\"year\").gte(2025)");
    if type_of(v)? != ValueType::Object {
        return Err(not_a_filter());
    }
    let f = unsafe { v.cast::<ClassInstance<'_, Filter>>() }.map_err(|_| not_a_filter())?;
    Ok(f.0.clone())
}

fn optional_filter(v: Option<Unknown<'_>>) -> Result<Option<HFilter>> {
    present(v)?.map(|f| filter_of(&f)).transpose()
}

// MARK: Values out

fn safe_int(x: i64) -> bool {
    x.unsigned_abs() <= MAX_SAFE as u64
}

/// An attribute's value for JavaScript: a number, a bigint (an integer beyond 2^53), a string,
/// or an array of strings (tags).
type JsAttr = Either4<f64, BigInt, String, Vec<String>>;

fn js_value(v: &Value) -> JsAttr {
    match v {
        Value::Int(x) if safe_int(*x) => Either4::A(*x as f64),
        Value::Int(x) => Either4::B(BigInt::from(*x)),
        Value::Float(x) => Either4::A(*x),
        Value::Str(s) => Either4::C(s.clone()),
        Value::Tags(tags) => Either4::D(tags.clone()),
    }
}

fn attrs_object<'e>(env: &'e Env, attrs: &[(String, Value)]) -> Result<Object<'e>> {
    let mut o = Object::new(env).js()?;
    for (k, v) in attrs {
        o.set(k, js_value(v)).js()?;
    }
    Ok(o)
}

fn text_object<'e>(env: &'e Env, text: &[(String, String)]) -> Result<Object<'e>> {
    let mut o = Object::new(env).js()?;
    for (k, v) in text {
        o.set(k, v.as_str()).js()?;
    }
    Ok(o)
}

/// Byte offsets in `s` as offsets in UTF-16 code units, as JavaScript indexes strings
/// (`snippet.slice(start, end)`). `bytes` is sorted.
fn utf16_offsets(s: &str, bytes: &[usize]) -> Vec<u32> {
    let mut out = Vec::with_capacity(bytes.len());
    let (mut at_byte, mut at_unit) = (0usize, 0u32);
    for &b in bytes {
        let b = b.min(s.len());
        if b >= at_byte {
            at_unit += s[at_byte..b].encode_utf16().count() as u32;
        } else {
            at_unit = s[..b].encode_utf16().count() as u32;
        }
        at_byte = b;
        out.push(at_unit);
    }
    out
}

fn hit<'e>(env: &'e Env, h: hinny::Hit) -> Result<Object<'e>> {
    let mut o = Object::new(env).js()?;
    o.set("key", BigInt::from(h.key)).js()?;
    o.set("score", h.score as f64).js()?;
    o.set("textScore", h.text_score.map(|x| x as f64)).js()?;
    o.set("vectorScore", h.vector_score.map(|x| x as f64)).js()?;
    o.set("attrs", attrs_object(env, &h.attributes)?).js()?;
    o.set("text", text_object(env, &h.text)?).js()?;
    match h.snippet {
        Some((field, s)) => {
            let mut bytes: Vec<usize> = s.highlights.iter().flat_map(|r| [r.start, r.end]).collect();
            let sorted = bytes.is_sorted();
            if !sorted {
                bytes.sort_unstable();
            }
            let units = utf16_offsets(&s.text, &bytes);
            let at = |b: usize| units[bytes.binary_search(&b).unwrap_or_else(|i| i)];
            let highlights: Vec<(u32, u32)> = s.highlights.iter().map(|r| (at(r.start), at(r.end))).collect();
            o.set("snippet", s.text).js()?;
            o.set("snippetField", field).js()?;
            o.set("highlights", highlights).js()?;
        }
        None => {
            o.set("snippet", Null).js()?;
            o.set("snippetField", Null).js()?;
            o.set("highlights", Vec::<(u32, u32)>::new()).js()?;
        }
    }
    Ok(o)
}

fn hits<'e>(env: &'e Env, hs: Vec<hinny::Hit>) -> Result<Array<'e>> {
    let mut a = env.create_array(hs.len() as u32).js()?;
    for (i, h) in hs.into_iter().enumerate() {
        a.set(i as u32, hit(env, h)?).js()?;
    }
    Ok(a)
}

fn explained<'e>(env: &'e Env, (hs, stats): (Vec<hinny::Hit>, Option<hinny::FusionStats>)) -> Result<Object<'e>> {
    let mut out = Object::new(env).js()?;
    out.set("hits", hits(env, hs)?).js()?;
    match stats {
        Some(s) => {
            let mut fusion = Object::new(env).js()?;
            for (name, r) in [("lexical", s.lexical), ("vector", s.vector)] {
                let mut d = Object::new(env).js()?;
                d.set("depth", r.depth as f64).js()?;
                d.set("found", r.found as f64).js()?;
                d.set("scores", r.scores.map(|(lo, hi)| (lo as f64, hi as f64))).js()?;
                fusion.set(name, d).js()?;
            }
            fusion.set("both", s.both as f64).js()?;
            fusion.set("fused", s.fused as f64).js()?;
            out.set("fusion", fusion).js()?;
        }
        None => out.set("fusion", Null).js()?,
    }
    Ok(out)
}

fn counts_by(counts: Vec<(Value, u64)>) -> Vec<(JsAttr, f64)> {
    counts.iter().map(|(v, n)| (js_value(v), *n as f64)).collect()
}

/// key → version as one array of pairs, [key, version, key, version, …]; `index.js` makes a Map.
fn version_pairs(versions: std::collections::HashMap<u64, u64>) -> BigUint64Array {
    let mut pairs: Vec<(u64, u64)> = versions.into_iter().collect();
    pairs.sort_unstable();
    BigUint64Array::new(pairs.into_iter().flat_map(|(k, v)| [k, v]).collect())
}

// MARK: Filters

/// A condition on attributes. Build one with `attr("year").gte(2025)`; combine with `and`,
/// `or` and `not`.
#[napi]
pub struct Filter(HFilter);

#[napi]
impl Filter {
    /// Both this and `other`.
    #[napi(catch_unwind)]
    pub fn and(&self, other: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(self.0.clone() & filter_of(&other)?))
    }

    /// This or `other`, or both.
    #[napi(catch_unwind)]
    pub fn or(&self, other: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(self.0.clone() | filter_of(&other)?))
    }

    /// Everything this filter doesn't pass.
    #[napi(catch_unwind)]
    pub fn not(&self) -> Filter {
        Filter(!self.0.clone())
    }

    #[napi(js_name = "toString", catch_unwind)]
    pub fn describe(&self) -> String {
        format!("Filter({:?})", self.0)
    }
}

/// An attribute to filter on: `attr("kind").eq("pdf")`, `attr("year").between(2020, 2025)`.
#[napi]
pub struct Attr(String);

#[napi]
impl Attr {
    #[napi(catch_unwind)]
    pub fn eq(&self, v: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(hinny::attr(&self.0).eq(value(&v, "the value")?)))
    }

    #[napi(catch_unwind)]
    pub fn ne(&self, v: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(hinny::attr(&self.0).ne(value(&v, "the value")?)))
    }

    #[napi(catch_unwind)]
    pub fn gt(&self, x: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(hinny::attr(&self.0).gt(condition_number(&x, "the bound")?)))
    }

    #[napi(catch_unwind)]
    pub fn gte(&self, x: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(hinny::attr(&self.0).gte(condition_number(&x, "the bound")?)))
    }

    #[napi(catch_unwind)]
    pub fn lt(&self, x: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(hinny::attr(&self.0).lt(condition_number(&x, "the bound")?)))
    }

    #[napi(catch_unwind)]
    pub fn lte(&self, x: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(hinny::attr(&self.0).lte(condition_number(&x, "the bound")?)))
    }

    /// From `low` to `high`, both included.
    #[napi(catch_unwind)]
    pub fn between(&self, low: Unknown<'_>, high: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(hinny::attr(&self.0).between(condition_number(&low, "low")?, condition_number(&high, "high")?)))
    }

    #[napi(catch_unwind)]
    pub fn one_of(&self, values: Unknown<'_>) -> Result<Filter> {
        let vs = elements(&values, "oneOf's values")?.iter().map(|v| value(v, "each value")).collect::<Result<Vec<_>>>()?;
        Ok(Filter(hinny::attr(&self.0).one_of(vs)))
    }

    /// A tags attribute has this tag.
    #[napi(catch_unwind)]
    pub fn has(&self, tag: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(hinny::attr(&self.0).has(string(&tag, "the tag")?)))
    }

    /// A tags attribute has at least one of these.
    #[napi(catch_unwind)]
    pub fn has_any(&self, tags: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(hinny::attr(&self.0).has_any(strings(&tags, "hasAny's tags")?)))
    }

    /// A tags attribute has every one of these.
    #[napi(catch_unwind)]
    pub fn has_all(&self, tags: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(hinny::attr(&self.0).has_all(strings(&tags, "hasAll's tags")?)))
    }

    /// Starts with `p` (ignoring case). On tags: one of them does.
    #[napi(catch_unwind)]
    pub fn prefix(&self, p: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(hinny::attr(&self.0).prefix(&string(&p, "the prefix")?)))
    }

    /// Contains `p` (ignoring case).
    #[napi(catch_unwind)]
    pub fn contains(&self, p: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(hinny::attr(&self.0).contains(&string(&p, "the part")?)))
    }

    /// The whole string matches a pattern: `*` is any run of characters, `?` any one, `\` takes
    /// the next as itself (`attr('path').glob('/home/*/taxes/*.pdf')`). On tags: one of them does.
    #[napi(catch_unwind)]
    pub fn glob(&self, pattern: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(hinny::attr(&self.0).glob(&string(&pattern, "the pattern")?)))
    }

    /// `glob`, whatever the case.
    #[napi(catch_unwind)]
    pub fn iglob(&self, pattern: Unknown<'_>) -> Result<Filter> {
        Ok(Filter(hinny::attr(&self.0).iglob(&string(&pattern, "the pattern")?)))
    }

    #[napi(catch_unwind)]
    pub fn exists(&self) -> Filter {
        Filter(hinny::attr(&self.0).exists())
    }

    #[napi(js_name = "toString", catch_unwind)]
    pub fn describe(&self) -> String {
        format!("attr({:?})", self.0)
    }
}

/// An attribute to filter on.
#[napi(catch_unwind)]
pub fn attr(name: Unknown<'_>) -> Result<Attr> {
    Ok(Attr(string(&name, "an attribute name")?))
}

/// A stable key for a string id (FNV-1a, 63 bits), the same as Python's `hinny.key` and C's
/// `hinny_key`.
#[napi(catch_unwind)]
pub fn key(id: Unknown<'_>) -> Result<BigInt> {
    let id = string(&id, "an id")?;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in id.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    Ok(BigInt::from(h & (u64::MAX >> 1)))
}

// MARK: Handles

/// What a `Namespace` refers to, until it's closed. Its snapshots are listed so that closing
/// the namespace closes them too: each holds the index open (and its lock).
struct NsHandle {
    ns: Mutex<Option<Arc<hinny::Namespace>>>,
    snapshots: Mutex<Vec<Weak<SnapHandle>>>,
    /// The namespaces its database has opened: where a fork of it is listed too.
    opened: Opened,
}

/// The namespaces a database has opened, so its `close()` closes them.
type Opened = Arc<Mutex<Vec<Weak<NsHandle>>>>;

/// A handle for a namespace just opened, listed with its database's.
fn adopt(opened: &Opened, name: String, ns: hinny::Namespace) -> Namespace {
    let handle = Arc::new(NsHandle { ns: Mutex::new(Some(Arc::new(ns))), snapshots: Mutex::new(Vec::new()), opened: opened.clone() });
    let mut open = opened.lock().unwrap_or_else(|e| e.into_inner());
    open.retain(|w| w.strong_count() > 0);
    open.push(Arc::downgrade(&handle));
    Namespace { h: handle, name }
}

type SnapHandle = Mutex<Option<Arc<hinny::Snapshot>>>;

impl NsHandle {
    fn get(&self) -> Result<Arc<hinny::Namespace>> {
        self.ns.lock().unwrap_or_else(|e| e.into_inner()).clone().ok_or_else(|| napi::Error::new(Code::Closed, "the namespace is closed"))
    }

    fn close(&self) {
        let ns = self.ns.lock().unwrap_or_else(|e| e.into_inner()).take();
        let snaps = std::mem::take(&mut *self.snapshots.lock().unwrap_or_else(|e| e.into_inner()));
        for s in snaps.iter().filter_map(Weak::upgrade) {
            s.lock().unwrap_or_else(|e| e.into_inner()).take();
        }
        drop(ns);
    }
}

// MARK: Databases

/// A directory of namespaces.
#[napi]
pub struct Database {
    db: hinny::Database,
    path: String,
    namespaces: Opened,
    closed: AtomicBool,
}

const LANGUAGES: &str = "arabic, danish, dutch, english, finnish, french, german, greek, hungarian, italian, norwegian, portuguese, romanian, russian, spanish, swedish, tamil or turkish";

fn schema_of(v: Option<Unknown<'_>>) -> Result<hinny::Schema> {
    let opts = options(v, "the schema", &["text", "ints", "floats", "strings", "tags", "sparse", "vector", "metric", "storeText", "language", "stopwords"])?;
    let get = |name: &str| opts.iter().find(|(k, _)| k == name).map(|(_, v)| v);
    let mut schema = hinny::Schema::new();
    if let Some(v) = get("storeText") {
        schema = schema.store_text(boolean(v, "storeText")?);
    }
    let stopwords = get("stopwords").map(|v| boolean(v, "stopwords")).transpose()?.unwrap_or(true);
    if let Some(v) = get("language") {
        let l = string(v, "language")?;
        let lang = hinny::Language::from_name(&l).ok_or_else(|| invalid(format!("no stemmer for {l:?}: {LANGUAGES}")))?;
        schema = schema.analyzer(hinny::Analyzer::language(lang).without_stopwords(stopwords));
    }
    if let Some(v) = get("text") {
        if type_of(v)? == ValueType::String {
            schema = schema.text(&string(v, "text")?, 1.0);
        } else if v.is_array().js()? {
            for f in strings(v, "text")? {
                schema = schema.text(&f, 1.0);
            }
        } else {
            for (f, b) in entries(&object(v, "text (a field name, an array of them, or field → boost)")?)? {
                schema = schema.text(&f, number(&b, &format!("text.{f}'s boost"))? as f32);
            }
        }
    }
    if let Some(v) = get("sparse") {
        for n in strings(v, "sparse")? {
            schema = schema.sparse(&n);
        }
    }
    if let Some(v) = get("ints") {
        for n in strings(v, "ints")? {
            schema = schema.int(&n);
        }
    }
    if let Some(v) = get("floats") {
        for n in strings(v, "floats")? {
            schema = schema.float(&n);
        }
    }
    if let Some(v) = get("strings") {
        for n in strings(v, "strings")? {
            schema = schema.string(&n);
        }
    }
    if let Some(v) = get("tags") {
        for n in strings(v, "tags")? {
            schema = schema.tags(&n);
        }
    }
    if let Some(v) = get("vector") {
        let d = count(v, "vector (its dimensions)")?;
        if d == 0 {
            return Err(invalid("vector (its dimensions) is 1 or more"));
        }
        schema = schema.vector(d);
    }
    if let Some(v) = get("metric") {
        let m = string(v, "metric")?;
        let metric = hinny::Metric::from_name(&m).ok_or_else(|| invalid(format!("no vector metric {m:?}: cosine, euclidean or dot")))?;
        let Some(d) = schema.vector else { return Err(invalid("a metric is for a vector: give its dimensions as vector")) };
        schema = schema.vector_with(d, metric);
    }
    Ok(schema)
}

#[napi]
impl Database {
    /// Opens the database in a directory, creating it if there is none.
    #[napi(constructor, catch_unwind)]
    pub fn new(path: Unknown<'_>) -> Result<Self> {
        let path = string(&path, "the path")?;
        Ok(Database { db: hinny::Database::open(&path).js()?, path, namespaces: Arc::new(Mutex::new(Vec::new())), closed: AtomicBool::new(false) })
    }

    #[napi(getter, catch_unwind)]
    pub fn path(&self) -> String {
        self.path.clone()
    }

    /// Opens a namespace, or creates it with this schema.
    #[napi(catch_unwind)]
    pub fn namespace(&self, name: Unknown<'_>, schema: Option<Unknown<'_>>) -> Result<Namespace> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(napi::Error::new(Code::Closed, "the database is closed"));
        }
        let name = string(&name, "the namespace's name")?;
        let schema = schema_of(schema)?;
        let ns = self.db.namespace(&name, schema).js()?;
        Ok(self.adopt(name, ns))
    }

    /// Opens a namespace that exists, with the schema it has.
    #[napi(catch_unwind)]
    pub fn open(&self, name: Unknown<'_>) -> Result<Namespace> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(napi::Error::new(Code::Closed, "the database is closed"));
        }
        let name = string(&name, "the namespace's name")?;
        let ns = self.db.open_namespace(&name).js()?;
        Ok(self.adopt(name, ns))
    }

    /// Deletes a namespace and everything in it. It must not be open: close its `Namespace`
    /// first (and its snapshots, which `close()` closes with it), or this throws
    /// ERR_HINNY_LOCKED and leaves it as it is.
    #[napi(catch_unwind)]
    pub fn delete(&self, name: Unknown<'_>) -> Result<()> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(napi::Error::new(Code::Closed, "the database is closed"));
        }
        self.db.delete_namespace(&string(&name, "the namespace's name")?).js()
    }

    /// The namespaces in the directory.
    #[napi(catch_unwind)]
    pub fn namespaces(&self) -> Result<Vec<String>> {
        self.db.namespaces().js()
    }

    /// Closes every namespace opened from this database (and their snapshots).
    #[napi(catch_unwind)]
    pub fn close(&self) {
        self.closed.store(true, Ordering::Relaxed);
        let open = std::mem::take(&mut *self.namespaces.lock().unwrap_or_else(|e| e.into_inner()));
        for h in open.iter().filter_map(Weak::upgrade) {
            h.close();
        }
    }
}

// MARK: Namespaces

/// Documents sharing a schema.
#[napi]
pub struct Namespace {
    h: Arc<NsHandle>,
    name: String,
}

/// A document's or a patch's `attrs`: name → value, an array being tags.
fn attributes(a: &Unknown<'_>) -> Result<Vec<(String, Value)>> {
    entries(&object(a, "attrs")?)?
        .into_iter()
        .map(|(name, v)| {
            // An array: tags.
            if v.is_array().js()? {
                let tags = strings(&v, &format!("attrs.{name}"))?;
                return Ok((name, Value::Tags(tags)));
            }
            match value(&v, &format!("attrs.{name}"))? {
                // The engine's mark for "no value" (`hinny::attrs::NO_INT`): it would be dropped.
                Value::Int(i64::MIN) => Err(invalid(format!("attrs.{name}: -2^63 is reserved (it marks a missing integer)"))),
                v => Ok((name, v)),
            }
        })
        .collect()
}

/// A patch from `{text, attrs, vector, version, sparse, remove}`: the parts of a document to
/// change.
fn patch_of(v: &Unknown<'_>) -> Result<hinny::Patch> {
    let o = options(Some(*v), "a patch", &["text", "attrs", "vector", "version", "sparse", "remove"])?;
    let get = |name: &str| o.iter().find(|(k, _)| k == name).map(|(_, v)| v);
    let mut p = hinny::Patch::new();
    if let Some(v) = get("version") {
        p = p.version(u64_of(v, "a version")?);
    }
    if let Some(t) = get("text") {
        for (field, text) in entries(&object(t, "text (field → text)")?)? {
            p = p.text(&field, string(&text, &format!("text.{field}"))?);
        }
    }
    if let Some(a) = get("attrs") {
        for (name, v) in attributes(a)? {
            p = match v {
                Value::Int(x) => p.int(&name, x),
                Value::Float(x) => p.float(&name, x),
                Value::Str(s) => p.string(&name, s),
                Value::Tags(t) => p.tags(&name, t),
            };
        }
    }
    if let Some(v) = get("vector") {
        p = p.vector(vector(v)?);
    }
    if let Some(v) = get("sparse") {
        for (field, terms) in sparse(v)? {
            p = p.sparse(&field, &terms);
        }
    }
    if let Some(r) = get("remove") {
        for name in strings(r, "remove")? {
            p = p.remove(&name);
        }
    }
    Ok(p)
}

/// A document from `{key, text, attrs, vector, version, sparse}`.
fn document(ns: &hinny::Namespace, v: &Unknown<'_>) -> Result<hinny::Document> {
    let o = options(Some(*v), "a document", &["key", "text", "attrs", "vector", "version", "sparse"])?;
    let get = |name: &str| o.iter().find(|(k, _)| k == name).map(|(_, v)| v);
    let key = u64_of(get("key").ok_or_else(|| invalid("a document needs a key"))?, "a key")?;
    let mut d = hinny::Document::new(key);
    if let Some(v) = get("version") {
        d = d.version(u64_of(v, "a version")?);
    }
    if let Some(v) = get("sparse") {
        for (field, terms) in sparse(v)? {
            d = d.sparse(&field, &terms);
        }
    }
    if let Some(t) = get("text") {
        if type_of(t)? == ValueType::String {
            // The schema's only text field (sparse fields aside).
            let schema = ns.schema();
            let fields: Vec<&String> = schema.text.iter().zip(&schema.analysis).filter(|(_, a)| !a.sparse).map(|((n, _), _)| n).collect();
            if fields.len() != 1 {
                return Err(invalid("text as a string needs a schema with one text field; pass an object of field → text"));
            }
            d = d.text(fields[0], string(t, "text")?);
        } else {
            for (field, text) in entries(&object(t, "text (a string, or field → text)")?)? {
                d = d.text(&field, string(&text, &format!("text.{field}"))?);
            }
        }
    }
    if let Some(a) = get("attrs") {
        for (name, v) in attributes(a)? {
            d = match v {
                Value::Int(x) => d.int(&name, x),
                Value::Float(x) => d.float(&name, x),
                Value::Str(s) => d.string(&name, s),
                Value::Tags(t) => d.tags(&name, t),
            };
        }
    }
    if let Some(v) = get("vector") {
        d = d.vector(vector(v)?);
    }
    Ok(d)
}

fn documents(ns: &hinny::Namespace, docs: &Unknown<'_>) -> Result<Vec<hinny::Document>> {
    elements(docs, "the documents")?.iter().map(|d| document(ns, d)).collect()
}

fn keys(v: &Unknown<'_>) -> Result<Vec<u64>> {
    elements(v, "the keys")?.iter().map(|k| u64_of(k, "a key")).collect()
}

/// The query options, read before the query runs.
struct Search {
    text: Option<String>,
    vector: Option<Vec<f32>>,
    sparse: SparseTerms,
    filter: Option<HFilter>,
    limit: usize,
    order_by: Option<(String, bool)>,
    all_words: bool,
    typing: usize,
    nprobe: usize,
    snippet: usize,
    with_text: bool,
    text_weight: f32,
    rrf: bool,
    candidates: (usize, usize, usize),
    timeout: Option<std::time::Duration>,
}

const QUERY_OPTIONS: &[&str] = &["text", "vector", "sparse", "filter", "limit", "orderBy", "allWords", "typing", "nprobe", "snippet", "withText", "textWeight", "fusion", "textCandidates", "vectorCandidates", "sparseCandidates", "timeout"];

impl Search {
    fn from(v: Option<Unknown<'_>>) -> Result<Search> {
        let mut s = Search { text: None, vector: None, sparse: Vec::new(), filter: None, limit: 10, order_by: None, all_words: false, typing: 0, nprobe: 0, snippet: 240, with_text: true, text_weight: 0.5, rrf: false, candidates: (0, 0, 0), timeout: None };
        for (k, v) in options(v, "the query", QUERY_OPTIONS)? {
            match k.as_str() {
                "text" => s.text = Some(string(&v, "text")?),
                "vector" => s.vector = Some(vector(&v)?),
                "sparse" => s.sparse = sparse(&v)?,
                "filter" => s.filter = Some(filter_of(&v)?),
                "limit" => s.limit = count(&v, "limit")?,
                "orderBy" => {
                    let o = string(&v, "orderBy")?;
                    s.order_by = Some(match o.strip_prefix('-') {
                        Some(n) => (n.to_string(), true),
                        None => (o, false),
                    });
                }
                "allWords" => s.all_words = boolean(&v, "allWords")?,
                "typing" => s.typing = count(&v, "typing")?,
                "nprobe" => s.nprobe = count(&v, "nprobe")?,
                "snippet" => s.snippet = count(&v, "snippet")?,
                "withText" => s.with_text = boolean(&v, "withText")?,
                "textWeight" => {
                    let w = number(&v, "textWeight")?;
                    if !(0.0..=1.0).contains(&w) {
                        return Err(invalid("textWeight is from 0 to 1"));
                    }
                    s.text_weight = w as f32;
                }
                "fusion" => {
                    s.rrf = match string(&v, "fusion")?.as_str() {
                        "convex" => false,
                        "rrf" => true,
                        other => return Err(invalid(format!("fusion is \"convex\" or \"rrf\", not {other:?}"))),
                    }
                }
                "textCandidates" => s.candidates.0 = count(&v, "textCandidates")?,
                "vectorCandidates" => s.candidates.1 = count(&v, "vectorCandidates")?,
                "sparseCandidates" => s.candidates.2 = count(&v, "sparseCandidates")?,
                "timeout" => {
                    let ms = number(&v, "timeout")?;
                    if !(ms.is_finite() && ms >= 0.0) {
                        return Err(invalid("timeout is a number of milliseconds, 0 or more"));
                    }
                    s.timeout = Some(std::time::Duration::from_secs_f64(ms / 1000.0));
                }
                _ => unreachable!(),
            }
        }
        Ok(s)
    }

    fn run(self, base: hinny::Query<'_>, explain: bool) -> std::io::Result<(Vec<hinny::Hit>, Option<hinny::FusionStats>)> {
        let mut q = base;
        for (field, terms) in &self.sparse {
            q = q.sparse(field, terms.iter().map(|(t, w)| (t.as_str(), *w)));
        }
        let fusion = if self.rrf { hinny::Fusion::Rrf { k: 60.0 } } else { hinny::Fusion::Convex { text_weight: self.text_weight } };
        q = q.limit(self.limit).nprobe(self.nprobe).snippet(self.snippet).with_text(self.with_text).fusion(fusion).candidates(self.candidates.0, self.candidates.1).sparse_candidates(self.candidates.2);
        if let Some(t) = self.text {
            q = q.text(t);
        }
        if let Some(v) = self.vector {
            q = q.vector(v);
        }
        if let Some(f) = self.filter {
            q = q.filter(f);
        }
        if let Some((name, desc)) = &self.order_by {
            q = q.order_by(name, *desc);
        }
        if self.all_words {
            q = q.all_words();
        }
        if self.typing > 0 {
            q = q.typing(self.typing);
        }
        if let Some(t) = self.timeout {
            q = q.deadline(t);
        }
        if explain { q.explain() } else { q.run().map(|h| (h, None)) }
    }
}

/// What `verify` read: segments checked against their checksums, segments from before there
/// were checksums (`optimize` rewrites them), and files of stored text.
#[napi(object)]
pub struct Verified {
    pub segments: u32,
    pub unchecked_segments: u32,
    pub text_files: u32,
}

impl From<hinny::index::Verified> for Verified {
    fn from(v: hinny::index::Verified) -> Verified {
        Verified { segments: v.segments as u32, unchecked_segments: v.unchecked_segments as u32, text_files: v.blobs as u32 }
    }
}

#[napi]
impl Namespace {
    #[napi(getter, catch_unwind)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// The schema: `{text, sparse, ints, floats, strings, tags, vector, metric, storeText}`.
    #[napi(getter, catch_unwind)]
    pub fn schema<'e>(&self, env: &'e Env) -> Result<Object<'e>> {
        let ns = self.h.get()?;
        let s = ns.schema();
        let mut o = Object::new(env).js()?;
        let mut text = Object::new(env).js()?;
        let mut sparse = Vec::new();
        for ((n, b), a) in s.text.iter().zip(&s.analysis) {
            if a.sparse {
                sparse.push(n.clone());
            } else {
                text.set(n, *b as f64).js()?;
            }
        }
        let kind = |k: hinny::Kind| s.attrs.iter().filter(|(_, x)| *x == k).map(|(n, _)| n.clone()).collect::<Vec<_>>();
        o.set("text", text).js()?;
        o.set("sparse", sparse).js()?;
        o.set("ints", kind(hinny::Kind::Int)).js()?;
        o.set("floats", kind(hinny::Kind::Float)).js()?;
        o.set("strings", kind(hinny::Kind::Str)).js()?;
        o.set("tags", kind(hinny::Kind::Tags)).js()?;
        o.set("vector", s.vector.map(|d| d as f64)).js()?;
        o.set("metric", s.vector.map(|_| s.metric.name())).js()?;
        o.set("storeText", s.store_text).js()?;
        Ok(o)
    }

    /// Adds a document, or replaces the one with its key.
    #[napi(catch_unwind)]
    pub fn upsert(&self, doc: Unknown<'_>) -> Result<()> {
        let ns = self.h.get()?;
        let d = document(&ns, &doc)?;
        ns.upsert(d).js()
    }

    /// Documents as one batch: all or none of them. Returns how many.
    #[napi(catch_unwind)]
    pub fn upsert_many(&self, docs: Unknown<'_>) -> Result<u32> {
        let ns = self.h.get()?;
        let batch = documents(&ns, &docs)?;
        let n = batch.len() as u32;
        ns.upsert_many(batch).js()?;
        Ok(n)
    }

    /// Upserts and deletes as one batch: all or none of it, and no query or snapshot sees part
    /// of it. Returns how many changes.
    #[napi(catch_unwind)]
    pub fn apply(&self, changes: Unknown<'_>) -> Result<u32> {
        let ns = self.h.get()?;
        let o = options(Some(changes), "the changes", &["upsert", "delete"])?;
        let mut batch = Vec::new();
        for (k, v) in &o {
            match k.as_str() {
                "upsert" => batch.extend(documents(&ns, v)?.into_iter().map(hinny::Change::Put)),
                _ => batch.extend(keys(v)?.into_iter().map(hinny::Change::Delete)),
            }
        }
        let n = batch.len() as u32;
        ns.apply(batch).js()?;
        Ok(n)
    }

    #[napi(catch_unwind)]
    pub fn delete(&self, key: Unknown<'_>) -> Result<()> {
        let ns = self.h.get()?;
        ns.delete(u64_of(&key, "a key")?).js()
    }

    /// Deletes these keys as one batch. Returns how many.
    #[napi(catch_unwind)]
    pub fn delete_many(&self, keys: Unknown<'_>) -> Result<u32> {
        let ns = self.h.get()?;
        let ks = self::keys(&keys)?;
        let n = ks.len() as u32;
        ns.apply(ks.into_iter().map(hinny::Change::Delete)).js()?;
        Ok(n)
    }

    /// A document's attributes, or null if there is no document with that key.
    #[napi(catch_unwind)]
    pub fn get<'e>(&self, env: &'e Env, key: Unknown<'_>) -> Result<Option<Object<'e>>> {
        let ns = self.h.get()?;
        ns.get(u64_of(&key, "a key")?).map(|a| attrs_object(env, &a)).transpose()
    }

    /// A document's stored text, field → text, or null.
    #[napi(catch_unwind)]
    pub fn text<'e>(&self, env: &'e Env, key: Unknown<'_>) -> Result<Option<Object<'e>>> {
        let ns = self.h.get()?;
        ns.text(u64_of(&key, "a key")?).map(|t| text_object(env, &t)).transpose()
    }

    #[napi(js_name = "_versions", catch_unwind)]
    pub fn versions_pairs(&self) -> Result<BigUint64Array> {
        Ok(version_pairs(self.h.get()?.versions()))
    }

    /// How many documents pass `filter` (every document, without one).
    #[napi(catch_unwind)]
    pub fn count(&self, filter: Option<Unknown<'_>>) -> Result<f64> {
        let ns = self.h.get()?;
        Ok(ns.count(optional_filter(filter)?.as_ref()) as f64)
    }

    /// Deletes every document that passes `filter` now; how many that was.
    #[napi(catch_unwind)]
    pub fn delete_where(&self, filter: Unknown<'_>) -> Result<f64> {
        let ns = self.h.get()?;
        let filter = optional_filter(Some(filter))?.ok_or_else(|| invalid("deleteWhere takes a filter"))?;
        Ok(ns.delete_where(&filter).js()? as f64)
    }

    /// Changes part of the document with a key and leaves the rest (see `patch_of`); with a
    /// condition, only if the document passes it. Whether it was changed: false if there is no
    /// such document, or it didn't pass.
    #[napi(catch_unwind)]
    pub fn patch(&self, key: Unknown<'_>, patch: Unknown<'_>, condition: Option<Unknown<'_>>) -> Result<bool> {
        let ns = self.h.get()?;
        let key = u64_of(&key, "a key")?;
        let patch = patch_of(&patch)?;
        match optional_filter(condition)? {
            Some(c) => ns.patch_if(key, &patch, &c),
            None => ns.patch(key, &patch),
        }
        .js()
    }

    /// `patch` for every document that passes `filter` now; how many that was.
    #[napi(catch_unwind)]
    pub fn patch_where(&self, filter: Unknown<'_>, patch: Unknown<'_>) -> Result<f64> {
        let ns = self.h.get()?;
        let filter = optional_filter(Some(filter))?.ok_or_else(|| invalid("patchWhere takes a filter"))?;
        let patch = patch_of(&patch)?;
        Ok(ns.patch_where(&filter, &patch).js()? as f64)
    }

    /// `upsert`, unless there is a document with this key that doesn't pass `condition`:
    /// written if there is none yet or if the one there passes. Whether it was written.
    #[napi(catch_unwind)]
    pub fn upsert_if(&self, doc: Unknown<'_>, condition: Unknown<'_>) -> Result<bool> {
        let ns = self.h.get()?;
        let d = document(&ns, &doc)?;
        ns.upsert_if(d, &filter_of(&condition)?).js()
    }

    /// `delete`, only if the document exists and passes `condition`. Whether it was deleted.
    #[napi(catch_unwind)]
    pub fn delete_if(&self, key: Unknown<'_>, condition: Unknown<'_>) -> Result<bool> {
        let ns = self.h.get()?;
        ns.delete_if(u64_of(&key, "a key")?, &filter_of(&condition)?).js()
    }

    /// How many documents have each value of `attr`, among those that pass `filter`: the most
    /// frequent first, at most `limit` (0: every value).
    #[napi(ts_return_type = "Array<[number | bigint | string, number]>", catch_unwind)]
    pub fn count_by(&self, attr: Unknown<'_>, filter: Option<Unknown<'_>>, limit: Option<Unknown<'_>>) -> Result<Vec<(JsAttr, f64)>> {
        let ns = self.h.get()?;
        let attr = string(&attr, "the attribute")?;
        let limit = present(limit)?.map(|l| count(&l, "limit")).transpose()?.unwrap_or(0);
        Ok(counts_by(ns.count_by(&attr, optional_filter(filter)?.as_ref(), limit)))
    }

    /// The number of documents.
    #[napi(getter, catch_unwind)]
    pub fn size(&self) -> Result<f64> {
        Ok(self.h.get()?.count(None) as f64)
    }

    #[napi(catch_unwind)]
    pub fn query<'e>(&self, env: &'e Env, options: Option<Unknown<'_>>) -> Result<Array<'e>> {
        let ns = self.h.get()?;
        let (h, _) = Search::from(options)?.run(ns.query(), false).js()?;
        hits(env, h)
    }

    #[napi(catch_unwind)]
    pub fn explain<'e>(&self, env: &'e Env, options: Option<Unknown<'_>>) -> Result<Object<'e>> {
        let ns = self.h.get()?;
        let r = Search::from(options)?.run(ns.query(), true).js()?;
        explained(env, r)
    }

    /// The namespace as of now.
    #[napi(catch_unwind)]
    pub fn snapshot(&self) -> Result<Snapshot> {
        let ns = self.h.get()?;
        let handle = Arc::new(Mutex::new(Some(Arc::new(ns.snapshot()))));
        let mut snaps = self.h.snapshots.lock().unwrap_or_else(|e| e.into_inner());
        snaps.retain(|w| w.strong_count() > 0);
        snaps.push(Arc::downgrade(&handle));
        Ok(Snapshot { h: handle })
    }

    #[napi(catch_unwind)]
    pub fn sync(&self) -> Result<()> {
        self.h.get()?.sync().js()
    }

    /// A namespace of its own, named `name` in the same database, holding what this one
    /// holds now. Closed with the database, as this one is.
    #[napi(catch_unwind)]
    pub fn fork(&self, name: Unknown<'_>) -> Result<Namespace> {
        let name = string(&name, "the fork's name")?;
        let fork = self.h.get()?.fork(&name).js()?;
        Ok(adopt(&self.h.opened, name, fork))
    }

    #[napi(catch_unwind)]
    pub fn flush(&self) -> Result<()> {
        self.h.get()?.flush().js()
    }

    #[napi(catch_unwind)]
    pub fn optimize(&self) -> Result<()> {
        self.h.get()?.optimize().js()
    }

    #[napi(catch_unwind)]
    pub fn verify(&self) -> Result<Verified> {
        Ok(self.h.get()?.verify().js()?.into())
    }

    #[napi(ts_return_type = "Promise<void>", catch_unwind)]
    pub fn flush_async(&self) -> Result<AsyncTask<Maintenance>> {
        Ok(AsyncTask::new(Maintenance { ns: self.h.get()?, op: Op::Flush }))
    }

    #[napi(ts_return_type = "Promise<void>", catch_unwind)]
    pub fn optimize_async(&self) -> Result<AsyncTask<Maintenance>> {
        Ok(AsyncTask::new(Maintenance { ns: self.h.get()?, op: Op::Optimize }))
    }

    #[napi(ts_return_type = "Promise<Verified>", catch_unwind)]
    pub fn verify_async(&self) -> Result<AsyncTask<Maintenance>> {
        Ok(AsyncTask::new(Maintenance { ns: self.h.get()?, op: Op::Verify }))
    }

    #[napi(catch_unwind)]
    pub fn stats<'e>(&self, env: &'e Env) -> Result<Object<'e>> {
        let s = self.h.get()?.stats();
        let mut o = Object::new(env).js()?;
        let fields: [(&str, f64); 13] = [
            ("documents", (s.passages - s.deleted) as f64),
            ("segments", s.segments as f64),
            ("inMemory", s.memory as f64),
            ("memtableBytes", s.memtable_bytes as f64),
            ("logBytes", s.log_bytes as f64),
            ("pinnedSegments", s.pinned_segments as f64),
            ("pinnedBytes", s.pinned_bytes as f64),
            ("vectorCaches", s.vector_caches as f64),
            ("vectorCacheBytes", s.vector_cache_bytes as f64),
            ("keptFilters", s.kept_filters as f64),
            ("keptFilterBytes", s.kept_filter_bytes as f64),
            ("vectorGenerations", s.vector_generations as f64),
            ("vectorClusters", s.vector_clusters as f64),
        ];
        for (k, v) in fields {
            o.set(k, v).js()?;
        }
        o.set("merging", s.merging).js()?;
        o.set("flushing", s.flushing).js()?;
        Ok(o)
    }

    /// The writer's limits: `maxBatchBytes`, the largest batch accepted, and `flushBytes`, the
    /// bytes in memory before a flush. Omitted ones stay as they are.
    #[napi(catch_unwind)]
    pub fn set_limits(&self, limits: Unknown<'_>) -> Result<()> {
        let ns = self.h.get()?;
        let (mut max_batch, mut flush) = ns.limits();
        for (k, v) in options(Some(limits), "the limits", &["maxBatchBytes", "flushBytes"])? {
            let n = count(&v, &k)?;
            if k == "maxBatchBytes" {
                max_batch = n;
            } else {
                flush = n;
            }
        }
        ns.set_limits(max_batch, flush);
        Ok(())
    }

    /// Lets go of the namespace (and its snapshots): writes are synchronized and the directory's
    /// lock released, so it can be opened again. Later calls throw ERR_HINNY_CLOSED.
    #[napi(catch_unwind)]
    pub fn close(&self) {
        self.h.close();
    }

    #[napi(getter, catch_unwind)]
    pub fn closed(&self) -> bool {
        self.h.get().is_err()
    }
}

// MARK: Slow calls off the event loop

pub enum Op {
    Flush,
    Optimize,
    Verify,
}

pub struct Maintenance {
    ns: Arc<hinny::Namespace>,
    op: Op,
}

impl Task for Maintenance {
    type Output = std::io::Result<Option<Verified>>;
    type JsValue = Either<(), Verified>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        Ok(match self.op {
            Op::Flush => self.ns.flush().map(|_| None),
            Op::Optimize => self.ns.optimize().map(|_| None),
            Op::Verify => self.ns.verify().map(|v| Some(v.into())),
        })
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        match output {
            Ok(None) => Ok(Either::A(())),
            Ok(Some(v)) => Ok(Either::B(v)),
            Err(e) => {
                // An Error with our `code` (a task's errors would otherwise be "GenericFailure").
                let mut err = env.create_error(napi::Error::new(Status::GenericFailure, e.to_string()))?;
                err.set("code", code(&e).as_ref())?;
                Err(napi::Error::from(err.to_unknown()))
            }
        }
    }
}

// MARK: Snapshots

/// The namespace as of one moment (`Namespace.snapshot()`).
#[napi]
pub struct Snapshot {
    h: Arc<SnapHandle>,
}

impl Snapshot {
    fn snap(&self) -> Result<Arc<hinny::Snapshot>> {
        self.h.lock().unwrap_or_else(|e| e.into_inner()).clone().ok_or_else(|| napi::Error::new(Code::Closed, "the snapshot is closed"))
    }
}

#[napi]
impl Snapshot {
    #[napi(catch_unwind)]
    pub fn query<'e>(&self, env: &'e Env, options: Option<Unknown<'_>>) -> Result<Array<'e>> {
        let s = self.snap()?;
        let (h, _) = Search::from(options)?.run(s.query(), false).js()?;
        hits(env, h)
    }

    #[napi(catch_unwind)]
    pub fn explain<'e>(&self, env: &'e Env, options: Option<Unknown<'_>>) -> Result<Object<'e>> {
        let s = self.snap()?;
        let r = Search::from(options)?.run(s.query(), true).js()?;
        explained(env, r)
    }

    #[napi(catch_unwind)]
    pub fn get<'e>(&self, env: &'e Env, key: Unknown<'_>) -> Result<Option<Object<'e>>> {
        let s = self.snap()?;
        s.get(u64_of(&key, "a key")?).map(|a| attrs_object(env, &a)).transpose()
    }

    #[napi(catch_unwind)]
    pub fn text<'e>(&self, env: &'e Env, key: Unknown<'_>) -> Result<Option<Object<'e>>> {
        let s = self.snap()?;
        s.text(u64_of(&key, "a key")?).map(|t| text_object(env, &t)).transpose()
    }

    #[napi(js_name = "_versions", catch_unwind)]
    pub fn versions_pairs(&self) -> Result<BigUint64Array> {
        Ok(version_pairs(self.snap()?.versions()))
    }

    #[napi(catch_unwind)]
    pub fn count(&self, filter: Option<Unknown<'_>>) -> Result<f64> {
        let s = self.snap()?;
        Ok(s.count(optional_filter(filter)?.as_ref()) as f64)
    }

    #[napi(ts_return_type = "Array<[number | bigint | string, number]>", catch_unwind)]
    pub fn count_by(&self, attr: Unknown<'_>, filter: Option<Unknown<'_>>, limit: Option<Unknown<'_>>) -> Result<Vec<(JsAttr, f64)>> {
        let s = self.snap()?;
        let attr = string(&attr, "the attribute")?;
        let limit = present(limit)?.map(|l| count(&l, "limit")).transpose()?.unwrap_or(0);
        Ok(counts_by(s.count_by(&attr, optional_filter(filter)?.as_ref(), limit)))
    }

    /// The sequence number of the newest batch it includes.
    #[napi(getter, catch_unwind)]
    pub fn seq(&self) -> Result<BigInt> {
        Ok(BigInt::from(self.snap()?.seq()))
    }

    /// Lets go of the snapshot: the segments only it held can be removed.
    #[napi(catch_unwind)]
    pub fn close(&self) {
        self.h.lock().unwrap_or_else(|e| e.into_inner()).take();
    }
}

impl Database {
    /// A handle for a namespace just opened, remembered so `close()` closes it.
    fn adopt(&self, name: String, ns: hinny::Namespace) -> Namespace {
        adopt(&self.namespaces, name, ns)
    }
}
