//! hinny's JSON: schemas, documents, batches of changes, queries and filters in; hits, attributes
//! and counts out. The shapes are the ones `crates/hinny-c/include/hinny.h` documents; the C
//! interface and the server both speak them through this crate, so a client written against one
//! reads the other.
//!
//! Errors are messages (`String`): everything here is about what the caller sent, so each says
//! what was wrong and what was expected. The engine's own errors (`std::io::Error`) come from the
//! calls the caller makes with what this crate builds.

use hinny::index::{Stats, Verified};
use hinny::{Analyzer, Change, Document, Filter, Fusion, Hit, Language, Metric, Namespace, Patch, Query, Schema, Value};
use hinny::{Kind, Snippet};
use serde_json::{Map, Value as Json, json};

pub type Fallible<T> = Result<T, String>;

// MARK: JSON in

pub fn parse(text: &str, what: &str) -> Fallible<Json> {
    serde_json::from_str(text).map_err(|e| format!("{what} isn't JSON: {e}"))
}

pub fn object<'a>(value: &'a Json, what: &str) -> Fallible<&'a Map<String, Json>> {
    value.as_object().ok_or_else(|| format!("{what} must be an object"))
}

/// A key: a number, or a string of digits (for languages whose numbers are doubles).
pub fn key(value: &Json) -> Fallible<u64> {
    match value {
        Json::Number(n) => n.as_u64().ok_or_else(|| format!("a key must be an integer from 0 to 2^64 - 1, not {n}")),
        Json::String(s) => s.parse().map_err(|_| format!("a key must be an integer, not {s:?}")),
        other => Err(format!("a key must be an integer, not {other}")),
    }
}

fn names(value: Option<&Json>, what: &str) -> Fallible<Vec<String>> {
    let Some(value) = value else { return Ok(Vec::new()) };
    value.as_array().ok_or_else(|| format!("{what} must be an array of names"))?.iter().map(|n| n.as_str().map(str::to_string).ok_or_else(|| format!("{what} must be an array of names"))).collect()
}

/// A schema from its JSON text.
pub fn schema(text: &str) -> Fallible<Schema> {
    schema_from(&parse(text, "the schema")?)
}

/// A schema from parsed JSON.
pub fn schema_from(value: &Json) -> Fallible<Schema> {
    let fields = object(value, "the schema")?;
    let mut schema = Schema::new();
    for (name, _) in fields {
        if !["text", "sparse", "ints", "floats", "strings", "tags", "vector", "store_text"].contains(&name.as_str()) {
            return Err(format!("the schema has no part {name:?} (text, sparse, ints, floats, strings, tags, vector, store_text)"));
        }
    }
    if let Some(texts) = fields.get("text") {
        for field in texts.as_array().ok_or("the schema's text must be an array")? {
            // A name alone, or {"name", "boost", "language"}.
            let (name, boost, language) = match field {
                Json::String(name) => (name.as_str(), 1.0, None),
                Json::Object(f) => (
                    f.get("name").and_then(Json::as_str).ok_or("a text field needs a name")?,
                    f.get("boost").map(|b| b.as_f64().ok_or("a text field's boost must be a number")).transpose()?.unwrap_or(1.0) as f32,
                    f.get("language").filter(|l| !l.is_null()).map(|l| l.as_str().ok_or("a text field's language must be a name")).transpose()?,
                ),
                _ => return Err("a text field must be a name or an object".to_string()),
            };
            schema = match language {
                Some(l) => schema.text_with(name, boost, Analyzer::language(Language::from_name(l).ok_or_else(|| format!("no analyzer for the language {l:?}"))?)),
                None => schema.text(name, boost),
            };
        }
    }
    for name in names(fields.get("sparse"), "the schema's sparse")? {
        schema = schema.sparse(&name);
    }
    for name in names(fields.get("ints"), "the schema's ints")? {
        schema = schema.int(&name);
    }
    for name in names(fields.get("floats"), "the schema's floats")? {
        schema = schema.float(&name);
    }
    for name in names(fields.get("strings"), "the schema's strings")? {
        schema = schema.string(&name);
    }
    for name in names(fields.get("tags"), "the schema's tags")? {
        schema = schema.tags(&name);
    }
    if let Some(v) = fields.get("vector").filter(|d| !d.is_null()) {
        let (dims, metric) = vector_part(v)?;
        schema = schema.vector_with(dims, metric);
    }
    if let Some(store) = fields.get("store_text") {
        schema = schema.store_text(store.as_bool().ok_or("the schema's store_text must be true or false")?);
    }
    Ok(schema)
}

/// A schema's vector: a number of dimensions (compared by cosine), or {"dimensions", "metric"}
/// with the metric "cosine", "euclidean" or "dot".
fn vector_part(value: &Json) -> Fallible<(usize, Metric)> {
    let dimensions = |d: Option<&Json>| -> Fallible<usize> { d.and_then(Json::as_u64).filter(|d| *d > 0).map(|d| d as usize).ok_or_else(|| "the schema's vector must be a number of dimensions, or {\"dimensions\": n, \"metric\": name}".to_string()) };
    let Json::Object(parts) = value else { return Ok((dimensions(Some(value))?, Metric::Cosine)) };
    for name in parts.keys() {
        if !["dimensions", "metric"].contains(&name.as_str()) {
            return Err(format!("the schema's vector has no part {name:?} (dimensions, metric)"));
        }
    }
    let metric = match parts.get("metric").filter(|m| !m.is_null()) {
        None => Metric::Cosine,
        Some(m) => {
            let name = m.as_str().ok_or("the schema's vector metric must be a name (cosine, euclidean, dot)")?;
            Metric::from_name(name).ok_or_else(|| format!("no vector metric {name:?} (cosine, euclidean, dot)"))?
        }
    };
    Ok((dimensions(parts.get("dimensions"))?, metric))
}

/// Weighted terms, {"term": weight}.
pub fn terms(value: &Json, what: &str) -> Fallible<Vec<(String, f32)>> {
    object(value, what)?.iter().map(|(t, w)| w.as_f64().map(|w| (t.clone(), w as f32)).ok_or_else(|| format!("{what}: the weight of {t:?} must be a number"))).collect()
}

/// A vector given as a JSON array of numbers.
pub fn vector(value: &Json, what: &str) -> Fallible<Vec<f32>> {
    let wrong = || format!("{what} must be an array of numbers");
    value.as_array().ok_or_else(wrong)?.iter().map(|x| x.as_f64().map(|x| x as f32).ok_or_else(wrong)).collect()
}

/// A vector given as base64 (the standard alphabet, padding optional) of little-endian `f32`s:
/// a quarter of the bytes of the same numbers written out in JSON, and no float parsing, for
/// bulk writes and for clients that already hold the floats in memory.
pub fn vector_b64(value: &Json, what: &str) -> Fallible<Vec<f32>> {
    let text = value.as_str().ok_or_else(|| format!("{what} must be a base64 string of little-endian 32-bit floats"))?;
    let bytes = base64(text).ok_or_else(|| format!("{what} isn't base64 (the standard alphabet, A-Z a-z 0-9 + /)"))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{what} holds {} bytes, not a whole number of 32-bit floats (4 bytes each)", bytes.len()));
    }
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

/// Decodes standard base64; `None` if it isn't. A dozen lines instead of a dependency.
fn base64(text: &str) -> Option<Vec<u8>> {
    let text = text.trim_end_matches('=');
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in text.bytes() {
        let six = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        };
        acc = (acc << 6) | six as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    // One character left over (6 bits) can't be a byte: the text was cut short.
    if bits >= 6 {
        return None;
    }
    Some(out)
}

/// Takes the vector out of an object that may hold one as "vector" (numbers) or "vector_b64"
/// (base64), not both: for a query, whose other parts are checked against a fixed list.
pub fn take_vector(parts: &mut Map<String, Json>, what: &str) -> Fallible<Option<Vec<f32>>> {
    let numbers = parts.remove("vector").filter(|v| !v.is_null());
    let encoded = parts.remove("vector_b64").filter(|v| !v.is_null());
    match (numbers, encoded) {
        (Some(_), Some(_)) => Err(format!("{what} has both vector and vector_b64; give one")),
        (Some(v), None) => vector(&v, &format!("{what}'s vector")).map(Some),
        (None, Some(v)) => vector_b64(&v, &format!("{what}'s vector_b64")).map(Some),
        (None, None) => Ok(None),
    }
}

/// A document; `vector` is its embedding when it isn't in the JSON.
/// The most completions of a last word a query in JSON may ask for (`"typing"`).
pub const MAX_TYPING: usize = 1000;

pub fn document(value: &Json, vector: Option<Vec<f32>>) -> Fallible<Document> {
    let fields = object(value, "a document")?;
    let mut doc = Document::new(key(fields.get("key").ok_or("a document needs a key")?)?);
    if fields.contains_key("vector") && fields.contains_key("vector_b64") {
        return Err("a document has both vector and vector_b64; give one".to_string());
    }
    // Inside the document or beside it, not both: which was meant can't be told.
    if vector.is_some() && (fields.contains_key("vector") || fields.contains_key("vector_b64")) {
        return Err("a document's vector is given twice: inside the document and beside it; give one".to_string());
    }
    for (name, part) in fields {
        match name.as_str() {
            "key" => {}
            "version" => doc = doc.version(part.as_u64().ok_or("a document's version must be a non-negative integer")?),
            "text" => {
                for (field, text) in object(part, "a document's text")? {
                    doc = doc.text(field, text.as_str().ok_or_else(|| format!("the text of {field:?} must be a string"))?);
                }
            }
            "sparse" => {
                for (field, weights) in object(part, "a document's sparse")? {
                    doc = doc.sparse(field, &terms(weights, "a document's sparse terms")?);
                }
            }
            "ints" => {
                for (name, v) in object(part, "a document's ints")? {
                    doc = doc.int(name, v.as_i64().ok_or_else(|| format!("{name:?} must be an integer"))?);
                }
            }
            "floats" => {
                for (name, v) in object(part, "a document's floats")? {
                    doc = doc.float(name, v.as_f64().ok_or_else(|| format!("{name:?} must be a number"))?);
                }
            }
            "strings" => {
                for (name, v) in object(part, "a document's strings")? {
                    doc = doc.string(name, v.as_str().ok_or_else(|| format!("{name:?} must be a string"))?);
                }
            }
            "tags" => {
                for (name, v) in object(part, "a document's tags")? {
                    doc = doc.tags(name, names(Some(v), &format!("the tags {name:?}"))?);
                }
            }
            "vector" => doc = doc.vector(self::vector(part, "a document's vector")?),
            "vector_b64" => doc = doc.vector(vector_b64(part, "a document's vector_b64")?),
            other => return Err(format!("a document has no part {other:?} (key, version, text, sparse, ints, floats, strings, tags, vector, vector_b64)")),
        }
    }
    Ok(match vector {
        Some(v) => doc.vector(v),
        None => doc,
    })
}

/// A patch: the parts of a document that are to change, as a document's but without the key
/// (`version`, `text`, `sparse`, `ints`, `floats`, `strings`, `tags`, `vector` or `vector_b64`),
/// and `"remove": [names]` of the text fields and attributes to take away. `vector` is the patch's
/// embedding when it isn't in the JSON. In a namespace with sparse fields, a patch gives each
/// one's terms again or removes it: they aren't kept as they were given.
pub fn patch(value: &Json, vector: Option<Vec<f32>>) -> Fallible<Patch> {
    let fields = object(value, "a patch")?;
    let mut patch = Patch::new();
    if fields.contains_key("vector") && fields.contains_key("vector_b64") {
        return Err("a patch has both vector and vector_b64; give one".to_string());
    }
    for (name, part) in fields {
        match name.as_str() {
            "version" => patch = patch.version(part.as_u64().ok_or("a patch's version must be a non-negative integer")?),
            "text" => {
                for (field, text) in object(part, "a patch's text")? {
                    patch = patch.text(field, text.as_str().ok_or_else(|| format!("the text of {field:?} must be a string"))?);
                }
            }
            "sparse" => {
                for (field, weights) in object(part, "a patch's sparse")? {
                    patch = patch.sparse(field, &terms(weights, "a patch's sparse terms")?);
                }
            }
            "ints" => {
                for (name, v) in object(part, "a patch's ints")? {
                    patch = patch.int(name, v.as_i64().ok_or_else(|| format!("{name:?} must be an integer"))?);
                }
            }
            "floats" => {
                for (name, v) in object(part, "a patch's floats")? {
                    patch = patch.float(name, v.as_f64().ok_or_else(|| format!("{name:?} must be a number"))?);
                }
            }
            "strings" => {
                for (name, v) in object(part, "a patch's strings")? {
                    patch = patch.string(name, v.as_str().ok_or_else(|| format!("{name:?} must be a string"))?);
                }
            }
            "tags" => {
                for (name, v) in object(part, "a patch's tags")? {
                    patch = patch.tags(name, names(Some(v), &format!("the tags {name:?}"))?);
                }
            }
            "vector" => patch = patch.vector(self::vector(part, "a patch's vector")?),
            "vector_b64" => patch = patch.vector(vector_b64(part, "a patch's vector_b64")?),
            "remove" => {
                for name in names(Some(part), "a patch's remove")? {
                    patch = patch.remove(&name);
                }
            }
            other => return Err(format!("a patch has no part {other:?} (version, text, sparse, ints, floats, strings, tags, vector, vector_b64, remove)")),
        }
    }
    Ok(match vector {
        Some(v) => patch.vector(v),
        None => patch,
    })
}

/// A batch of changes: an array of documents and of {"delete": key}.
pub fn changes(value: &Json) -> Fallible<Vec<Change>> {
    value
        .as_array()
        .ok_or("the changes must be an array")?
        .iter()
        .map(|c| match c.get("delete") {
            Some(k) if c.as_object().is_some_and(|o| o.len() == 1) => key(k).map(Change::Delete),
            _ => document(c, None).map(Change::Put),
        })
        .collect()
}

pub fn scalar(value: &Json) -> Fallible<Value> {
    match value {
        Json::String(s) => Ok(Value::Str(s.clone())),
        // A whole number is an integer, any other a float (either finds the other's column).
        Json::Number(n) => n.as_i64().map(Value::Int).or_else(|| n.as_f64().map(Value::Float)).ok_or_else(|| format!("{n} isn't a number an attribute can hold")),
        other => Err(format!("an attribute's value must be a number or a string, not {other}")),
    }
}

pub fn filter(value: &Json) -> Fallible<Filter> {
    let parts = object(value, "a filter")?;
    let (op, arg) = parts.iter().next().filter(|_| parts.len() == 1).ok_or("a filter is an object with one of eq, in, range, prefix, contains, glob, iglob, exists, and, or, not")?;
    // [name, ...]: the attribute and the rest.
    let named = |n: usize| -> Fallible<(String, &[Json])> {
        let a = arg.as_array().filter(|a| a.len() == n).ok_or_else(|| format!("{op} takes [name{}]", if n == 3 { ", min, max" } else { ", value" }))?;
        Ok((a[0].as_str().ok_or_else(|| format!("{op}: the attribute's name must be a string"))?.to_string(), &a[1..]))
    };
    let string = |v: &Json| v.as_str().map(str::to_string).ok_or_else(|| format!("{op} takes a string"));
    let list = || -> Fallible<Vec<Filter>> { arg.as_array().ok_or_else(|| format!("{op} takes an array of filters"))?.iter().map(filter).collect() };
    let whole = |v: &Json| v.is_null() || v.as_i64().is_some();
    let end = |v: &Json| -> Fallible<Option<i64>> { if v.is_null() { Ok(None) } else { v.as_i64().map(Some).ok_or_else(|| "a range's ends must be numbers or null".to_string()) } };
    let float_end = |v: &Json| -> Fallible<Option<f64>> { if v.is_null() { Ok(None) } else { v.as_f64().map(Some).ok_or_else(|| "a range's ends must be numbers or null".to_string()) } };
    Ok(match op.as_str() {
        "eq" => {
            let (name, rest) = named(2)?;
            Filter::Eq(name, scalar(&rest[0])?)
        }
        "in" => {
            let (name, rest) = named(2)?;
            Filter::In(name, rest[0].as_array().ok_or("in takes [name, [values]]")?.iter().map(scalar).collect::<Fallible<_>>()?)
        }
        "range" => {
            let (name, rest) = named(3)?;
            match whole(&rest[0]) && whole(&rest[1]) {
                true => Filter::Range(name, end(&rest[0])?, end(&rest[1])?),
                false => Filter::FloatRange(name, float_end(&rest[0])?, float_end(&rest[1])?),
            }
        }
        "prefix" => {
            let (name, rest) = named(2)?;
            Filter::Prefix(name, string(&rest[0])?)
        }
        "contains" => {
            let (name, rest) = named(2)?;
            Filter::Contains(name, string(&rest[0])?)
        }
        "glob" | "iglob" => {
            let (name, rest) = named(2)?;
            Filter::Glob(name, string(&rest[0])?, op == "iglob")
        }
        "exists" => Filter::Exists(string(arg)?),
        "and" => Filter::And(list()?),
        "or" => Filter::Or(list()?),
        "not" => Filter::Not(Box::new(filter(arg)?)),
        other => return Err(format!("no filter {other:?} (eq, in, range, prefix, contains, glob, iglob, exists, and, or, not)")),
    })
}

/// A query on `ns`, built from its JSON; `vector` is the query's embedding. A vector inside the
/// JSON isn't one of its parts here: a caller that allows one takes it out first (`take_vector`).
pub fn query<'a>(ns: &'a Namespace, value: &Json, vector: Option<Vec<f32>>) -> Fallible<Query<'a>> {
    let parts = object(value, "the query")?;
    let mut q = ns.query();
    let number = |name: &str| -> Fallible<Option<f64>> { parts.get(name).filter(|v| !v.is_null()).map(|v| v.as_f64().ok_or_else(|| format!("the query's {name} must be a number"))).transpose() };
    let count = |name: &str| -> Fallible<Option<usize>> { parts.get(name).filter(|v| !v.is_null()).map(|v| v.as_u64().map(|n| n as usize).ok_or_else(|| format!("the query's {name} must be a non-negative integer"))).transpose() };
    let flag = |name: &str| -> Fallible<Option<bool>> { parts.get(name).filter(|v| !v.is_null()).map(|v| v.as_bool().ok_or_else(|| format!("the query's {name} must be true or false"))).transpose() };
    for name in parts.keys() {
        if !["text", "sparse", "filter", "limit", "all_words", "typing", "order_by", "fusion", "text_weight", "with_text", "snippet", "nprobe", "timeout_ms", "text_candidates", "vector_candidates", "sparse_candidates"].contains(&name.as_str()) {
            return Err(format!("a query has no part {name:?}"));
        }
    }
    if let Some(t) = parts.get("text").filter(|v| !v.is_null()) {
        q = q.text(t.as_str().ok_or("the query's text must be a string")?);
    }
    if let Some(s) = parts.get("sparse").filter(|v| !v.is_null()) {
        for (field, weights) in object(s, "the query's sparse")? {
            q = q.sparse(field, terms(weights, "the query's sparse terms")?);
        }
    }
    if let Some(v) = vector {
        q = q.vector(v);
    }
    if let Some(f) = parts.get("filter").filter(|v| !v.is_null()) {
        q = q.filter(filter(f)?);
    }
    if let Some(n) = count("limit")? {
        q = q.limit(n);
    }
    if flag("all_words")? == Some(true) {
        q = q.all_words();
    }
    if let Some(n) = count("typing")? {
        // Every completion is another word searched for: a few dozen serve a search box, and
        // a number without end would let one request ask for every word that starts with a letter.
        if n > MAX_TYPING {
            return Err(format!("the query's typing is {n}; at most {MAX_TYPING} (it is how many completions of the last word are searched for)"));
        }
        q = q.typing(n);
    }
    if let Some(o) = parts.get("order_by").filter(|v| !v.is_null()) {
        let o = object(o, "the query's order_by")?;
        q = q.order_by(o.get("attribute").and_then(Json::as_str).ok_or("order_by needs an attribute")?, o.get("descending").and_then(Json::as_bool).unwrap_or(false));
    }
    match parts.get("fusion").filter(|v| !v.is_null()).map(|f| f.as_str()) {
        None => {}
        Some(Some("convex")) => {}
        Some(Some("rrf")) => q = q.fusion(Fusion::Rrf { k: 60.0 }),
        Some(_) => return Err("the query's fusion must be \"convex\" or \"rrf\"".to_string()),
    }
    if let Some(w) = number("text_weight")? {
        q = q.text_weight(w as f32);
    }
    if let Some(t) = flag("with_text")? {
        q = q.with_text(t);
    }
    if let Some(n) = count("snippet")? {
        q = q.snippet(n);
    }
    if let Some(n) = count("nprobe")? {
        q = q.nprobe(n);
    }
    let (t, v) = (count("text_candidates")?, count("vector_candidates")?);
    if t.is_some() || v.is_some() {
        q = q.candidates(t.unwrap_or(0), v.unwrap_or(0));
    }
    if let Some(n) = count("sparse_candidates")? {
        q = q.sparse_candidates(n);
    }
    if let Some(ms) = number("timeout_ms")? {
        q = q.deadline(std::time::Duration::from_secs_f64(ms.max(0.0) / 1000.0));
    }
    Ok(q)
}

// MARK: JSON out

/// A schema in the form `schema` reads, so what comes out goes back in. Text fields come before
/// sparse ones there; a schema built in Rust with a sparse field between text fields comes out
/// in that order and can't go back in as it is. A language's analyzer is written as its name,
/// which reads back with its stopwords (`Analyzer::language`). A vector compared by cosine is
/// its number of dimensions; by another metric, {"dimensions", "metric"}.
pub fn schema_json(schema: &Schema) -> Json {
    let mut text = Vec::new();
    let mut sparse = Vec::new();
    for (i, (name, boost)) in schema.text.iter().enumerate() {
        let analyzer = schema.analysis.get(i).copied().unwrap_or_default();
        if analyzer.sparse {
            sparse.push(json!(name));
            continue;
        }
        let mut field = Map::new();
        field.insert("name".into(), json!(name));
        field.insert("boost".into(), json!(boost));
        if let Some(language) = analyzer.language {
            field.insert("language".into(), json!(language.name()));
        }
        text.push(Json::Object(field));
    }
    let of = |kind: Kind| -> Vec<&str> { schema.attrs.iter().filter(|(_, k)| *k == kind).map(|(n, _)| n.as_str()).collect() };
    let vector = match schema.vector {
        Some(d) if schema.metric != Metric::Cosine => json!({"dimensions": d, "metric": schema.metric.name()}),
        d => json!(d),
    };
    json!({
        "text": text,
        "sparse": sparse,
        "ints": of(Kind::Int),
        "floats": of(Kind::Float),
        "strings": of(Kind::Str),
        "tags": of(Kind::Tags),
        "vector": vector,
        "store_text": schema.store_text,
    })
}

pub fn attribute_json(value: &Value) -> Json {
    match value {
        Value::Int(i) => json!(i),
        Value::Str(s) => json!(s),
        // (An infinity has no JSON: null.)
        Value::Float(x) => json!(x),
        Value::Tags(tags) => json!(tags),
    }
}

pub fn attributes(pairs: &[(String, Value)]) -> Json {
    Json::Object(pairs.iter().map(|(n, v)| (n.clone(), attribute_json(v))).collect())
}

pub fn texts(pairs: &[(String, String)]) -> Json {
    Json::Object(pairs.iter().map(|(n, t)| (n.clone(), json!(t))).collect())
}

/// A stored document, {"attributes": {..}, "text": {..}}, as `hinny_get` returns it.
pub fn stored(attrs: &[(String, Value)], text: &[(String, String)]) -> Json {
    json!({"attributes": attributes(attrs), "text": texts(text)})
}

fn snippet(field: &str, snippet: &Snippet) -> Json {
    let highlights: Vec<Json> = snippet.highlights.iter().map(|r| json!([r.start, r.end])).collect();
    json!({"field": field, "text": snippet.text, "highlights": highlights})
}

pub fn hit(h: &Hit) -> Json {
    let mut out = Map::new();
    out.insert("key".into(), json!(h.key));
    out.insert("score".into(), json!(h.score));
    if let Some(s) = h.text_score {
        out.insert("text_score".into(), json!(s));
    }
    if let Some(s) = h.vector_score {
        out.insert("vector_score".into(), json!(s));
    }
    out.insert("attributes".into(), attributes(&h.attributes));
    if !h.text.is_empty() {
        out.insert("text".into(), texts(&h.text));
    }
    if let Some((field, s)) = &h.snippet {
        out.insert("snippet".into(), snippet(field, s));
    }
    Json::Object(out)
}

pub fn hits(hits: &[Hit]) -> Json {
    Json::Array(hits.iter().map(hit).collect())
}

/// `count_by`'s answer: [[value, count]], the most frequent first.
pub fn counts(counts: &[(Value, u64)]) -> Json {
    Json::Array(counts.iter().map(|(value, n)| json!([attribute_json(value), n])).collect())
}

pub fn stats(s: &Stats) -> Json {
    json!({
        "documents": s.passages - s.deleted,
        "segments": s.segments,
        "in_memory": s.memory,
        "merging": s.merging,
        "flushing": s.flushing,
        "memtable_bytes": s.memtable_bytes,
        "log_bytes": s.log_bytes,
        "pinned_segments": s.pinned_segments,
        "pinned_bytes": s.pinned_bytes,
    })
}

pub fn verified(v: &Verified) -> Json {
    json!({"segments": v.segments, "unchecked_segments": v.unchecked_segments, "text_files": v.blobs})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b64(bytes: &[u8]) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let n = chunk.iter().enumerate().fold(0u32, |n, (i, b)| n | (*b as u32) << (16 - 8 * i));
            for i in 0..=chunk.len() {
                out.push(ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char);
            }
        }
        while out.len() % 4 != 0 {
            out.push('=');
        }
        out
    }

    #[test]
    fn base64_round_trips_every_length() {
        for len in 0..20u8 {
            let bytes: Vec<u8> = (0..len).map(|i| i.wrapping_mul(37).wrapping_add(250)).collect();
            let text = b64(&bytes);
            assert_eq!(base64(&text).unwrap(), bytes, "{text}");
            assert_eq!(base64(text.trim_end_matches('=')).unwrap(), bytes, "without padding");
        }
        assert_eq!(base64("AAAA").unwrap(), [0, 0, 0]);
        assert!(base64("A").is_none(), "one character is six bits, not a byte");
        assert!(base64("AA-A").is_none(), "the URL-safe alphabet isn't this one");
    }

    #[test]
    fn vector_b64_reads_little_endian_floats() {
        let v = [1.0f32, -0.5, 3.25];
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(vector_b64(&json!(b64(&bytes)), "v").unwrap(), v);
        assert!(vector_b64(&json!(b64(&bytes[..5])), "v").unwrap_err().contains("5 bytes"));
        assert!(vector_b64(&json!([1.0]), "v").unwrap_err().contains("base64 string"));
    }

    #[test]
    fn take_vector_takes_either_form_not_both() {
        let bytes: Vec<u8> = [0.5f32, 0.25].iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut parts = object(&json!({"text": "x", "vector": [0.5, 0.25]}), "q").unwrap().clone();
        assert_eq!(take_vector(&mut parts, "the query").unwrap(), Some(vec![0.5, 0.25]));
        assert!(!parts.contains_key("vector"));
        let mut parts = object(&json!({"vector_b64": b64(&bytes)}), "q").unwrap().clone();
        assert_eq!(take_vector(&mut parts, "the query").unwrap(), Some(vec![0.5, 0.25]));
        let mut parts = object(&json!({"vector": [1.0], "vector_b64": "AAAAAA=="}), "q").unwrap().clone();
        assert!(take_vector(&mut parts, "the query").unwrap_err().contains("both"));
    }

    #[test]
    fn patches_read_the_parts_of_a_document_but_the_key() {
        let given = json!({"version": 3, "text": {"body": "new"}, "sparse": {"terms": {"lease": 1.4}}, "ints": {"year": 2026}, "floats": {"score": 0.5}, "strings": {"kind": "pdf"}, "tags": {"labels": ["a"]}, "vector": [1.0, 0.0], "remove": ["old"]});
        assert!(patch(&given, None).is_ok());
        let bytes: Vec<u8> = [0.5f32, 0.25].iter().flat_map(|x| x.to_le_bytes()).collect();
        assert!(patch(&json!({"vector_b64": b64(&bytes)}), None).is_ok());
        assert!(patch(&json!({}), Some(vec![1.0])).is_ok(), "the vector may come beside it");
        assert!(patch(&json!({"vector": [1.0], "vector_b64": "AAAAAA=="}), None).unwrap_err().contains("both"));
        let m = patch(&json!({"key": 1}), None).unwrap_err();
        assert!(m.contains("no part \"key\"") && m.contains("remove"), "{m}");
        assert!(patch(&json!({"remove": "old"}), None).unwrap_err().contains("array of names"));
        assert!(patch(&json!({"ints": {"year": "x"}}), None).unwrap_err().contains("integer"));
        assert!(patch(&json!([1]), None).unwrap_err().contains("object"));
    }

    #[test]
    fn schemas_come_out_as_they_go_in() {
        let given = json!({"text": [{"name": "title", "boost": 2.0}, {"name": "body", "boost": 1.0, "language": "english"}], "sparse": ["terms"], "ints": ["year"], "floats": ["score"], "strings": ["kind"], "tags": ["labels"], "vector": 3, "store_text": false});
        let schema = schema_from(&given).unwrap();
        assert_eq!(schema_json(&schema), given);
        assert_eq!(schema_json(&schema_from(&schema_json(&schema)).unwrap()), given);
        let bare = schema_json(&Schema::new());
        assert_eq!(bare["vector"], Json::Null);
        assert_eq!(bare["store_text"], json!(true));
    }

    #[test]
    fn a_vector_names_its_metric_unless_it_is_cosine() {
        assert_eq!(schema_from(&json!({"vector": 3})).unwrap().metric, Metric::Cosine);
        for metric in [Metric::Cosine, Metric::Euclidean, Metric::Dot] {
            let given = json!({"vector": {"dimensions": 3, "metric": metric.name()}});
            let schema = schema_from(&given).unwrap();
            assert_eq!((schema.vector, schema.metric), (Some(3), metric));
            let out = schema_json(&schema);
            // Cosine comes out as the number it always was.
            let want = if metric == Metric::Cosine { json!(3) } else { given["vector"].clone() };
            assert_eq!(out["vector"], want, "{}", metric.name());
            assert_eq!(schema_from(&out).unwrap(), schema, "{} goes back in", metric.name());
        }
        assert_eq!(schema_from(&json!({"vector": {"dimensions": 3}})).unwrap().metric, Metric::Cosine);
        let m = schema_from(&json!({"vector": {"dimensions": 3, "metric": "manhattan"}})).unwrap_err();
        assert!(m.contains("\"manhattan\"") && m.contains("cosine, euclidean, dot"), "{m}");
        assert!(schema_from(&json!({"vector": {"metric": "dot"}})).unwrap_err().contains("number of dimensions"));
        assert!(schema_from(&json!({"vector": {"dimensions": 0, "metric": "dot"}})).unwrap_err().contains("number of dimensions"));
        assert!(schema_from(&json!({"vector": {"dimensions": 3, "metric": 1}})).unwrap_err().contains("must be a name"));
        assert!(schema_from(&json!({"vector": {"dims": 3}})).unwrap_err().contains("no part \"dims\""));
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("hinny-json-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// Three vectors against the query [1, 0]: by cosine the short one in the query's direction
    /// is first and [3, 1] second; by Euclidean distance [1, 1] is nearer than [3, 1]; by dot
    /// product the long [3, 1] is first and the short one last.
    #[test]
    fn a_namespace_ranks_by_its_metric() {
        let dir = scratch("metrics");
        let db = hinny::Database::open(&dir).unwrap();
        let vectors = [(1u64, [0.5f32, 0.0]), (2, [3.0, 1.0]), (3, [1.0, 1.0])];
        let cases = [("cosine", [(1u64, 1.0f32), (2, 0.948_683), (3, std::f32::consts::FRAC_1_SQRT_2)]), ("euclidean", [(1, -0.25), (3, -1.0), (2, -5.0)]), ("dot", [(2, 3.0), (3, 1.0), (1, 0.5)])];
        for (metric, want) in cases {
            let ns = db.namespace(metric, schema_from(&json!({"vector": {"dimensions": 2, "metric": metric}})).unwrap()).unwrap();
            for (key, v) in vectors {
                ns.upsert(document(&json!({"key": key, "vector": v}), None).unwrap()).unwrap();
            }
            let hits = query(&ns, &json!({}), Some(vec![1.0, 0.0])).unwrap().run().unwrap();
            let got: Vec<(u64, f32)> = hits.iter().map(|h| (h.key, h.score)).collect();
            assert_eq!(got.len(), 3, "{metric}");
            for ((gk, gs), (wk, ws)) in got.iter().zip(want) {
                assert_eq!(*gk, wk, "{metric}: {got:?}");
                assert!((gs - ws).abs() < 1e-4, "{metric}: {got:?}");
            }
            assert_eq!(schema_json(ns.schema())["vector"], if metric == "cosine" { json!(2) } else { json!({"dimensions": 2, "metric": metric}) });
            drop(ns);
            let other = if metric == "dot" { "euclidean" } else { "dot" };
            let m = db.namespace(metric, schema_from(&json!({"vector": {"dimensions": 2, "metric": other}})).unwrap()).err().unwrap().to_string();
            assert!(m.contains("another schema"), "{metric} reopened as {other}: {m}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patches_give_sparse_terms_as_a_document_does() {
        assert!(patch(&json!({"sparse": {"terms": {"lease": 1.4}}}), None).is_ok());
        assert!(patch(&json!({"sparse": {"terms": {"lease": "x"}}}), None).unwrap_err().contains("must be a number"));
        assert!(patch(&json!({"sparse": ["lease"]}), None).unwrap_err().contains("object"));
        let dir = scratch("sparse-patch");
        let db = hinny::Database::open(&dir).unwrap();
        let ns = db.namespace("docs", schema_from(&json!({"text": ["body"], "sparse": ["terms"], "ints": ["year"]})).unwrap()).unwrap();
        ns.upsert(document(&json!({"key": 1, "text": {"body": "the lease"}, "sparse": {"terms": {"lease": 1.4}}, "ints": {"year": 2025}}), None).unwrap()).unwrap();
        let m = ns.patch(1, &patch(&json!({"ints": {"year": 2026}}), None).unwrap()).unwrap_err().to_string();
        assert!(m.contains("sparse"), "{m}");
        assert!(ns.patch(1, &patch(&json!({"ints": {"year": 2026}, "sparse": {"terms": {"lease": 1.4}}}), None).unwrap()).unwrap());
        assert_eq!(ns.get(1).unwrap().iter().find(|(n, _)| n == "year").map(|(_, v)| v.clone()), Some(Value::Int(2026)));
        let found = |ns: &Namespace| query(ns, &json!({"sparse": {"terms": {"lease": 1.0}}}), None).unwrap().run().unwrap().iter().map(|h| h.key).collect::<Vec<u64>>();
        assert_eq!(found(&ns), [1], "the terms are still there");
        assert!(ns.patch(1, &patch(&json!({"ints": {"year": 2027}, "remove": ["terms"]}), None).unwrap()).unwrap());
        assert_eq!(found(&ns), Vec::<u64>::new(), "removed");
        drop(ns);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn glob_and_iglob_take_a_name_and_a_pattern() {
        assert_eq!(filter(&json!({"glob": ["path", "/home/*/taxes/*.pdf"]})).unwrap(), Filter::Glob("path".into(), "/home/*/taxes/*.pdf".into(), false));
        assert_eq!(filter(&json!({"iglob": ["path", "*.PDF"]})).unwrap(), Filter::Glob("path".into(), "*.PDF".into(), true));
        assert_eq!(filter(&json!({"not": {"glob": ["path", "a?c"]}})).unwrap(), Filter::Not(Box::new(Filter::Glob("path".into(), "a?c".into(), false))));
        assert!(filter(&json!({"glob": ["path", 1]})).unwrap_err().contains("takes a string"));
        assert!(filter(&json!({"iglob": "path"})).unwrap_err().contains("takes [name, value]"));
        let m = filter(&json!({"like": ["path", "a"]})).unwrap_err();
        assert!(m.contains("glob") && m.contains("iglob"), "{m}");
    }
}
