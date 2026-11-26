//! Python bindings for hinny (the `hinny` package's native module, `hinny._hinny`).
//!
//! Writes and queries release the GIL, so Python threads search and write in parallel. Vectors
//! are read through the buffer protocol: a NumPy float32 array is used in place, without a copy.

use std::collections::HashMap;

use hinny::{Filter, Value};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyString};

fn err(e: std::io::Error) -> PyErr {
    match e.kind() {
        std::io::ErrorKind::InvalidInput => PyValueError::new_err(e.to_string()),
        std::io::ErrorKind::TimedOut => pyo3::exceptions::PyTimeoutError::new_err(e.to_string()),
        // (Subclasses of OSError, as the rest are: a caller that wants to tell "still open" from
        // "not there" can, and one that catches OSError catches them all.)
        std::io::ErrorKind::AlreadyExists => pyo3::exceptions::PyFileExistsError::new_err(e.to_string()),
        std::io::ErrorKind::NotFound => pyo3::exceptions::PyFileNotFoundError::new_err(e.to_string()),
        std::io::ErrorKind::WouldBlock => pyo3::exceptions::PyBlockingIOError::new_err(e.to_string()),
        _ => PyOSError::new_err(e.to_string()),
    }
}

/// Terms with weights: a dict of term → weight, or a sequence of (term, weight).
fn weighted(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(String, f32)>> {
    if let Ok(d) = obj.cast::<PyDict>() {
        return d.iter().map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<f32>()?))).collect();
    }
    obj.try_iter()?.map(|i| i?.extract::<(String, f32)>()).collect()
}

/// A Python string as text: lone surrogates (which JSON escapes and some mail produce, and which
/// aren't Unicode) become U+FFFD instead of failing the document.
fn lossy(s: &Bound<'_, PyString>) -> String {
    s.to_string_lossy().into_owned()
}

/// A vector from a 1-D float32 or float64 buffer (NumPy), or any sequence of numbers.
fn vector(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(b) = PyBuffer::<f32>::get(obj) {
        if b.dimensions() > 1 {
            return Err(PyValueError::new_err("a vector is one-dimensional"));
        }
        return b.to_vec(obj.py());
    }
    if let Ok(b) = PyBuffer::<f64>::get(obj) {
        if b.dimensions() > 1 {
            return Err(PyValueError::new_err("a vector is one-dimensional"));
        }
        return Ok(b.to_vec(obj.py())?.into_iter().map(|x| x as f32).collect());
    }
    let v: Vec<f64> = obj.extract().map_err(|_| PyTypeError::new_err("a vector is a NumPy array or a sequence of numbers"))?;
    Ok(v.into_iter().map(|x| x as f32).collect())
}

fn value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_instance_of::<PyBool>() {
        return Ok(Value::Int(obj.extract::<bool>()? as i64));
    }
    if obj.is_instance_of::<PyInt>() {
        return Ok(Value::Int(obj.extract()?));
    }
    if obj.is_instance_of::<PyString>() {
        return Ok(Value::Str(obj.extract()?));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(Value::Float(obj.extract()?));
    }
    Err(PyTypeError::new_err(format!("attributes are integers, floats or strings, not {}", obj.get_type().name()?)))
}

/// Tags: a list, tuple or set of strings.
fn tags(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if obj.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err("tags are a list of strings, not one string"));
    }
    obj.try_iter()
        .map_err(|_| PyTypeError::new_err(format!("tags are a list of strings, not {}", obj.get_type().name().map(|n| n.to_string()).unwrap_or_default())))?
        .map(|t| t?.extract::<String>().map_err(|_| PyTypeError::new_err("a tag is a string")))
        .collect()
}

/// A patch from what `ns.patch` takes: `text` a dict of field → text, `attrs` as for `upsert`,
/// `vector`, `version`, `sparse` as for `upsert`, and the names to `remove`.
fn patch_of(text: Option<&Bound<'_, PyAny>>, attrs: Option<&Bound<'_, PyDict>>, vec: Option<&Bound<'_, PyAny>>, version: Option<u64>, remove: &[String], sparse: Option<&Bound<'_, PyAny>>) -> PyResult<hinny::Patch> {
    let mut p = hinny::Patch::new();
    if let Some(v) = version {
        p = p.version(v);
    }
    if let Some(sp) = sparse.filter(|s| !s.is_none()) {
        let dict = sp.cast::<PyDict>().map_err(|_| PyTypeError::new_err("sparse is a dict of field → {term: weight}"))?;
        for (field, terms) in dict.iter() {
            p = p.sparse(&field.extract::<String>()?, &weighted(&terms)?);
        }
    }
    if let Some(text) = text.filter(|t| !t.is_none()) {
        let dict = text.cast::<PyDict>().map_err(|_| PyTypeError::new_err("text is a dict of field → text"))?;
        for (k, v) in dict.iter() {
            let v = v.cast::<PyString>().map_err(|_| PyTypeError::new_err("a field's text is a string"))?;
            p = p.text(&k.extract::<String>()?, lossy(v));
        }
    }
    if let Some(attrs) = attrs {
        for (k, v) in attrs.iter() {
            let name: String = k.extract()?;
            p = match attribute(&v)? {
                Value::Int(x) => p.int(&name, x),
                Value::Str(s) => p.string(&name, s),
                Value::Float(x) => p.float(&name, x),
                Value::Tags(t) => p.tags(&name, t),
            };
        }
    }
    if let Some(v) = vec.filter(|v| !v.is_none()) {
        p = p.vector(vector(v)?);
    }
    for name in remove {
        p = p.remove(name);
    }
    Ok(p)
}

/// A document's attribute: what `value` takes, or tags (a list, tuple or set of strings).
fn attribute(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_instance_of::<pyo3::types::PyList>() || obj.is_instance_of::<pyo3::types::PyTuple>() || obj.is_instance_of::<pyo3::types::PySet>() || obj.is_instance_of::<pyo3::types::PyFrozenSet>() {
        return Ok(Value::Tags(tags(obj)?));
    }
    value(obj)
}

/// A number in a condition: an int for integers, a float for floats (either finds a column of
/// the other kind).
fn number(obj: &Bound<'_, PyAny>) -> PyResult<hinny::Number> {
    match value(obj)? {
        Value::Int(x) => Ok(hinny::Number::Int(x)),
        Value::Float(x) => Ok(hinny::Number::Float(x)),
        Value::Str(_) | Value::Tags(_) => Err(PyTypeError::new_err("a number, not a string")),
    }
}

fn to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Int(x) => x.into_pyobject(py)?.into_any(),
        Value::Str(s) => s.into_pyobject(py)?.into_any(),
        Value::Float(x) => x.into_pyobject(py)?.into_any(),
        Value::Tags(tags) => pyo3::types::PyList::new(py, tags)?.into_any(),
    })
}

fn attrs_dict<'py>(py: Python<'py>, attrs: &[(String, Value)]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in attrs {
        d.set_item(k, to_py(py, v)?)?;
    }
    Ok(d)
}

// MARK: Filters

/// A condition on attributes. Build one with `attr("year") >= 2025`, combine with `&`, `|`, `~`.
#[pyclass(module = "hinny", frozen, skip_from_py_object, name = "Filter")]
#[derive(Clone)]
pub struct PyFilter(Filter);

#[pymethods]
impl PyFilter {
    fn __and__(&self, other: &PyFilter) -> PyFilter {
        PyFilter(Filter::And(vec![self.0.clone(), other.0.clone()]))
    }

    fn __or__(&self, other: &PyFilter) -> PyFilter {
        PyFilter(Filter::Or(vec![self.0.clone(), other.0.clone()]))
    }

    fn __invert__(&self) -> PyFilter {
        PyFilter(Filter::Not(Box::new(self.0.clone())))
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err("a Filter has no truth value: combine filters with &, | and ~ (not and, or, not)"))
    }

    fn __repr__(&self) -> String {
        format!("Filter({:?})", self.0)
    }
}

/// An attribute to filter on: `attr("kind") == "pdf"`, `attr("year").between(2020, 2025)`.
#[pyclass(module = "hinny", frozen, name = "Attr")]
pub struct PyAttr(String);

#[pymethods]
impl PyAttr {
    fn __eq__(&self, v: &Bound<'_, PyAny>) -> PyResult<PyFilter> {
        Ok(PyFilter(Filter::Eq(self.0.clone(), value(v)?)))
    }

    fn __ne__(&self, v: &Bound<'_, PyAny>) -> PyResult<PyFilter> {
        Ok(PyFilter(Filter::Not(Box::new(Filter::Eq(self.0.clone(), value(v)?)))))
    }

    fn __ge__(&self, x: &Bound<'_, PyAny>) -> PyResult<PyFilter> {
        Ok(PyFilter(hinny::attr(&self.0).gte(number(x)?)))
    }

    fn __gt__(&self, x: &Bound<'_, PyAny>) -> PyResult<PyFilter> {
        Ok(PyFilter(hinny::attr(&self.0).gt(number(x)?)))
    }

    fn __le__(&self, x: &Bound<'_, PyAny>) -> PyResult<PyFilter> {
        Ok(PyFilter(hinny::attr(&self.0).lte(number(x)?)))
    }

    fn __lt__(&self, x: &Bound<'_, PyAny>) -> PyResult<PyFilter> {
        Ok(PyFilter(hinny::attr(&self.0).lt(number(x)?)))
    }

    /// From `low` to `high`, inclusive.
    fn between(&self, low: &Bound<'_, PyAny>, high: &Bound<'_, PyAny>) -> PyResult<PyFilter> {
        Ok(PyFilter(hinny::attr(&self.0).between(number(low)?, number(high)?)))
    }

    fn one_of(&self, values: &Bound<'_, PyAny>) -> PyResult<PyFilter> {
        let vs = values.try_iter()?.map(|v| value(&v?)).collect::<PyResult<Vec<Value>>>()?;
        Ok(PyFilter(Filter::In(self.0.clone(), vs)))
    }

    /// A tags attribute has this tag.
    fn has(&self, tag: &str) -> PyFilter {
        PyFilter(hinny::attr(&self.0).has(tag))
    }

    /// A tags attribute has at least one of these.
    fn has_any(&self, tags: &Bound<'_, PyAny>) -> PyResult<PyFilter> {
        Ok(PyFilter(hinny::attr(&self.0).has_any(self::tags(tags)?)))
    }

    /// A tags attribute has every one of these.
    fn has_all(&self, tags: &Bound<'_, PyAny>) -> PyResult<PyFilter> {
        Ok(PyFilter(hinny::attr(&self.0).has_all(self::tags(tags)?)))
    }

    /// Starts with `p` (ignoring case). On tags: one of them does.
    fn prefix(&self, p: &str) -> PyFilter {
        PyFilter(Filter::Prefix(self.0.clone(), p.to_string()))
    }

    /// Contains the characters `p` (ignoring case). On tags: one of them does; for "has this
    /// tag", `has`.
    fn contains(&self, p: &str) -> PyFilter {
        PyFilter(Filter::Contains(self.0.clone(), p.to_string()))
    }

    /// The whole string matches a pattern: `*` is any run of characters, `?` any one, `\` takes
    /// the next as itself (`attr("path").glob("/home/*/taxes/*.pdf")`). On tags: one of them does.
    fn glob(&self, pattern: &str) -> PyFilter {
        PyFilter(Filter::Glob(self.0.clone(), pattern.to_string(), false))
    }

    /// `glob`, whatever the case.
    fn iglob(&self, pattern: &str) -> PyFilter {
        PyFilter(Filter::Glob(self.0.clone(), pattern.to_string(), true))
    }

    fn exists(&self) -> PyFilter {
        PyFilter(Filter::Exists(self.0.clone()))
    }

    fn __hash__(&self) -> PyResult<isize> {
        Err(PyTypeError::new_err("Attr isn't hashable (== builds a Filter)"))
    }

    fn __repr__(&self) -> String {
        format!("attr({:?})", self.0)
    }
}

#[pyfunction]
fn attr(name: &str) -> PyAttr {
    PyAttr(name.to_string())
}

/// A stable integer key for a string id (FNV-1a, 63 bits): `ns.upsert(hinny.key("doc-17"), …)`.
#[pyfunction]
fn key(id: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in id.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h & (u64::MAX >> 1)
}

// MARK: Results

/// A result: `key`, `score` (BM25, cosine, or fused), the parts it came from, `attrs`, the stored
/// `text` by field, and a `snippet` of the best-matching field with `highlights`: (start, end)
/// indices of the query's words in it.
#[pyclass(module = "hinny", frozen, get_all, name = "Hit")]
pub struct PyHit {
    key: u64,
    score: f32,
    text_score: Option<f32>,
    vector_score: Option<f32>,
    attrs: Py<PyDict>,
    text: Py<PyDict>,
    snippet: Option<String>,
    snippet_field: Option<String>,
    highlights: Vec<(usize, usize)>,
}

impl PyHit {
    fn from(py: Python<'_>, h: hinny::Hit) -> PyResult<PyHit> {
        let text = PyDict::new(py);
        for (field, t) in &h.text {
            text.set_item(field, t)?;
        }
        let (snippet_field, snippet, highlights) = match h.snippet {
            Some((field, s)) => {
                // Python indexes strings by code point; the library's ranges are bytes.
                let at = |byte: usize| s.text[..byte].chars().count();
                let highlights = s.highlights.iter().map(|r| (at(r.start), at(r.end))).collect();
                (Some(field), Some(s.text), highlights)
            }
            None => (None, None, Vec::new()),
        };
        Ok(PyHit { key: h.key, score: h.score, text_score: h.text_score, vector_score: h.vector_score, attrs: attrs_dict(py, &h.attributes)?.unbind(), text: text.unbind(), snippet, snippet_field, highlights })
    }
}

#[pymethods]
impl PyHit {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let attrs = self.attrs.bind(py).repr()?.to_string();
        let snippet = match &self.snippet {
            Some(s) => format!(", snippet={:?}", if s.chars().count() > 60 { format!("{}…", s.chars().take(60).collect::<String>()) } else { s.clone() }),
            None => String::new(),
        };
        Ok(format!("Hit(key={}, score={:.4}, text_score={:?}, vector_score={:?}, attrs={attrs}{snippet})", self.key, self.score, self.text_score, self.vector_score))
    }
}

// MARK: Databases and namespaces

/// A directory of namespaces.
#[pyclass(module = "hinny", frozen, name = "Database")]
pub struct PyDatabase(hinny::Database);

#[pymethods]
impl PyDatabase {
    #[new]
    fn new(path: std::path::PathBuf) -> PyResult<Self> {
        Ok(PyDatabase(hinny::Database::open(path).map_err(err)?))
    }

    /// Opens a namespace, or creates it with this schema. `text`: field names, or field → boost
    /// (BM25). `ints`, `strings`: attribute names. `vector`: embedding dimensions. `metric`: how
    /// vectors are compared, the one the embedding model was trained for: "cosine" (the
    /// default), "euclidean" or "dot"; fixed once set. A hit's vector score is higher for a
    /// nearer vector under each: the cosine, the squared distance negated, the dot product.
    /// Under "euclidean" and "dot" vectors are kept as given, not normalized.
    /// `store_text`: keep the text, for hits' `text` and snippets (default True). `language`
    /// ("english", "german", …): stem the text fields in it and, with `stopwords`, drop its
    /// stopwords; fixed once the namespace holds data. Default: words as written.
    /// `sparse`: names of sparse fields (terms with weights, from a learned sparse model).
    /// `floats`: names of attributes that are numbers with a fraction.
    /// `tags`: names of attributes that are several strings (labels, owners), filtered with
    /// `attr(name).has(tag)`.
    #[pyo3(signature = (name, text=None, ints=Vec::new(), strings=Vec::new(), vector=None, store_text=true, language=None, stopwords=true, sparse=Vec::new(), floats=Vec::new(), tags=Vec::new(), metric="cosine"))]
    #[allow(clippy::too_many_arguments)]
    fn namespace(&self, name: &str, text: Option<&Bound<'_, PyAny>>, ints: Vec<String>, strings: Vec<String>, vector: Option<usize>, store_text: bool, language: Option<&str>, stopwords: bool, sparse: Vec<String>, floats: Vec<String>, tags: Vec<String>, metric: &str) -> PyResult<PyNamespace> {
        let metric = hinny::Metric::from_name(metric).ok_or_else(|| PyValueError::new_err(format!("no vector metric {metric:?}: cosine, euclidean or dot")))?;
        if vector.is_none() && metric != hinny::Metric::Cosine {
            return Err(PyValueError::new_err("a metric is for a vector: give its dimensions as vector="));
        }
        let mut schema = hinny::Schema::new().store_text(store_text);
        if let Some(l) = language {
            let l = hinny::Language::from_name(l).ok_or_else(|| PyValueError::new_err(format!("no stemmer for {l:?}: arabic, danish, dutch, english, finnish, french, german, greek, hungarian, italian, norwegian, portuguese, romanian, russian, spanish, swedish, tamil or turkish")))?;
            schema = schema.analyzer(hinny::Analyzer::language(l).without_stopwords(stopwords));
        }
        if let Some(text) = text {
            if let Ok(d) = text.cast::<PyDict>() {
                for (k, v) in d.iter() {
                    schema = schema.text(&k.extract::<String>()?, v.extract::<f32>()?);
                }
            } else if let Ok(s) = text.extract::<String>() {
                schema = schema.text(&s, 1.0);
            } else {
                for f in text.try_iter()? {
                    schema = schema.text(&f?.extract::<String>()?, 1.0);
                }
            }
        }
        for n in &sparse {
            schema = schema.sparse(n);
        }
        for n in &ints {
            schema = schema.int(n);
        }
        for n in &floats {
            schema = schema.float(n);
        }
        for n in &strings {
            schema = schema.string(n);
        }
        for n in &tags {
            schema = schema.tags(n);
        }
        if let Some(d) = vector {
            schema = schema.vector_with(d, metric);
        }
        Ok(PyNamespace(self.0.namespace(name, schema).map_err(err)?))
    }

    /// Opens a namespace that exists, with the schema it has (`ns.schema` says what that is).
    /// Raises if there is none of that name.
    fn open(&self, name: &str) -> PyResult<PyNamespace> {
        Ok(PyNamespace(self.0.open_namespace(name).map_err(err)?))
    }

    fn namespaces(&self) -> PyResult<Vec<String>> {
        self.0.namespaces().map_err(err)
    }

    /// Deletes a namespace and everything in it. It must not be open: let go of every
    /// `Namespace` and `Snapshot` of it first (`del ns`), or this raises OSError and leaves it as
    /// it is. Raises OSError too if there is no namespace of that name, and ValueError for a name
    /// no namespace can have. A fork is a namespace of its own, deleted apart from the original.
    fn delete(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        py.detach(|| self.0.delete_namespace(name)).map_err(err)
    }
}

/// Documents sharing a schema. Thread-safe; buffered writes are searchable on return.
/// Call sync() for an explicit storage synchronization boundary.
#[pyclass(module = "hinny", frozen, name = "Namespace")]
pub struct PyNamespace(hinny::Namespace);

impl PyNamespace {
    /// Documents from an iterable of dicts with "key" and any of "text", "attrs", "vector",
    /// "version", "sparse".
    fn documents(&self, docs: &Bound<'_, PyAny>) -> PyResult<Vec<hinny::Document>> {
        let mut batch = Vec::new();
        for item in docs.try_iter()? {
            let item = item?;
            let d = item.cast::<PyDict>().map_err(|_| PyTypeError::new_err("each document is a dict"))?;
            let key: u64 = d.get_item("key")?.ok_or_else(|| PyValueError::new_err("a document needs a \"key\""))?.extract()?;
            let text = d.get_item("text")?;
            let attrs = d.get_item("attrs")?;
            let attrs = attrs.as_ref().map(|a| a.cast::<PyDict>()).transpose().map_err(|_| PyTypeError::new_err("attrs is a dict"))?;
            let vec = d.get_item("vector")?;
            let version: u64 = d.get_item("version")?.map(|v| v.extract()).transpose()?.unwrap_or(0);
            let sparse = d.get_item("sparse")?;
            batch.push(self.document(key, text.as_ref(), attrs, vec.as_ref(), version, sparse.as_ref())?);
        }
        Ok(batch)
    }

    fn document(&self, key: u64, text: Option<&Bound<'_, PyAny>>, attrs: Option<&Bound<'_, PyDict>>, vec: Option<&Bound<'_, PyAny>>, version: u64, sparse: Option<&Bound<'_, PyAny>>) -> PyResult<hinny::Document> {
        let mut d = hinny::Document::new(key).version(version);
        if let Some(sp) = sparse.filter(|s| !s.is_none()) {
            let dict = sp.cast::<PyDict>().map_err(|_| PyTypeError::new_err("sparse is a dict of field → {term: weight}"))?;
            for (field, terms) in dict.iter() {
                d = d.sparse(&field.extract::<String>()?, &weighted(&terms)?);
            }
        }
        if let Some(text) = text {
            if let Ok(t) = text.cast::<PyString>() {
                let t = lossy(t);
                // A plain string: the schema's only text field (sparse fields aside).
                let schema = self.0.schema();
                let fields: Vec<&String> = schema.text.iter().zip(&schema.analysis).filter(|(_, a)| !a.sparse).map(|((n, _), _)| n).collect();
                if fields.len() != 1 {
                    return Err(PyValueError::new_err("text as a string needs a schema with one text field; pass a dict of field → text"));
                }
                d = d.text(fields[0], t);
            } else {
                let dict = text.cast::<PyDict>().map_err(|_| PyTypeError::new_err("text is a string or a dict of field → text"))?;
                for (k, v) in dict.iter() {
                    let v = v.cast::<PyString>().map_err(|_| PyTypeError::new_err("a field's text is a string"))?;
                    d = d.text(&k.extract::<String>()?, lossy(v));
                }
            }
        }
        if let Some(attrs) = attrs {
            for (k, v) in attrs.iter() {
                let name: String = k.extract()?;
                d = match attribute(&v)? {
                    Value::Int(x) => d.int(&name, x),
                    Value::Str(s) => d.string(&name, s),
                    Value::Float(x) => d.float(&name, x),
                    Value::Tags(t) => d.tags(&name, t),
                };
            }
        }
        if let Some(v) = vec.filter(|v| !v.is_none()) {
            d = d.vector(vector(v)?);
        }
        Ok(d)
    }
}

#[pymethods]
impl PyNamespace {
    #[getter]
    fn name(&self) -> String {
        self.0.name().to_string()
    }

    /// The schema, as a dict.
    #[getter]
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.0.schema();
        let d = PyDict::new(py);
        let text = PyDict::new(py);
        for (n, b) in &s.text {
            text.set_item(n, b)?;
        }
        d.set_item("text", text)?;
        d.set_item("ints", s.attrs.iter().filter(|(_, k)| *k == hinny::Kind::Int).map(|(n, _)| n.clone()).collect::<Vec<_>>())?;
        d.set_item("floats", s.attrs.iter().filter(|(_, k)| *k == hinny::Kind::Float).map(|(n, _)| n.clone()).collect::<Vec<_>>())?;
        d.set_item("strings", s.attrs.iter().filter(|(_, k)| *k == hinny::Kind::Str).map(|(n, _)| n.clone()).collect::<Vec<_>>())?;
        d.set_item("tags", s.attrs.iter().filter(|(_, k)| *k == hinny::Kind::Tags).map(|(n, _)| n.clone()).collect::<Vec<_>>())?;
        d.set_item("vector", s.vector)?;
        d.set_item("metric", s.vector.map(|_| s.metric.name()))?;
        Ok(d)
    }

    /// Adds or replaces document `key`. `text`: a string (one text field) or field → text;
    /// `attrs`: name → int, float or str; `vector`: a NumPy array or a sequence of numbers.
    /// `sparse`: sparse field → {term: weight} (a learned sparse model's output).
    #[pyo3(signature = (key, text=None, attrs=None, vector=None, version=0, sparse=None))]
    #[allow(clippy::too_many_arguments)]
    fn upsert(&self, py: Python<'_>, key: u64, text: Option<&Bound<'_, PyAny>>, attrs: Option<&Bound<'_, PyDict>>, vector: Option<&Bound<'_, PyAny>>, version: u64, sparse: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let d = self.document(key, text, attrs, vector, version, sparse)?;
        py.detach(|| self.0.upsert(d)).map_err(err)
    }

    /// Many documents at once: an iterable of dicts with "key" and any of "text", "attrs",
    /// "vector", "version", "sparse". Converted first, then written as one batch (all or none of
    /// it, after a crash too) without the GIL.
    fn upsert_many(&self, py: Python<'_>, docs: &Bound<'_, PyAny>) -> PyResult<usize> {
        let batch = self.documents(docs)?;
        let n = batch.len();
        py.detach(|| self.0.upsert_many(batch)).map_err(err)?;
        Ok(n)
    }

    /// Upserts and deletes as one batch: all or none of it (after a crash too), and no query or
    /// snapshot sees part of it. `upsert`: documents as for `upsert_many`; `delete`: keys.
    #[pyo3(signature = (upsert=None, delete=None))]
    fn apply(&self, py: Python<'_>, upsert: Option<&Bound<'_, PyAny>>, delete: Option<&Bound<'_, PyAny>>) -> PyResult<usize> {
        let mut changes: Vec<hinny::Change> = match upsert {
            Some(u) => self.documents(u)?.into_iter().map(hinny::Change::Put).collect(),
            None => Vec::new(),
        };
        if let Some(keys) = delete {
            for k in keys.try_iter()? {
                changes.push(hinny::Change::Delete(k?.extract()?));
            }
        }
        let n = changes.len();
        py.detach(|| self.0.apply(changes)).map_err(err)?;
        Ok(n)
    }

    /// Deletes these keys as one batch.
    fn delete_many(&self, py: Python<'_>, keys: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.apply(py, None, Some(keys))
    }

    fn delete(&self, py: Python<'_>, key: u64) -> PyResult<()> {
        py.detach(|| self.0.delete(key)).map_err(err)
    }

    /// A document's attributes, or None.
    fn get<'py>(&self, py: Python<'py>, key: u64) -> PyResult<Option<Bound<'py, PyDict>>> {
        let a = py.detach(|| self.0.get(key));
        a.map(|a| attrs_dict(py, &a)).transpose()
    }

    /// A document's stored text: field → text.
    fn text<'py>(&self, py: Python<'py>, key: u64) -> PyResult<Option<Bound<'py, PyDict>>> {
        let t = py.detach(|| self.0.text(key));
        t.map(|t| {
            let d = PyDict::new(py);
            for (field, text) in t {
                d.set_item(field, text)?;
            }
            Ok(d)
        })
        .transpose()
    }

    /// key → version, for syncing only what changed.
    fn versions(&self, py: Python<'_>) -> HashMap<u64, u64> {
        py.detach(|| self.0.versions())
    }

    /// How many documents pass `filter` (every document, without one). Counted from the
    /// filter's columns, without reading the documents.
    #[pyo3(signature = (filter=None))]
    fn count(&self, py: Python<'_>, filter: Option<&PyFilter>) -> u64 {
        let filter = filter.map(|f| f.0.clone());
        py.detach(|| self.0.count(filter.as_ref()))
    }

    /// Deletes every document that passes `filter` now, and returns how many that was: a
    /// user's (`attr("user") == id`), the expired (`attr("at") < cutoff`). All of them or, on
    /// an error, none.
    fn delete_where(&self, py: Python<'_>, filter: &PyFilter) -> PyResult<u64> {
        let filter = filter.0.clone();
        py.detach(|| self.0.delete_where(&filter)).map_err(err)
    }

    /// Changes part of document `key` and leaves the rest: the text fields (`text`: field →
    /// text), attributes (`attrs`, as for `upsert`), sparse fields (`sparse`, as for `upsert`) and
    /// `vector` given are replaced, the names in `remove` (attributes or text fields) are taken
    /// away, `version` sets a new version, and the document's other fields and its version stay
    /// as they were. A sparse field's terms aren't kept as they were given, so in a namespace with
    /// sparse fields a patch gives each one's terms again in `sparse` or names it in `remove`;
    /// one that does neither raises ValueError. With `condition` (a `Filter`)
    /// the document is changed only if it passes, checked and written together. Returns False,
    /// and writes nothing, if there is no such document or it doesn't pass. The document is read
    /// and written again whole, so a patch costs what an upsert does, and it needs the text to
    /// have been stored.
    #[pyo3(signature = (key, text=None, attrs=None, vector=None, version=None, remove=Vec::new(), condition=None, sparse=None))]
    #[allow(clippy::too_many_arguments)]
    fn patch(&self, py: Python<'_>, key: u64, text: Option<&Bound<'_, PyAny>>, attrs: Option<&Bound<'_, PyDict>>, vector: Option<&Bound<'_, PyAny>>, version: Option<u64>, remove: Vec<String>, condition: Option<&PyFilter>, sparse: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
        let patch = patch_of(text, attrs, vector, version, &remove, sparse)?;
        let condition = condition.map(|c| c.0.clone());
        py.detach(|| match &condition {
            Some(c) => self.0.patch_if(key, &patch, c),
            None => self.0.patch(key, &patch),
        })
        .map_err(err)
    }

    /// `patch`, for every document that passes `filter` now (in batches of 10,000, each all or
    /// nothing); returns how many that was.
    #[pyo3(signature = (filter, text=None, attrs=None, vector=None, version=None, remove=Vec::new(), sparse=None))]
    #[allow(clippy::too_many_arguments)]
    fn patch_where(&self, py: Python<'_>, filter: &PyFilter, text: Option<&Bound<'_, PyAny>>, attrs: Option<&Bound<'_, PyDict>>, vector: Option<&Bound<'_, PyAny>>, version: Option<u64>, remove: Vec<String>, sparse: Option<&Bound<'_, PyAny>>) -> PyResult<u64> {
        let patch = patch_of(text, attrs, vector, version, &remove, sparse)?;
        let filter = filter.0.clone();
        py.detach(|| self.0.patch_where(&filter, &patch)).map_err(err)
    }

    /// `upsert`, unless there is a document with this key that doesn't pass `condition` (a
    /// `Filter`): written if there is none yet or if the one there passes (say a revision below
    /// the new one). Checked and written together, so two writers that both read revision 4 can't
    /// both write revision 5. Returns whether it was written.
    #[pyo3(signature = (key, condition, text=None, attrs=None, vector=None, version=0, sparse=None))]
    #[allow(clippy::too_many_arguments)]
    fn upsert_if(&self, py: Python<'_>, key: u64, condition: &PyFilter, text: Option<&Bound<'_, PyAny>>, attrs: Option<&Bound<'_, PyDict>>, vector: Option<&Bound<'_, PyAny>>, version: u64, sparse: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
        let d = self.document(key, text, attrs, vector, version, sparse)?;
        let condition = condition.0.clone();
        py.detach(|| self.0.upsert_if(d, &condition)).map_err(err)
    }

    /// `delete`, only if the document exists and passes `condition` (a `Filter`). Returns
    /// whether it was deleted.
    fn delete_if(&self, py: Python<'_>, key: u64, condition: &PyFilter) -> PyResult<bool> {
        let condition = condition.0.clone();
        py.detach(|| self.0.delete_if(key, &condition)).map_err(err)
    }

    /// How many documents have each value of the attribute `attr`, among those that pass
    /// `filter`: [(value, count), ...], the most frequent first, at most `limit` (0: every
    /// value). For attributes of few values (a kind, an owner, a status).
    #[pyo3(signature = (attr, filter=None, limit=0))]
    fn count_by<'py>(&self, py: Python<'py>, attr: String, filter: Option<&PyFilter>, limit: usize) -> PyResult<Vec<(Bound<'py, PyAny>, u64)>> {
        let filter = filter.map(|f| f.0.clone());
        let counts = py.detach(|| self.0.count_by(&attr, filter.as_ref(), limit));
        counts.iter().map(|(v, n)| Ok((to_py(py, v)?, *n))).collect()
    }

    /// Synchronizes prior writes to storage without building segments. Handle any I/O error.
    fn sync(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.0.sync()).map_err(err)
    }

    /// A namespace of its own, named `name` in the same database, holding what this one holds
    /// now: a sandbox to try something on real data. The two are apart from then on. Takes
    /// milliseconds whatever the size, and no more disk until they differ (its files are links
    /// to this one's). Raises FileExistsError if there is a namespace of that name.
    fn fork(&self, py: Python<'_>, name: &str) -> PyResult<PyNamespace> {
        py.detach(|| self.0.fork(name)).map(PyNamespace).map_err(err)
    }

    /// Synchronizes writes and writes memory as a segment (also happens on its own).
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.0.flush()).map_err(err)
    }

    fn optimize(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.0.optimize()).map_err(err)
    }

    /// Reads every file on disk against the checksums written with it. Returns what was read:
    /// `segments`, `unchecked_segments` (written before there were checksums; `optimize` rewrites
    /// them) and `text_files`. Raises OSError naming the first damaged file.
    fn verify<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let v = py.detach(|| self.0.verify()).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("segments", v.segments)?;
        d.set_item("unchecked_segments", v.unchecked_segments)?;
        d.set_item("text_files", v.blobs)?;
        Ok(d)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = py.detach(|| self.0.stats());
        let d = PyDict::new(py);
        d.set_item("documents", s.passages - s.deleted)?;
        d.set_item("segments", s.segments)?;
        d.set_item("in_memory", s.memory)?;
        d.set_item("merging", s.merging)?;
        d.set_item("flushing", s.flushing)?;
        d.set_item("memtable_bytes", s.memtable_bytes)?;
        d.set_item("log_bytes", s.log_bytes)?;
        d.set_item("pinned_segments", s.pinned_segments)?;
        d.set_item("pinned_bytes", s.pinned_bytes)?;
        d.set_item("vector_caches", s.vector_caches)?;
        d.set_item("vector_cache_bytes", s.vector_cache_bytes)?;
        d.set_item("kept_filters", s.kept_filters)?;
        d.set_item("kept_filter_bytes", s.kept_filter_bytes)?;
        d.set_item("vector_generations", s.vector_generations)?;
        d.set_item("vector_clusters", s.vector_clusters)?;
        Ok(d)
    }

    /// The writer's limits: the largest batch accepted, in bytes as logged (default 1 GiB; a larger
    /// one raises ValueError before anything is written), and the bytes in memory before a flush
    /// (default 256 MiB). Omitted ones stay as they are.
    #[pyo3(signature = (max_batch_bytes=None, flush_bytes=None))]
    fn set_limits(&self, py: Python<'_>, max_batch_bytes: Option<usize>, flush_bytes: Option<usize>) {
        py.detach(|| {
            let s = self.0.limits();
            self.0.set_limits(max_batch_bytes.unwrap_or(s.0), flush_bytes.unwrap_or(s.1));
        })
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        py.detach(|| self.0.count(None) as usize)
    }

    /// Searches: `text` (BM25), `vector` (by the namespace's metric), or both (hybrid, fused into one ranking);
    /// `filter` applies to both. Without text or a vector, lists documents by `filter`.
    /// `order_by`: an integer attribute, "-name" for largest first. `snippet`: snippet length in
    /// characters (0 for none). `with_text`: whether hits carry their whole stored text.
    /// Hybrid: `text_weight` (0–1, default 0.5) balances BM25 and cosine, each scaled to 0–1;
    /// `fusion="rrf"` uses reciprocal rank fusion instead. `nprobe`: vector clusters to probe
    /// per segment (0: each segment's calibrated budget, for recall@10 of 0.997).
    /// `text_candidates`, `vector_candidates`, `sparse_candidates`: how many each retriever of a
    /// hybrid query finds before fusion (0: the defaults; a larger limit raises them). `timeout`:
    /// seconds before the query gives up with TimeoutError (never partial results). `typing`:
    /// the text's last word is still being typed (a search box): it also matches the `typing`
    /// most common words that start with it (32 is plenty), so "tax ret" finds "tax return".
    #[pyo3(signature = (text=None, vector=None, filter=None, limit=10, order_by=None, all_words=false, nprobe=0, snippet=240, with_text=true, text_weight=0.5, fusion="convex", sparse=None, text_candidates=0, vector_candidates=0, sparse_candidates=0, timeout=None, typing=0))]
    #[allow(clippy::too_many_arguments)]
    fn query(&self, py: Python<'_>, text: Option<String>, vector: Option<&Bound<'_, PyAny>>, filter: Option<&PyFilter>, limit: usize, order_by: Option<String>, all_words: bool, nprobe: usize, snippet: usize, with_text: bool, text_weight: f32, fusion: &str, sparse: Option<&Bound<'_, PyAny>>, text_candidates: usize, vector_candidates: usize, sparse_candidates: usize, timeout: Option<f64>, typing: usize) -> PyResult<Vec<PyHit>> {
        run_query(py, &self.0, None, text, vector, filter, limit, order_by, all_words, nprobe, snippet, with_text, text_weight, fusion, sparse, text_candidates, vector_candidates, sparse_candidates, timeout, typing, false).map(|(h, _)| h)
    }

    /// As `query`, returning {"hits": [...], "fusion": ...}: for a hybrid query, what each retriever
    /// found before fusion (depth, found, scores: (lowest, highest)), how many both found, and how
    /// many were fused.
    #[pyo3(signature = (text=None, vector=None, filter=None, limit=10, order_by=None, all_words=false, nprobe=0, snippet=240, with_text=true, text_weight=0.5, fusion="convex", sparse=None, text_candidates=0, vector_candidates=0, sparse_candidates=0, timeout=None, typing=0))]
    #[allow(clippy::too_many_arguments)]
    fn explain(&self, py: Python<'_>, text: Option<String>, vector: Option<&Bound<'_, PyAny>>, filter: Option<&PyFilter>, limit: usize, order_by: Option<String>, all_words: bool, nprobe: usize, snippet: usize, with_text: bool, text_weight: f32, fusion: &str, sparse: Option<&Bound<'_, PyAny>>, text_candidates: usize, vector_candidates: usize, sparse_candidates: usize, timeout: Option<f64>, typing: usize) -> PyResult<Py<PyDict>> {
        run_query(py, &self.0, None, text, vector, filter, limit, order_by, all_words, nprobe, snippet, with_text, text_weight, fusion, sparse, text_candidates, vector_candidates, sparse_candidates, timeout, typing, true).and_then(|r| explained(py, r))
    }

    /// The namespace as of now: queries and reads on it see exactly the writes before it, however
    /// many come after, for as long as it's kept.
    fn snapshot(&self) -> PySnapshot {
        PySnapshot { ns: self.0.clone(), snap: self.0.snapshot() }
    }

    fn __repr__(&self) -> String {
        format!("Namespace({:?})", self.0.name())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_query(py: Python<'_>, ns: &hinny::Namespace, snap: Option<&hinny::Snapshot>, text: Option<String>, vector: Option<&Bound<'_, PyAny>>, filter: Option<&PyFilter>, limit: usize, order_by: Option<String>, all_words: bool, nprobe: usize, snippet: usize, with_text: bool, text_weight: f32, fusion: &str, sparse: Option<&Bound<'_, PyAny>>, text_candidates: usize, vector_candidates: usize, sparse_candidates: usize, timeout: Option<f64>, typing: usize, explain: bool) -> PyResult<(Vec<PyHit>, Option<hinny::FusionStats>)> {
    let timeout = match timeout {
        Some(t) if !(t.is_finite() && t >= 0.0) => return Err(PyValueError::new_err("timeout is a number of seconds, 0 or more")),
        t => t.map(std::time::Duration::from_secs_f64),
    };
    let mut sparse_terms: Vec<(String, Vec<(String, f32)>)> = Vec::new();
    if let Some(sp) = sparse.filter(|s| !s.is_none()) {
        let dict = sp.cast::<PyDict>().map_err(|_| PyTypeError::new_err("sparse is a dict of field → {term: weight}"))?;
        for (field, terms) in dict.iter() {
            sparse_terms.push((field.extract()?, weighted(&terms)?));
        }
    }
    let fusion = match fusion {
        "convex" => hinny::Fusion::Convex { text_weight },
        "rrf" => hinny::Fusion::Rrf { k: 60.0 },
        other => return Err(PyValueError::new_err(format!("fusion is \"convex\" or \"rrf\", not {other:?}"))),
    };
    let v = vector.filter(|v| !v.is_none()).map(self::vector).transpose()?;
    let filter = filter.map(|f| f.0.clone());
    let hits = py
        .detach(|| {
            let base = match snap {
                Some(s) => s.query(),
                None => ns.query(),
            };
            let mut base = base;
            for (field, terms) in &sparse_terms {
                base = base.sparse(field, terms.iter().map(|(t, w)| (t.as_str(), *w)));
            }
            let mut q = base.limit(limit).nprobe(nprobe).snippet(snippet).with_text(with_text).fusion(fusion).candidates(text_candidates, vector_candidates).sparse_candidates(sparse_candidates);
            if let Some(t) = text {
                q = q.text(t);
            }
            if let Some(v) = v {
                q = q.vector(v);
            }
            if let Some(f) = filter {
                q = q.filter(f);
            }
            if let Some(o) = &order_by {
                let (name, desc) = match o.strip_prefix('-') {
                    Some(n) => (n, true),
                    None => (o.as_str(), false),
                };
                q = q.order_by(name, desc);
            }
            if all_words {
                q = q.all_words();
            }
            if typing > 0 {
                q = q.typing(typing);
            }
            if let Some(t) = timeout {
                q = q.deadline(t);
            }
            if explain { q.explain() } else { q.run().map(|h| (h, None)) }
        })
        .map_err(err)?;
    let (hits, stats) = hits;
    Ok((hits.into_iter().map(|h| PyHit::from(py, h)).collect::<PyResult<Vec<_>>>()?, stats))
}

/// `explain`'s result: the hits, and how a hybrid query fused them (None otherwise).
fn explained(py: Python<'_>, (hits, stats): (Vec<PyHit>, Option<hinny::FusionStats>)) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("hits", hits)?;
    match stats {
        Some(s) => {
            let fusion = PyDict::new(py);
            for (name, r) in [("lexical", s.lexical), ("vector", s.vector)] {
                let d = PyDict::new(py);
                d.set_item("depth", r.depth)?;
                d.set_item("found", r.found)?;
                d.set_item("scores", r.scores)?;
                fusion.set_item(name, d)?;
            }
            fusion.set_item("both", s.both)?;
            fusion.set_item("fused", s.fused)?;
            out.set_item("fusion", fusion)?;
        }
        None => out.set_item("fusion", py.None())?,
    }
    Ok(out.unbind())
}


/// The namespace as of one moment (`Namespace.snapshot()`).
#[pyclass(module = "hinny", frozen, name = "Snapshot")]
pub struct PySnapshot {
    ns: hinny::Namespace,
    snap: hinny::Snapshot,
}

#[pymethods]
impl PySnapshot {
    /// As `Namespace.query`, on the snapshot.
    #[pyo3(signature = (text=None, vector=None, filter=None, limit=10, order_by=None, all_words=false, nprobe=0, snippet=240, with_text=true, text_weight=0.5, fusion="convex", sparse=None, text_candidates=0, vector_candidates=0, sparse_candidates=0, timeout=None, typing=0))]
    #[allow(clippy::too_many_arguments)]
    fn query(&self, py: Python<'_>, text: Option<String>, vector: Option<&Bound<'_, PyAny>>, filter: Option<&PyFilter>, limit: usize, order_by: Option<String>, all_words: bool, nprobe: usize, snippet: usize, with_text: bool, text_weight: f32, fusion: &str, sparse: Option<&Bound<'_, PyAny>>, text_candidates: usize, vector_candidates: usize, sparse_candidates: usize, timeout: Option<f64>, typing: usize) -> PyResult<Vec<PyHit>> {
        run_query(py, &self.ns, Some(&self.snap), text, vector, filter, limit, order_by, all_words, nprobe, snippet, with_text, text_weight, fusion, sparse, text_candidates, vector_candidates, sparse_candidates, timeout, typing, false).map(|(h, _)| h)
    }

    /// As `query`, returning {"hits": [...], "fusion": ...}: for a hybrid query, what each retriever
    /// found before fusion (depth, found, scores: (lowest, highest)), how many both found, and how
    /// many were fused.
    #[pyo3(signature = (text=None, vector=None, filter=None, limit=10, order_by=None, all_words=false, nprobe=0, snippet=240, with_text=true, text_weight=0.5, fusion="convex", sparse=None, text_candidates=0, vector_candidates=0, sparse_candidates=0, timeout=None, typing=0))]
    #[allow(clippy::too_many_arguments)]
    fn explain(&self, py: Python<'_>, text: Option<String>, vector: Option<&Bound<'_, PyAny>>, filter: Option<&PyFilter>, limit: usize, order_by: Option<String>, all_words: bool, nprobe: usize, snippet: usize, with_text: bool, text_weight: f32, fusion: &str, sparse: Option<&Bound<'_, PyAny>>, text_candidates: usize, vector_candidates: usize, sparse_candidates: usize, timeout: Option<f64>, typing: usize) -> PyResult<Py<PyDict>> {
        run_query(py, &self.ns, Some(&self.snap), text, vector, filter, limit, order_by, all_words, nprobe, snippet, with_text, text_weight, fusion, sparse, text_candidates, vector_candidates, sparse_candidates, timeout, typing, true).and_then(|r| explained(py, r))
    }

    fn get<'py>(&self, py: Python<'py>, key: u64) -> PyResult<Option<Bound<'py, PyDict>>> {
        let a = py.detach(|| self.snap.get(key));
        a.map(|a| attrs_dict(py, &a)).transpose()
    }

    fn text<'py>(&self, py: Python<'py>, key: u64) -> PyResult<Option<Bound<'py, PyDict>>> {
        let t = py.detach(|| self.snap.text(key));
        t.map(|t| {
            let d = PyDict::new(py);
            for (field, text) in t {
                d.set_item(field, text)?;
            }
            Ok(d)
        })
        .transpose()
    }

    fn versions(&self, py: Python<'_>) -> HashMap<u64, u64> {
        py.detach(|| self.snap.versions())
    }

    /// As `Namespace.count`, on the snapshot.
    #[pyo3(signature = (filter=None))]
    fn count(&self, py: Python<'_>, filter: Option<&PyFilter>) -> u64 {
        let filter = filter.map(|f| f.0.clone());
        py.detach(|| self.snap.count(filter.as_ref()))
    }

    /// As `Namespace.count_by`, on the snapshot.
    #[pyo3(signature = (attr, filter=None, limit=0))]
    fn count_by<'py>(&self, py: Python<'py>, attr: String, filter: Option<&PyFilter>, limit: usize) -> PyResult<Vec<(Bound<'py, PyAny>, u64)>> {
        let filter = filter.map(|f| f.0.clone());
        let counts = py.detach(|| self.snap.count_by(&attr, filter.as_ref(), limit));
        counts.iter().map(|(v, n)| Ok((to_py(py, v)?, *n))).collect()
    }

    /// The sequence number of the newest batch it includes.
    #[getter]
    fn seq(&self) -> u64 {
        self.snap.seq()
    }

    fn __repr__(&self) -> String {
        format!("Snapshot({:?}, seq={})", self.ns.name(), self.snap.seq())
    }
}

#[pymodule]
fn _hinny(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatabase>()?;
    m.add_class::<PyNamespace>()?;
    m.add_class::<PySnapshot>()?;
    m.add_class::<PyHit>()?;
    m.add_class::<PyFilter>()?;
    m.add_class::<PyAttr>()?;
    m.add_function(wrap_pyfunction!(attr, m)?)?;
    m.add_function(wrap_pyfunction!(key, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
