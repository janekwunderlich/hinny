//! The library interface: databases, namespaces with a schema, documents, and one query builder
//! for full text, vectors, or both (hybrid).
//!
//! ```no_run
//! use hinny::{Database, Document, Schema, attr};
//!
//! let db = Database::open("data")?;
//! let notes = db.namespace("notes", Schema::new().text("title", 2.0).text("body", 1.0).int("year").string("kind").vector(4))?;
//! notes.upsert(Document::new(1).text("title", "Lease").text("body", "The apartment lease for 2025").int("year", 2025).string("kind", "pdf").vector(vec![0.1, 0.9, 0.0, 0.2]))?;
//!
//! let hits = notes.query().text("apartment lease").vector(vec![0.1, 0.8, 0.1, 0.2]).filter(attr("year").gte(2025)).limit(10).run()?;
//! let newest = notes.query().filter(attr("kind").eq("pdf")).order_by("year", true).limit(5).run()?;
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! Hybrid queries run two retrievers with the same filter, BM25 over the text fields and nearest
//! neighbours over the vectors, and fuse them (`Fusion`). By default each retriever's scores are
//! scaled to 0–1 over what it found (min–max) and combined, half each: a convex combination,
//! which beat reciprocal rank fusion on four BEIR datasets and on a benchmark of questions about
//! a real mailbox (as Bruch et al., "An Analysis of Fusion Functions for Hybrid Retrieval", 2023,
//! found). `Query::text_weight` moves the balance: lower for a strong embedding model on its own
//! domain, higher where exact words and names matter.

use std::collections::HashMap;
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::analysis::Analyzer;
use crate::attrs::{Filter, Kind, Value};
use crate::index::{Doc, Found, Index, Op, Reader, Request, View};
use crate::segment::MAX_FIELDS;
use crate::snippet::{Snippet, snippet_analyzed};
use crate::vector::Metric;

/// Stored text is the index's (`Index::store_text`, in blob files). Before 0.2 it was string
/// attributes with this prefix, still read; so attribute names can't start with `_fugu.`.
const STORED: &str = crate::segment::LEGACY_TEXT;

// MARK: Schemas

/// What a namespace's documents hold: text fields (searched with BM25, each with a boost),
/// typed attributes (filtered and returned), and optionally a vector of fixed dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct Schema {
    pub text: Vec<(String, f32)>,
    /// How each text field's words become terms (in `text`'s order): stemmed in a language,
    /// stopwords dropped, or as written (`Analyzer::PLAIN`, the default). Fixed once a field holds
    /// data.
    pub analysis: Vec<Analyzer>,
    pub attrs: Vec<(String, Kind)>,
    pub vector: Option<usize>,
    /// How vectors are compared (cosine unless `vector_with` says otherwise). Fixed once the
    /// namespace has a vector.
    pub metric: Metric,
    /// Whether text fields are stored, to come back with hits and as snippets (on by default).
    pub store_text: bool,
    /// The analyzer `text` gives the fields declared after `analyzer` was called.
    next_analyzer: Analyzer,
}

impl Default for Schema {
    fn default() -> Schema {
        Schema { text: Vec::new(), analysis: Vec::new(), attrs: Vec::new(), vector: None, metric: Metric::Cosine, store_text: true, next_analyzer: Analyzer::PLAIN }
    }
}

impl Schema {
    pub fn new() -> Schema {
        Schema::default()
    }

    /// Whether to store text fields (default: yes). Without, hits have no text and no snippet,
    /// and the index is smaller: only its postings hold the text.
    pub fn store_text(mut self, store: bool) -> Schema {
        self.store_text = store;
        self
    }

    /// A text field, searched with BM25; `boost` weights its matches (1.0 is neutral). It's
    /// analyzed as `analyzer` says (by default: words as written).
    pub fn text(self, name: &str, boost: f32) -> Schema {
        let a = self.next_analyzer;
        self.text_with(name, boost, a)
    }

    /// A text field with its own analyzer: `Analyzer::english()` stems English and drops its
    /// stopwords, for instance.
    pub fn text_with(mut self, name: &str, boost: f32, analyzer: Analyzer) -> Schema {
        self.text.push((name.to_string(), boost));
        self.analysis.push(analyzer);
        self
    }

    /// A sparse field: terms with weights (from a learned sparse model such as SPLADE or an
    /// inference-free document encoder), searched with `Query::sparse` by dot product, on the
    /// same inverted index and pruning as BM25.
    pub fn sparse(self, name: &str) -> Schema {
        self.text_with(name, 1.0, Analyzer::SPARSE)
    }

    /// The analyzer for the text fields declared after this: `Schema::new().analyzer(Analyzer::english()).text("title", 2.0)…`.
    pub fn analyzer(mut self, analyzer: Analyzer) -> Schema {
        self.next_analyzer = analyzer;
        self
    }

    fn analyzers(&self) -> Vec<Analyzer> {
        (0..self.text.len()).map(|i| self.analysis.get(i).copied().unwrap_or_default()).collect()
    }

    pub fn int(mut self, name: &str) -> Schema {
        self.attrs.push((name.to_string(), Kind::Int));
        self
    }

    pub fn string(mut self, name: &str) -> Schema {
        self.attrs.push((name.to_string(), Kind::Str));
        self
    }

    /// An attribute that is a number with a fraction: a score, a price, a coordinate.
    pub fn float(mut self, name: &str) -> Schema {
        self.attrs.push((name.to_string(), Kind::Float));
        self
    }

    /// An attribute that is several strings: labels, owners, the groups that may read it. A
    /// set (each once, in no order of its own), filtered with `attr(name).has(tag)`.
    pub fn tags(mut self, name: &str) -> Schema {
        self.attrs.push((name.to_string(), Kind::Tags));
        self
    }

    /// Embeddings of `dims` dimensions (cosine similarity).
    pub fn vector(self, dims: usize) -> Schema {
        self.vector_with(dims, Metric::Cosine)
    }

    /// Embeddings of `dims` dimensions, compared by `metric`: the one the model that made them
    /// was trained for. `Metric::Cosine` normalizes vectors and queries; `Metric::Euclidean`
    /// and `Metric::Dot` take them as they come (kept in half precision: 11 bits of each
    /// component, which can be at most 65,504 in size). A hit's score is higher for a nearer
    /// vector under each: the cosine, the squared distance negated, the dot product.
    pub fn vector_with(mut self, dims: usize, metric: Metric) -> Schema {
        self.vector = Some(dims);
        self.metric = metric;
        self
    }

    fn field(&self, name: &str) -> Option<u8> {
        self.text.iter().position(|(n, _)| n == name).map(|i| i as u8)
    }

    fn kind(&self, name: &str) -> Option<Kind> {
        self.attrs.iter().find(|(n, _)| n == name).map(|(_, k)| *k)
    }

    /// The schema in a line, for an error to show: `text title (boost 2), body; int year;
    /// string kind; vector 384`.
    fn describe(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if !self.text.is_empty() {
            let fields: Vec<String> = self.text.iter().map(|(n, b)| if *b == 1.0 { n.clone() } else { format!("{n} (boost {b})") }).collect();
            parts.push(format!("text {}", fields.join(", ")));
        }
        for (kind, name) in [(Kind::Int, "int"), (Kind::Float, "float"), (Kind::Str, "string"), (Kind::Tags, "tags")] {
            let names: Vec<&str> = self.attrs.iter().filter(|(_, k)| *k == kind).map(|(n, _)| n.as_str()).collect();
            if !names.is_empty() {
                parts.push(format!("{name} {}", names.join(", ")));
            }
        }
        if let Some(d) = self.vector {
            parts.push(if self.metric == Metric::Cosine { format!("vector {d}") } else { format!("vector {d} ({})", self.metric.name()) });
        }
        if !self.store_text {
            parts.push("text not stored".to_string());
        }
        if parts.is_empty() { "nothing".to_string() } else { parts.join("; ") }
    }

    fn encode(&self) -> String {
        // 3 only with what 2 hasn't (a float attribute), 4 with what 3 hasn't (tags) and 5 with
        // what 4 hasn't (vectors compared by another metric than cosine), so a schema an earlier
        // release could read stays one it can.
        let has = |kind: Kind| self.attrs.iter().any(|(_, k)| *k == kind);
        let metric = self.vector.map(|_| self.metric).filter(|m| *m != Metric::Cosine);
        let version = if metric.is_some() { 5 } else if has(Kind::Tags) { 4 } else if has(Kind::Float) { 3 } else { 2 };
        let mut s = format!("fugudb schema {version}\n");
        for ((n, b), a) in self.text.iter().zip(self.analyzers()) {
            s.push_str(&format!("text {b} {} {n}\n", a.code()));
        }
        for (n, k) in &self.attrs {
            let kind = match k {
                Kind::Int => "int",
                Kind::Str => "string",
                Kind::Float => "float",
                Kind::Tags => "tags",
            };
            s.push_str(&format!("{kind} {n}\n"));
        }
        if let Some(d) = self.vector {
            s.push_str(&match metric {
                Some(m) => format!("vector {d} {}\n", m.name()),
                None => format!("vector {d}\n"),
            });
        }
        if !self.store_text {
            s.push_str("store_text false\n");
        }
        s
    }

    fn decode(text: &str) -> Option<Schema> {
        let mut s = Schema::new();
        let mut lines = text.lines();
        // 1: text fields without an analyzer (plain). 2: "text BOOST ANALYZER NAME". 3: float
        // attributes. 4: tags. 5: "vector DIMS METRIC".
        let version = lines.next()?.strip_prefix("fugudb schema ")?.parse::<u32>().ok()?;
        if !(1..=5).contains(&version) {
            return None;
        }
        for line in lines {
            // Attribute names, like text names, can contain spaces. Only the first separator
            // belongs to the encoding; control characters are rejected when opening a schema.
            if let Some((kind, name)) = line.split_once(' ') {
                let kind = match kind {
                    "int" => Some(Kind::Int),
                    "string" => Some(Kind::Str),
                    "float" if version >= 3 => Some(Kind::Float),
                    "tags" if version >= 4 => Some(Kind::Tags),
                    _ => None,
                };
                if let Some(kind) = kind {
                    s.attrs.push((name.to_string(), kind));
                    continue;
                }
            }
            if version >= 2 && line.starts_with("text ") {
                let mut it = line.splitn(4, ' ');
                let (_, b, a, n) = (it.next()?, it.next()?, it.next()?, it.next()?);
                s = s.text_with(n, b.parse().ok()?, Analyzer::from_code(a)?);
                continue;
            }
            let mut it = line.splitn(3, ' ');
            match (it.next()?, it.next(), it.next()) {
                ("text", Some(b), Some(n)) => s = s.text_with(n, b.parse().ok()?, Analyzer::PLAIN),
                ("int", Some(n), None) => s.attrs.push((n.to_string(), Kind::Int)),
                ("string", Some(n), None) => s.attrs.push((n.to_string(), Kind::Str)),
                ("vector", Some(d), None) => s.vector = Some(d.parse().ok()?),
                ("vector", Some(d), Some(m)) if version >= 5 => s = s.vector_with(d.parse().ok()?, Metric::from_name(m)?),
                ("store_text", Some(b), None) => s.store_text = b.parse().ok()?,
                _ => return None,
            }
        }
        Some(s)
    }

    /// Whether data written under `self` can be read under `next`: the same text fields in the
    /// same order (more may follow), the same vector dimensions compared the same way,
    /// attributes of unchanged kinds.
    fn evolves_to(&self, next: &Schema) -> bool {
        // (Analyzers are the index's to check: they can change only while it's empty.)
        self.text.iter().zip(&next.text).all(|(a, b)| a.0 == b.0)
            && next.text.len() >= self.text.len()
            && (self.vector.is_none() || self.vector == next.vector && self.metric == next.metric)
            && self.attrs.iter().all(|(n, k)| next.kind(n).is_none_or(|nk| nk == *k))
    }
}

// MARK: Databases and namespaces

/// A directory of namespaces.
pub struct Database {
    dir: PathBuf,
}

impl Database {
    pub fn open(dir: impl AsRef<Path>) -> Result<Database> {
        crate::index::create_dir_all_synced(dir.as_ref())?;
        Ok(Database { dir: dir.as_ref().to_path_buf() })
    }

    /// Opens a namespace (or creates it with `schema`). A namespace's schema can grow: more text
    /// fields at the end, more attributes, a vector if it had none.
    pub fn namespace(&self, name: &str, schema: Schema) -> Result<Namespace> {
        if !valid_name(name) {
            return Err(Error::new(ErrorKind::InvalidInput, "namespace names are letters, digits, - and _"));
        }
        if schema.text.len() > MAX_FIELDS {
            return Err(Error::new(ErrorKind::InvalidInput, format!("at most {MAX_FIELDS} text fields")));
        }
        let mut names: Vec<&str> = schema.text.iter().map(|(n, _)| n.as_str()).chain(schema.attrs.iter().map(|(n, _)| n.as_str())).collect();
        if names.iter().any(|n| n.is_empty() || n.chars().any(char::is_control)) {
            return Err(Error::new(ErrorKind::InvalidInput, "field names must be nonempty and contain no control characters"));
        }
        names.sort_unstable();
        if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
            return Err(Error::new(ErrorKind::InvalidInput, format!("{:?} is in the schema twice", w[0])));
        }
        // Pruning bounds a document's score by adding its terms' best contributions, which holds
        // only for weights that can't subtract.
        if let Some((n, b)) = schema.text.iter().find(|(_, b)| !b.is_finite() || *b < 0.0) {
            return Err(Error::new(ErrorKind::InvalidInput, format!("field {n:?} has a boost of {b}; boosts are finite and at least 0")));
        }
        if let Some((n, _)) = schema.attrs.iter().find(|(n, _)| n.starts_with("_fugu.")) {
            return Err(Error::new(ErrorKind::InvalidInput, format!("attribute names can't start with \"_fugu.\" ({n:?})")));
        }
        let dir = self.dir.join(name);
        let file = dir.join("schema");
        let validate = || -> Result<()> {
            match std::fs::read_to_string(&file) {
                Ok(old) => {
                    let old = Schema::decode(&old).ok_or_else(|| Error::new(ErrorKind::InvalidData, "unreadable schema"))?;
                    if !old.evolves_to(&schema) {
                        return Err(Error::new(ErrorKind::InvalidInput, format!("namespace {name:?} has another schema: it has {}; asked for {}", old.describe(), schema.describe())));
                    }
                }
                Err(e) if e.kind() == ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
            Ok(())
        };
        validate()?;
        // Revalidate under the lock before replay: a rejected schema must not leave a partially
        // opened Index whose Drop checkpoints data using the caller's obsolete field count.
        let mut index = Index::open_validated(&dir, schema.text.len().max(1), validate)?;
        index.store_text = schema.store_text;
        index.set_analyzers(schema.analyzers())?;
        // The index keeps its own record of how its vectors are compared (it replays its log
        // by it), and refuses another once it holds vectors.
        if schema.vector.is_some() {
            index.set_metric(schema.metric)?;
        }
        crate::index::write_atomic(&file, schema.encode().as_bytes())?;
        Ok(Namespace { name: name.to_string(), schema, home: self.dir.clone(), index: Arc::new(Shared::new(index)) })
    }

    /// Opens a namespace that exists, with the schema it has (`Namespace::schema` says what
    /// that is): for a tool that reads or maintains a database it didn't define. `NotFound` if
    /// there is none of that name.
    pub fn open_namespace(&self, name: &str) -> Result<Namespace> {
        let file = self.dir.join(name).join("schema");
        let stored = match std::fs::read_to_string(&file) {
            Ok(text) => text,
            Err(e) if e.kind() == ErrorKind::NotFound => return Err(Error::new(ErrorKind::NotFound, format!("no namespace {name:?} in {}", self.dir.display()))),
            Err(e) => return Err(e),
        };
        let schema = Schema::decode(&stored).ok_or_else(|| Error::new(ErrorKind::InvalidData, "unreadable schema"))?;
        self.namespace(name, schema)
    }

    /// Opens a namespace that exists for reading only: beside a writer in another process, or
    /// over files copied from one (a replica). It takes no lock and changes no file; writes on
    /// it fail with `PermissionDenied`. It shows the namespace as of when it was opened until
    /// `Namespace::refresh` reads it again.
    pub fn open_read_only(&self, name: &str) -> Result<Namespace> {
        if !valid_name(name) {
            return Err(Error::new(ErrorKind::InvalidInput, "namespace names are letters, digits, - and _"));
        }
        let dir = self.dir.join(name);
        let stored = match std::fs::read_to_string(dir.join("schema")) {
            Ok(text) => text,
            Err(e) if e.kind() == ErrorKind::NotFound => return Err(Error::new(ErrorKind::NotFound, format!("no namespace {name:?} in {}", self.dir.display()))),
            Err(e) => return Err(e),
        };
        let schema = Schema::decode(&stored).ok_or_else(|| Error::new(ErrorKind::InvalidData, "unreadable schema"))?;
        let seen = Index::on_disk(&dir);
        let index = Index::open_read_only(&dir, schema.text.len().max(1))?;
        // Vectors stored for one metric aren't another's: a schema beside an index that
        // compares its vectors otherwise is files of two namespaces, and is refused.
        if schema.vector.is_some() && index.view().dims() != 0 && index.metric() != schema.metric {
            return Err(Error::new(ErrorKind::InvalidData, format!("namespace {name:?}'s schema compares vectors by {}, its index by {}", schema.metric.name(), index.metric().name())));
        }
        let shared = Shared::new(index);
        shared.seen.store(seen, std::sync::atomic::Ordering::Relaxed);
        Ok(Namespace { name: name.to_string(), schema, home: self.dir.clone(), index: Arc::new(shared) })
    }

    /// Deletes a namespace and everything in it. It must not be open: close (drop) every
    /// handle and snapshot of it first, or this fails with `WouldBlock`. `NotFound` if there
    /// is none of that name. Nothing is written first, as dropping a handle would.
    pub fn delete_namespace(&self, name: &str) -> Result<()> {
        if !valid_name(name) {
            return Err(Error::new(ErrorKind::InvalidInput, "namespace names are letters, digits, - and _"));
        }
        let dir = self.dir.join(name);
        if !dir.join("schema").exists() {
            return Err(Error::new(ErrorKind::NotFound, format!("no namespace {name:?} in {}", self.dir.display())));
        }
        Index::remove(&dir, &self.dir.join(format!(".deleted-{name}-{}", std::process::id())))
    }

    /// The namespaces in this database.
    pub fn namespaces(&self) -> Result<Vec<String>> {
        // (Not what a fork is being built in, or was when a process died: no namespace's name.)
        let mut out: Vec<String> = std::fs::read_dir(&self.dir)?.flatten().filter(|e| e.path().join("schema").exists()).map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| valid_name(n)).collect();
        out.sort();
        Ok(out)
    }
}

/// Whether `name` can be a namespace's: letters, digits, `-` and `_` (ASCII), at least one.
/// A namespace is a directory of that name, so nothing else is allowed in it.
pub fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// The index: writes take turns on a mutex; queries take a view (`Reader`), never the mutex, so
/// neither waits for the other. (Before views, a reader–writer lock with a turnstile against
/// writer starvation; queries back to back could still keep a write waiting a whole query.)
struct Shared {
    writer: std::sync::Mutex<Index>,
    reader: Reader,
    /// A read-only namespace: what was on disk (`Index::on_disk`) when it was last read.
    seen: std::sync::atomic::AtomicU64,
}

impl Shared {
    fn new(index: Index) -> Shared {
        let reader = index.reader();
        Shared { writer: std::sync::Mutex::new(index), reader, seen: 0.into() }
    }

    fn write(&self) -> std::sync::MutexGuard<'_, Index> {
        self.writer.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn view(&self) -> Arc<View> {
        self.reader.view()
    }
}

/// A change in a batch (`Namespace::apply`).
pub enum Change {
    Put(Document),
    Delete(u64),
}

/// The namespace as of one moment: queries and reads on it see exactly the writes before it was
/// taken, however many come after, for as long as it's kept.
pub struct Snapshot {
    ns: Namespace,
    view: Arc<View>,
}

impl Snapshot {
    pub fn query(&self) -> Query<'_> {
        Query { view: Some(self.view.clone()), ..self.ns.query() }
    }

    pub fn get(&self, key: u64) -> Option<Vec<(String, Value)>> {
        Some(without_legacy_text(self.view.get(key)?))
    }

    pub fn text(&self, key: u64) -> Option<Vec<(String, String)>> {
        self.view.get(key)?;
        Some(named(&self.ns.schema, self.view.text(key)))
    }

    pub fn versions(&self) -> HashMap<u64, u64> {
        self.view.versions()
    }

    /// How many documents pass `filter` (see [`Namespace::count`]).
    pub fn count(&self, filter: Option<&Filter>) -> u64 {
        self.view.count(filter)
    }

    /// How many documents have each value of an attribute (see [`Namespace::count_by`]).
    pub fn count_by(&self, attr: &str, filter: Option<&Filter>, limit: usize) -> Vec<(Value, u64)> {
        self.view.count_by(attr, filter, limit)
    }

    /// The sequence number of the newest batch it includes.
    pub fn seq(&self) -> u64 {
        self.view.seq()
    }
}

/// A collection of documents sharing a schema. Cheap to clone; safe to share between threads.
#[derive(Clone)]
pub struct Namespace {
    name: String,
    schema: Schema,
    /// The database's directory.
    home: PathBuf,
    index: Arc<Shared>,
}

/// What `Namespace::files` lists.
#[derive(Clone, Debug)]
pub struct Files {
    /// The namespace's identity (`Namespace::id`): a copy of another one is not a newer state
    /// of the same.
    pub id: String,
    /// The contents of the namespace's `schema` file.
    pub schema: String,
    /// The contents of its `manifest` file.
    pub manifest: Vec<u8>,
    /// The files the manifest names, with their sizes in bytes.
    pub files: Vec<(String, u64)>,
}

/// What to change of a document that is there (`Namespace::patch`): the text fields,
/// attributes and vector named here are replaced, those in `remove` taken away, and everything
/// else stays as it is.
#[derive(Clone, Debug, Default)]
pub struct Patch {
    version: Option<u64>,
    texts: Vec<(String, String)>,
    attrs: Vec<(String, Value)>,
    removed: Vec<String>,
    vector: Option<Vec<f32>>,
}

impl Patch {
    pub fn new() -> Patch {
        Patch::default()
    }

    pub fn version(mut self, version: u64) -> Patch {
        self.version = Some(version);
        self
    }

    pub fn text(mut self, field: &str, text: impl Into<String>) -> Patch {
        self.texts.push((field.to_string(), text.into()));
        self
    }

    /// A sparse field's terms and weights, as `Document::sparse` takes them. A document in a
    /// namespace with sparse fields can only be patched with its sparse terms given again:
    /// they aren't kept as they were given.
    pub fn sparse<S: AsRef<str>>(self, field: &str, terms: &[(S, f32)]) -> Patch {
        self.text(field, crate::analysis::encode_sparse(terms))
    }

    pub fn int(mut self, name: &str, value: i64) -> Patch {
        self.attrs.push((name.to_string(), Value::Int(value)));
        self
    }

    pub fn string(mut self, name: &str, value: impl Into<String>) -> Patch {
        self.attrs.push((name.to_string(), Value::Str(value.into())));
        self
    }

    pub fn float(mut self, name: &str, value: f64) -> Patch {
        self.attrs.push((name.to_string(), Value::Float(value)));
        self
    }

    pub fn tags<S: Into<String>>(mut self, name: &str, tags: impl IntoIterator<Item = S>) -> Patch {
        self.attrs.push((name.to_string(), Value::Tags(tags.into_iter().map(Into::into).collect())));
        self
    }

    pub fn vector(mut self, vector: Vec<f32>) -> Patch {
        self.vector = Some(vector);
        self
    }

    /// Takes an attribute or a text field away from the document.
    pub fn remove(mut self, name: &str) -> Patch {
        self.removed.push(name.to_string());
        self
    }
}

/// A document to write: a key (unique in its namespace), text fields, attributes, a vector.
#[derive(Clone, Debug, Default)]
pub struct Document {
    key: u64,
    version: u64,
    texts: Vec<(String, String)>,
    attrs: Vec<(String, Value)>,
    vector: Option<Vec<f32>>,
}

impl Document {
    pub fn new(key: u64) -> Document {
        Document { key, ..Default::default() }
    }

    /// A version, for syncing from elsewhere: `Namespace::versions` tells what's stored.
    pub fn version(mut self, version: u64) -> Document {
        self.version = version;
        self
    }

    pub fn text(mut self, field: &str, text: impl Into<String>) -> Document {
        self.texts.push((field.to_string(), text.into()));
        self
    }

    /// A sparse field's terms and weights (kept to hundredths; terms with weights not above 0, or
    /// not below [`MAX_SPARSE_WEIGHT`](crate::analysis::MAX_SPARSE_WEIGHT), are left out).
    pub fn sparse<S: AsRef<str>>(self, field: &str, terms: &[(S, f32)]) -> Document {
        self.text(field, crate::analysis::encode_sparse(terms))
    }

    pub fn int(mut self, name: &str, value: i64) -> Document {
        self.attrs.push((name.to_string(), Value::Int(value)));
        self
    }

    pub fn string(mut self, name: &str, value: impl Into<String>) -> Document {
        self.attrs.push((name.to_string(), Value::Str(value.into())));
        self
    }

    /// A float attribute's value (not NaN: a document with one is refused).
    pub fn float(mut self, name: &str, value: f64) -> Document {
        self.attrs.push((name.to_string(), Value::Float(value)));
        self
    }

    /// A tags attribute's strings (each kept once; empty ones are none).
    pub fn tags<S: Into<String>>(mut self, name: &str, tags: impl IntoIterator<Item = S>) -> Document {
        self.attrs.push((name.to_string(), Value::Tags(tags.into_iter().map(Into::into).collect())));
        self
    }

    pub fn vector(mut self, v: Vec<f32>) -> Document {
        self.vector = Some(v);
        self
    }
}

/// A result: its key, its score (BM25, the vector score, or fused), the parts it came from,
/// its attributes, and its stored text with a snippet. Every score is higher for a better hit.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub key: u64,
    pub score: f32,
    /// BM25 score, when the query had text and this document matched it.
    pub text_score: Option<f32>,
    /// When the query had a vector: how near this document's is, by the schema's metric. The
    /// cosine; for `Metric::Euclidean` the squared distance negated (0 for the query itself,
    /// lower the farther); for `Metric::Dot` the dot product.
    pub vector_score: Option<f32>,
    pub attributes: Vec<(String, Value)>,
    /// Stored text by field (empty when the schema doesn't store text, or `Query::with_text(false)`).
    pub text: Vec<(String, String)>,
    /// The field whose text best matches the query, and the best part of it with the query's
    /// words marked (for a vector query alone: the start of the first field).
    pub snippet: Option<(String, Snippet)>,
}

impl Hit {
    pub fn attr(&self, name: &str) -> Option<&Value> {
        self.attributes.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }

    /// A text field's stored text.
    pub fn text(&self, field: &str) -> Option<&str> {
        self.text.iter().find(|(n, _)| n == field).map(|(_, t)| t.as_str())
    }
}

impl Namespace {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    fn to_doc(&self, mut d: Document) -> Result<Doc> {
        let mut texts = vec![String::new(); self.schema.text.len().max(1)];
        for (field, t) in d.texts {
            let f = self.schema.field(&field).ok_or_else(|| Error::new(ErrorKind::InvalidInput, format!("no text field {field:?} in the schema")))?;
            texts[f as usize] = t;
        }
        for (name, v) in &mut d.attrs {
            match (self.schema.kind(name), &*v) {
                (Some(Kind::Float), Value::Float(x)) if x.is_nan() => return Err(Error::new(ErrorKind::InvalidInput, format!("attribute {name:?} is not a number (NaN)"))),
                // An integer given for a float attribute is that number.
                (Some(Kind::Float), Value::Int(x)) => *v = Value::Float(*x as f64),
                // One string given for a tags attribute is a set of one.
                (Some(Kind::Tags), Value::Str(t)) => *v = Value::Tags(vec![t.clone()]),
                (Some(k), _) if k == v.kind() => {}
                (Some(_), _) => return Err(Error::new(ErrorKind::InvalidInput, format!("attribute {name:?} has another type in the schema"))),
                (None, _) => return Err(Error::new(ErrorKind::InvalidInput, format!("no attribute {name:?} in the schema"))),
            }
        }
        if let Some(v) = &d.vector {
            if self.schema.vector != Some(v.len()) {
                return Err(Error::new(ErrorKind::InvalidInput, format!("a vector of {} dimensions; the schema has {:?}", v.len(), self.schema.vector)));
            }
        }
        Ok(Doc { key: d.key, version: d.version, passages: vec![texts], attrs: d.attrs, vector: d.vector })
    }

    /// Adds or replaces a document, searchable on return and recoverable after a process crash.
    /// Buffered WAL writes can be lost on OS crash/power loss until `sync` or `flush` succeeds.
    pub fn upsert(&self, d: Document) -> Result<()> {
        let doc = self.to_doc(d)?;
        self.index.write().upsert(&doc)
    }

    /// The document with `key` as it is, with `patch` applied; none if there is no such
    /// document.
    fn patched(&self, view: &crate::index::View, key: u64, patch: &Patch) -> Result<Option<Doc>> {
        let Some(whole) = view.document(key) else { return Ok(None) };
        let refuse = |what: String| Err(Error::new(ErrorKind::InvalidInput, what));
        // Its text is indexed again from what was stored: without that there is none to index.
        if !self.schema.text.is_empty() && !self.schema.store_text {
            return refuse(format!("namespace {:?} doesn't keep text (store_text is off), so a document can't be changed in part: write it whole", self.name));
        }
        if whole.passages > 1 {
            return refuse(format!("document {key} has {} passages; one of several passages can't be changed in part", whole.passages));
        }
        if let Some((name, _)) = patch.texts.iter().find(|(n, _)| self.schema.field(n).is_none()) {
            return refuse(format!("no text field {name:?} in the schema"));
        }
        if let Some(name) = patch.removed.iter().find(|n| self.schema.field(n).is_none() && self.schema.kind(n).is_none()) {
            return refuse(format!("no text field or attribute {name:?} in the schema"));
        }
        let gone = |name: &str| patch.removed.iter().any(|r| r == name);
        // A sparse field's terms aren't kept as they were given (they aren't text to show), so
        // there is nothing to write them again from: the patch has to bring them, or drop them.
        for ((field, _), analyzer) in self.schema.text.iter().zip(self.schema.analyzers()) {
            if analyzer.sparse && !gone(field) && !patch.texts.iter().any(|(n, _)| n == field) {
                return refuse(format!("namespace {:?} has the sparse field {field:?}, whose terms aren't kept as they were given: a patch must give them again (Patch::sparse) or remove the field, or the document be written whole", self.name));
            }
        }
        let mut d = Document::new(key).version(patch.version.unwrap_or(whole.version));
        for (field, text) in named(&self.schema, whole.text) {
            if !gone(&field) && !patch.texts.iter().any(|(n, _)| *n == field) {
                d = d.text(&field, text);
            }
        }
        d.texts.extend(patch.texts.iter().cloned());
        d.attrs = without_legacy_text(whole.attrs).into_iter().filter(|(n, _)| !gone(n) && !patch.attrs.iter().any(|(p, _)| p == n)).collect();
        d.attrs.extend(patch.attrs.iter().cloned());
        d.vector = patch.vector.clone().or(whole.vector);
        self.to_doc(d).map(Some)
    }

    /// Changes part of a document: the text fields, attributes and vector `patch` names are
    /// replaced (or, with `Patch::remove`, taken away) and the rest stays as it is. False if
    /// there is no document with that key (nothing is written then).
    ///
    /// The document is read and written again whole under the writer's lock, so a patch costs
    /// what an upsert does, and needs the text to have been stored (the default).
    pub fn patch(&self, key: u64, patch: &Patch) -> Result<bool> {
        let mut index = self.index.write();
        match self.patched(&index.view(), key, patch)? {
            Some(doc) => index.upsert(&doc).map(|()| true),
            None => Ok(false),
        }
    }

    /// `patch`, for every document that passes `filter` now; how many that was. In batches of
    /// 10,000 documents, each all or nothing.
    pub fn patch_where(&self, filter: &Filter, patch: &Patch) -> Result<u64> {
        let mut index = self.index.write();
        let keys = index.view().keys(filter);
        for batch in keys.chunks(10_000) {
            let view = index.view();
            let docs: Vec<Doc> = batch.iter().filter_map(|&k| self.patched(&view, k, patch).transpose()).collect::<Result<_>>()?;
            index.write(&docs.iter().map(crate::index::Op::Put).collect::<Vec<_>>())?;
        }
        Ok(keys.len() as u64)
    }

    /// `upsert`, unless there is a document with this key that doesn't pass `condition`:
    /// written if there is none yet, or if the one there passes (a revision below the new one,
    /// a state it may leave). Whether it was written. Checked and written under the writer's
    /// lock, so nothing comes between.
    pub fn upsert_if(&self, d: Document, condition: &Filter) -> Result<bool> {
        let doc = self.to_doc(d)?;
        let mut index = self.index.write();
        match index.view().passes(doc.key, condition) {
            Some(false) => Ok(false),
            _ => index.upsert(&doc).map(|()| true),
        }
    }

    /// `patch`, only if the document passes `condition`. Whether it was changed.
    pub fn patch_if(&self, key: u64, patch: &Patch, condition: &Filter) -> Result<bool> {
        let mut index = self.index.write();
        let view = index.view();
        if view.passes(key, condition) != Some(true) {
            return Ok(false);
        }
        match self.patched(&view, key, patch)? {
            Some(doc) => index.upsert(&doc).map(|()| true),
            None => Ok(false),
        }
    }

    /// `delete`, only if the document passes `condition`. Whether it was deleted.
    pub fn delete_if(&self, key: u64, condition: &Filter) -> Result<bool> {
        let mut index = self.index.write();
        if index.view().passes(key, condition) != Some(true) {
            return Ok(false);
        }
        index.delete(key).map(|()| true)
    }

    /// Several at once, as one batch: all or (after a crash mid-write) none of them, and queries
    /// see all of them or none.
    pub fn upsert_many(&self, docs: impl IntoIterator<Item = Document>) -> Result<()> {
        self.apply(docs.into_iter().map(Change::Put))
    }

    /// Puts and deletes as one batch, atomically (for replacing a set of documents, say).
    pub fn apply(&self, changes: impl IntoIterator<Item = Change>) -> Result<()> {
        enum Prepared {
            Put(Doc),
            Delete(u64),
        }
        let prepared: Vec<Prepared> = changes
            .into_iter()
            .map(|c| match c {
                Change::Put(d) => self.to_doc(d).map(Prepared::Put),
                Change::Delete(k) => Ok(Prepared::Delete(k)),
            })
            .collect::<Result<_>>()?;
        let ops: Vec<Op> = prepared
            .iter()
            .map(|p| match p {
                Prepared::Put(d) => Op::Put(d),
                Prepared::Delete(k) => Op::Delete(*k),
            })
            .collect();
        self.index.write().write(&ops)
    }

    pub fn delete(&self, key: u64) -> Result<()> {
        self.index.write().delete(key)
    }

    /// Deletes every document that passes `filter` now, and says how many that was: what a
    /// user asked to have forgotten (`attr("user").eq(id)`), what has expired
    /// (`attr("at").lt(cutoff)`). One batch: all of them go or, on an error, none; a document
    /// written while it runs is not looked at.
    pub fn delete_where(&self, filter: &Filter) -> Result<u64> {
        self.index.write().delete_where(filter)
    }

    /// The namespace as of now, kept consistent while it's held.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot { ns: self.clone(), view: self.index.view() }
    }

    /// A document's attributes, if it exists.
    pub fn get(&self, key: u64) -> Option<Vec<(String, Value)>> {
        let attrs = self.index.view().get(key)?;
        Some(without_legacy_text(attrs))
    }

    /// A document's stored text, by field (fields without text left out).
    pub fn text(&self, key: u64) -> Option<Vec<(String, String)>> {
        let view = self.index.view();
        view.get(key)?;
        Some(named(&self.schema, view.text(key)))
    }

    /// Every document's version, by key.
    pub fn versions(&self) -> HashMap<u64, u64> {
        self.index.view().versions()
    }

    /// How many documents pass `filter` (every document, without one). Counted from the
    /// filter's columns, without reading the documents: a million of them, a twentieth of a
    /// millisecond a condition.
    pub fn count(&self, filter: Option<&Filter>) -> u64 {
        self.index.view().count(filter)
    }

    /// How many documents have each value of the attribute `attr`, among those that pass
    /// `filter`: the most frequent first, at most `limit` (0: every value). For attributes of
    /// few values (a kind, an owner, a status): it holds a count per value.
    pub fn count_by(&self, attr: &str, filter: Option<&Filter>, limit: usize) -> Vec<(Value, u64)> {
        self.index.view().count_by(attr, filter, limit)
    }

    /// Synchronizes earlier writes to storage without building segments. Covers writes completed
    /// before this call takes the writer mutex; concurrent later writes need another sync.
    /// Uses file synchronization and Unix directory synchronization, subject to the filesystem
    /// and device honoring those requests. Non-Unix directory entries are not synchronized.
    /// Handle errors explicitly: dropping a namespace only checkpoints on a best-effort basis.
    pub fn sync(&self) -> Result<()> {
        self.index.write().sync()
    }

    /// Synchronizes writes and writes memory as a segment (also happens on its own).
    pub fn flush(&self) -> Result<()> {
        self.index.write().checkpoint()
    }

    /// Whether this handle only reads (`Database::open_read_only`).
    pub fn read_only(&self) -> bool {
        self.index.write().read_only()
    }

    /// For a read-only namespace: reads the files again if they have changed since it was
    /// opened or last refreshed, so queries from now on see what the writer (or a copy from
    /// it) has brought. Whether anything had changed. Queries running meanwhile finish on what
    /// they started with. On a namespace opened for writing this does nothing: it is always
    /// current. If the schema has grown since, open the namespace again to see the new one.
    pub fn refresh(&self) -> Result<bool> {
        let mut index = self.index.write();
        if !index.read_only() {
            return Ok(false);
        }
        let dir = self.home.join(&self.name);
        let now = Index::on_disk(&dir);
        if now == self.index.seen.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(false);
        }
        let fresh = Index::open_read_only(&dir, self.schema.text.len().max(1))?;
        self.index.reader.adopt(fresh.reader().view());
        *index = fresh;
        self.index.seen.store(now, std::sync::atomic::Ordering::Relaxed);
        Ok(true)
    }

    /// What tells this namespace from every other, one of the same name made later in its
    /// place included (32 hex digits; a fork has its own). Empty for a read-only copy of a
    /// namespace written before these were kept.
    pub fn id(&self) -> String {
        self.index.write().id()
    }

    /// The directory this namespace's files are in.
    pub fn dir(&self) -> PathBuf {
        self.home.join(&self.name)
    }

    /// What a copy of this namespace is made of, as of now: its schema, its manifest, and the
    /// names and sizes of the files the manifest names, all in `dir()`. What is in memory is
    /// written out first. Those files are never written again once named, so a copy can take
    /// them one by one; if one has gone meanwhile (a merge), ask again and start from the new
    /// manifest. A directory holding the files, then the schema, then the manifest (last) is
    /// this namespace as it was, and `Database::open_read_only` reads it.
    pub fn files(&self) -> Result<Files> {
        let mut index = self.index.write();
        let (manifest, files) = index.files()?;
        Ok(Files { id: index.id(), schema: self.schema.encode(), manifest, files })
    }

    /// A namespace of its own, named `name` in the same database, holding what this one holds
    /// now: for trying something on real data (an agent's sandbox, a migration, an
    /// experiment) without touching it. From then on the two are apart: writes, deletes and
    /// merges in one change nothing in the other, and either can be deleted.
    ///
    /// Takes milliseconds whatever the namespace's size, and no more disk until the two
    /// differ: its files are links to this one's, which are never changed once written (see
    /// `Index::fork`). What is in memory is flushed first. `AlreadyExists` if there is a
    /// namespace of that name.
    pub fn fork(&self, name: &str) -> Result<Namespace> {
        self.fork_into(&Database { dir: self.home.clone() }, name)
    }

    /// `fork`, into another database (copied, not linked, if that is on another filesystem).
    pub fn fork_into(&self, database: &Database, name: &str) -> Result<Namespace> {
        if !valid_name(name) {
            return Err(Error::new(ErrorKind::InvalidInput, "namespace names are letters, digits, - and _"));
        }
        let to = database.dir.join(name);
        if to.exists() {
            return Err(Error::new(ErrorKind::AlreadyExists, format!("there is a namespace {name:?} in {} already", database.dir.display())));
        }
        // Built under a name no namespace can have and renamed whole, so a fork is there or it
        // isn't; what a process that died left under that name goes first.
        let building = database.dir.join(format!(".fork-{name}"));
        let _ = std::fs::remove_dir_all(&building);
        let mut renamed = false;
        let made = (|| -> Result<Namespace> {
            self.index.write().fork(&building)?;
            crate::index::write_atomic(&building.join("schema"), self.schema.encode().as_bytes())?;
            std::fs::rename(&building, &to)?;
            renamed = true;
            crate::index::sync_dir(&database.dir)?;
            database.namespace(name, self.schema.clone())
        })();
        // A fork that failed is not there, whichever step failed (unless someone else has it
        // open by now: then it is theirs).
        if let Err(e) = &made {
            let _ = std::fs::remove_dir_all(&building);
            if renamed && e.kind() != ErrorKind::WouldBlock {
                let _ = std::fs::remove_dir_all(&to);
            }
        }
        made
    }

    /// Flushes, then merges every segment into one: the fastest layout for queries, after a bulk
    /// load. Queries run meanwhile; writes wait only for brief moments.
    pub fn optimize(&self) -> Result<()> {
        self.flush()?;
        let mut started = false;
        loop {
            {
                let mut index = self.index.write();
                index.poll_merge()?;
                if !index.stats().merging {
                    if started || !index.start_merge(true) {
                        return Ok(());
                    }
                    started = true;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    pub fn stats(&self) -> crate::index::Stats {
        self.index.write().stats()
    }

    /// Reads every file on disk against the checksums written with it and says what it read; an
    /// error names the first damaged file. Slow (it reads everything); queries and writes go on
    /// meanwhile. Files from before there were checksums are counted as unchecked: `optimize`
    /// rewrites them.
    pub fn verify(&self) -> Result<crate::index::Verified> {
        self.index.view().verify()
    }

    /// The writer's limits: the largest batch it accepts, as logged (default 1 GiB; larger ones are
    /// refused before anything is written), and the bytes in memory before a flush (256 MiB),
    /// alongside its 20,000 passages.
    /// (max_batch_bytes, flush_bytes), as `set_limits` takes them.
    pub fn limits(&self) -> (usize, usize) {
        let index = self.index.write();
        (index.max_batch_bytes, index.flush_bytes)
    }

    pub fn set_limits(&self, max_batch_bytes: usize, flush_bytes: usize) {
        let mut index = self.index.write();
        index.max_batch_bytes = max_batch_bytes;
        index.flush_bytes = flush_bytes.max(1);
    }

    pub fn query(&self) -> Query<'_> {
        Query { ns: self, view: None, text: None, vector: None, filter: None, limit: 10, order_by: None, all_words: false, typing: 0, nprobe: 0, snippet: 240, with_text: true, fusion: Fusion::default(), sparse: Vec::new(), candidates: (0, 0), sparse_candidates: 0, deadline: None, cancel: None }
    }
}

// MARK: Queries

/// How a hybrid query combines its two retrievers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fusion {
    /// `text_weight` × BM25 + (1 − `text_weight`) × the vector score (the cosine, or what the
    /// schema's metric gives), each scaled to 0–1 over the results it retrieved (a document one
    /// retriever didn't find gets 0 from it). The default, at 0.5.
    Convex { text_weight: f32 },
    /// Reciprocal rank fusion: the sum of 1 / (k + rank) over both rankings (k = 60 is usual).
    /// Ignores scores, so needs no weighting, but weighs a weak retriever as much as a strong one.
    Rrf { k: f32 },
}

impl Default for Fusion {
    fn default() -> Fusion {
        Fusion::Convex { text_weight: 0.5 }
    }
}

/// How many candidates each retriever of a hybrid query finds by default, before fusion: the
/// same however many results are asked for (up to these), so asking for 10 or 11 results can't
/// change the ranking. Chosen on BEIR dev and train queries, never test ones
/// (benchmarks/beir/candidates.jsonl): more text candidates helped on all four sets; more vector
/// ones than 100 hurt on SciFact and Quora, since each list's scores are scaled over what it found.
/// Against the old 40 and 100 at a limit of 10: FiQA nDCG@10 0.358 → 0.370 for 0.28 → 0.39 ms,
/// SciFact 0.908 → 0.927, Quora 0.877 → 0.884, NFCorpus 0.351 → 0.354.
pub const TEXT_CANDIDATES: usize = 200;
pub const VECTOR_CANDIDATES: usize = 100;
/// The lexical side's default when it's a sparse field alone (with text too, the larger of the
/// two budgets).
pub const SPARSE_CANDIDATES: usize = 200;

/// The most words (or sparse terms) a query may have: a guard against runaway inputs.
pub const MAX_QUERY_TERMS: usize = 4096;

/// Fuses ranked (key, score) lists: (key, fused score, text score, vector score), best first.
pub fn fuse(text: &[(u64, f32)], vector: &[(u64, f32)], fusion: Fusion) -> Vec<(u64, f32, Option<f32>, Option<f32>)> {
    let mut fused: HashMap<u64, (f32, Option<f32>, Option<f32>)> = HashMap::new();
    let scaled = |list: &[(u64, f32)]| -> Vec<(u64, f32, f32)> {
        let (lo, hi) = list.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &(_, s)| (lo.min(s), hi.max(s)));
        list.iter().map(|&(k, s)| (k, s, if hi > lo { (s - lo) / (hi - lo) } else { 1.0 })).collect()
    };
    match fusion {
        Fusion::Convex { text_weight } => {
            let w = text_weight.clamp(0.0, 1.0);
            for (k, s, x) in scaled(text) {
                let e = fused.entry(k).or_default();
                e.0 += w * x;
                e.1 = Some(s);
            }
            for (k, s, x) in scaled(vector) {
                let e = fused.entry(k).or_default();
                e.0 += (1.0 - w) * x;
                e.2 = Some(s);
            }
        }
        Fusion::Rrf { k: rk } => {
            for (rank, &(k, s)) in text.iter().enumerate() {
                let e = fused.entry(k).or_default();
                e.0 += 1.0 / (rk + rank as f32 + 1.0);
                e.1 = Some(s);
            }
            for (rank, &(k, s)) in vector.iter().enumerate() {
                let e = fused.entry(k).or_default();
                e.0 += 1.0 / (rk + rank as f32 + 1.0);
                e.2 = Some(s);
            }
        }
    }
    let mut v: Vec<_> = fused.into_iter().map(|(k, (s, t, ve))| (k, s, t, ve)).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

/// What one retriever of a hybrid query found before fusion: its budget, how many candidates it
/// returned, and the range of their raw scores, which min–max fusion scales to 0–1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Retrieved {
    pub depth: usize,
    pub found: usize,
    /// (lowest, highest) raw score; none when it found nothing.
    pub scores: Option<(f32, f32)>,
}

/// How a hybrid query's two rankings became one (`Query::explain`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusionStats {
    /// BM25 over the text fields, plus any sparse fields.
    pub lexical: Retrieved,
    pub vector: Retrieved,
    /// Candidates both retrievers found.
    pub both: usize,
    /// Distinct candidates fused, before the limit.
    pub fused: usize,
}

impl Retrieved {
    fn of(depth: usize, found: &[Found]) -> Retrieved {
        let scores = found.iter().map(|h| h.score).fold(None, |r: Option<(f32, f32)>, s| Some(r.map_or((s, s), |(lo, hi)| (lo.min(s), hi.max(s)))));
        Retrieved { depth, found: found.len(), scores }
    }
}

/// A query: text, a vector, or both; a filter; a limit; or attributes alone, ordered.
pub struct Query<'a> {
    ns: &'a Namespace,
    /// A snapshot's view; none for the newest.
    view: Option<Arc<View>>,
    text: Option<String>,
    vector: Option<Vec<f32>>,
    filter: Option<Filter>,
    limit: usize,
    order_by: Option<(String, bool)>,
    all_words: bool,
    /// Completions of the text's last word to search for too (0: it is a whole word).
    typing: usize,
    nprobe: usize,
    snippet: usize,
    with_text: bool,
    fusion: Fusion,
    /// Weighted terms for sparse fields: (field, term, weight).
    sparse: Vec<(String, String, f32)>,
    /// A hybrid query's candidates per retriever (text, vector) and for a sparse field; 0 for the
    /// defaults.
    candidates: (usize, usize),
    sparse_candidates: usize,
    deadline: Option<std::time::Duration>,
    cancel: Option<crate::stop::Cancel>,
}

impl Query<'_> {
    /// Weighted terms for a sparse field (a learned sparse model's query side: SPLADE's encoder,
    /// or for inference-free models the tokenizer's tokens with the model's weights). Scored by
    /// dot product with the documents' weights, together with any `text`, as the lexical side of
    /// a hybrid query.
    pub fn sparse<S: Into<String>>(mut self, field: &str, terms: impl IntoIterator<Item = (S, f32)>) -> Self {
        for (t, w) in terms {
            self.sparse.push((field.to_string(), t.into(), w));
        }
        self
    }

    /// Full-text search (BM25) over the schema's text fields.
    pub fn text(mut self, q: impl Into<String>) -> Self {
        self.text = Some(q.into());
        self
    }

    /// Nearest neighbours by cosine similarity. With `text` too, the query is hybrid.
    pub fn vector(mut self, v: Vec<f32>) -> Self {
        self.vector = Some(v);
        self
    }

    pub fn filter(mut self, f: Filter) -> Self {
        self.filter = Some(match self.filter.take() {
            Some(prev) => Filter::And(vec![prev, f]),
            None => f,
        });
        self
    }

    pub fn limit(mut self, n: usize) -> Self {
        self.limit = n;
        self
    }

    /// Orders by an integer attribute (`descending`: largest first). Without text or a vector,
    /// lists documents by it; with them, reorders the best matches by it.
    pub fn order_by(mut self, attr: &str, descending: bool) -> Self {
        self.order_by = Some((attr.to_string(), descending));
        self
    }

    /// Every word of the text must match (in some field).
    pub fn all_words(mut self) -> Self {
        self.all_words = true;
        self
    }

    /// The text's last word is still being typed (a search box, as the user types): it also
    /// matches the words that start with it, the `completions` most common of them (32 is
    /// plenty). "tax ret" finds "tax return" and "tax returns".
    pub fn typing(mut self, completions: usize) -> Self {
        self.typing = completions;
        self
    }

    /// Vector clusters probed per segment before stopping (default 16): more is slower and
    /// closer to exact.
    /// How a hybrid query fuses text and vector results (default: `Fusion::Convex`, 0.5).
    pub fn fusion(mut self, f: Fusion) -> Self {
        self.fusion = f;
        self
    }

    /// Shorthand for `Fusion::Convex { text_weight }`: 0 ranks by the vector alone, 1 by text.
    /// How many candidates a hybrid query's text and vector retrievers each find before they're
    /// fused (0 for the default, `TEXT_CANDIDATES` and `VECTOR_CANDIDATES`). Each retriever's
    /// scores are scaled over what it found, so these shape the ranking; a limit above a budget
    /// raises it to the limit.
    pub fn candidates(mut self, text: usize, vector: usize) -> Self {
        self.candidates = (text, vector);
        self
    }

    /// Gives up after this long, with `ErrorKind::TimedOut` (never partial results). Checked between
    /// segments, MaxScore windows, term-at-a-time lists and vector clusters.
    pub fn deadline(mut self, after: std::time::Duration) -> Self {
        self.deadline = Some(after);
        self
    }

    /// Gives up, with `ErrorKind::Interrupted`, once `cancel` is cancelled from another thread.
    pub fn cancel(mut self, cancel: &crate::stop::Cancel) -> Self {
        self.cancel = Some(cancel.clone());
        self
    }

    /// The lexical side's candidates when it searches a sparse field (0 for `SPARSE_CANDIDATES`);
    /// with text as well, the larger of this and the text budget.
    pub fn sparse_candidates(mut self, n: usize) -> Self {
        self.sparse_candidates = n;
        self
    }

    pub fn text_weight(mut self, w: f32) -> Self {
        self.fusion = Fusion::Convex { text_weight: w };
        self
    }

    /// Snippets of about this many characters (default 240; 0 for none).
    pub fn snippet(mut self, chars: usize) -> Self {
        self.snippet = chars;
        self
    }

    /// Whether hits carry their whole stored text (default: yes). Snippets come either way.
    pub fn with_text(mut self, yes: bool) -> Self {
        self.with_text = yes;
        self
    }

    /// Clusters each segment probes for the vector search (more: slower, closer to exact). By
    /// default (0) each segment's own budget, calibrated when it's built for recall@10 of 0.997.
    pub fn nprobe(mut self, n: usize) -> Self {
        self.nprobe = n;
        self
    }

    pub fn run(self) -> Result<Vec<Hit>> {
        Ok(self.execute(false)?.0)
    }

    /// Runs the query and says how a hybrid one was fused: what each retriever found and how
    /// much they overlapped (none for a query with one retriever, or none).
    pub fn explain(self) -> Result<(Vec<Hit>, Option<FusionStats>)> {
        self.execute(true)
    }

    fn execute(self, explain: bool) -> Result<(Vec<Hit>, Option<FusionStats>)> {
        let stop = crate::stop::Stop { deadline: self.deadline.map(|d| std::time::Instant::now() + d), cancel: self.cancel.clone() };
        let words = self.text.as_deref().map_or(0, |t| t.split_whitespace().count());
        if words > MAX_QUERY_TERMS || self.sparse.len() > MAX_QUERY_TERMS {
            return Err(Error::new(ErrorKind::InvalidInput, format!("a query of {} words and {} sparse terms; at most {MAX_QUERY_TERMS} of each", words, self.sparse.len())));
        }
        // Weights of 0 or less are left out (as a document's are); infinite or NaN ones would rank
        // arbitrarily.
        if let Some((_, t, w)) = self.sparse.iter().find(|(_, _, w)| !w.is_finite()) {
            return Err(Error::new(ErrorKind::InvalidInput, format!("sparse term {t:?} has a weight of {w}; weights are finite")));
        }
        let schema = &self.ns.schema;
        if let Some(v) = &self.vector {
            if schema.vector != Some(v.len()) {
                return Err(Error::new(ErrorKind::InvalidInput, format!("a query vector of {} dimensions; the schema has {:?}", v.len(), schema.vector)));
            }
        }
        // A snapshot's view, or the newest for the length of the query (no shared count to bump).
        let current;
        let index: &View = match &self.view {
            Some(v) => v,
            None => {
                current = self.ns.index.reader.current();
                &current
            }
        };
        let fields: Vec<(u8, f32)> = schema.text.iter().enumerate().map(|(i, (_, b))| (i as u8, *b)).collect();
        let mut sparse: Vec<(u8, String, f32)> = Vec::new();
        for (field, term, w) in &self.sparse {
            let f = schema.field(field).filter(|&f| schema.analyzers()[f as usize].sparse).ok_or_else(|| Error::new(ErrorKind::InvalidInput, format!("no sparse field {field:?} in the schema")))?;
            sparse.push((f, term.clone(), *w));
        }
        let text = self.text.as_deref().filter(|t| !t.trim().is_empty());
        // The lexical retriever: BM25 over the text fields, plus sparse fields' dot products.
        let lexical = text.is_some() || !sparse.is_empty();
        let hybrid = lexical && self.vector.is_some();
        // Fused retrievers look deeper than the limit, by fixed budgets rather than a multiple of
        // it: scores are scaled over what each found, so a depth that grew with the limit made the
        // ranking depend on how many results were asked for.
        let budget = |set: usize, default: usize| if set == 0 { default } else { set };
        let lexical_budget = match (text.is_some(), !sparse.is_empty()) {
            (_, false) => budget(self.candidates.0, TEXT_CANDIDATES),
            (false, true) => budget(self.sparse_candidates, SPARSE_CANDIDATES),
            (true, true) => budget(self.candidates.0, TEXT_CANDIDATES).max(budget(self.sparse_candidates, SPARSE_CANDIDATES)),
        };
        let depth = if hybrid { lexical_budget.max(self.limit) } else { self.limit };
        let vector_depth = if hybrid { budget(self.candidates.1, VECTOR_CANDIDATES).max(self.limit) } else { self.limit };
        let lexical_hits = || -> Vec<Found> {
            if lexical {
                index.search(&Request { text: text.unwrap_or(""), fields: &fields, k: depth, all: self.all_words, expand: 0, prefix_last: self.typing, filter: self.filter.as_ref(), sparse: &sparse, stop: Some(&stop) })
            } else {
                Vec::new()
            }
        };
        let nearest_hits = || -> Vec<Found> { self.vector.as_ref().map_or_else(Vec::new, |v| index.nearest_until(v, vector_depth, self.filter.as_ref(), self.nprobe, &stop)) };
        // A hybrid query's two retrievers at once: it takes as long as the slower one.
        let (text_hits, vector_hits) = if hybrid { crate::index::query_pool().join(lexical_hits, nearest_hits) } else { (lexical_hits(), nearest_hits()) };
        // A search that gave up returned what it had: report the stop, not partial results.
        if stop.stopped() {
            return Err(stop.error());
        }
        let stats = (hybrid && explain).then(|| {
            let lexical: std::collections::HashSet<u64> = text_hits.iter().map(|h| h.key).collect();
            let both = vector_hits.iter().filter(|h| lexical.contains(&h.key)).count();
            FusionStats { lexical: Retrieved::of(depth, &text_hits), vector: Retrieved::of(vector_depth, &vector_hits), both, fused: text_hits.len() + vector_hits.len() - both }
        });
        let mut scored: Vec<(u64, f32, Option<f32>, Option<f32>)> = if hybrid {
            let t: Vec<(u64, f32)> = text_hits.iter().map(|h| (h.key, h.score)).collect();
            let v: Vec<(u64, f32)> = vector_hits.iter().map(|h| (h.key, h.score)).collect();
            fuse(&t, &v, self.fusion)
        } else if lexical {
            text_hits.iter().map(|h| (h.key, h.score, Some(h.score), None)).collect()
        } else if self.vector.is_some() {
            vector_hits.iter().map(|h| (h.key, h.score, None, Some(h.score))).collect()
        } else {
            let order = self.order_by.as_ref().map(|(a, d)| (a.as_str(), *d));
            index.query(self.filter.as_ref(), order, self.limit).iter().map(|h| (h.key, 0.0, None, None)).collect()
        };
        scored.truncate(self.limit);
        // The query's words, weighted by how informative they are here (idf), for snippets.
        let weights = if self.snippet == 0 { Vec::new() } else { index.term_weights(text.unwrap_or(""), &fields) };
        let mut hits: Vec<Hit> = scored
            .into_iter()
            .map(|(key, score, text_score, vector_score)| {
                // Text only when it's wanted: it's read from a blob file and decompressed. The item
                // is looked up once for both.
                let (attributes, texts) = index.hydrate(key, self.with_text || self.snippet > 0).unwrap_or_default();
                let attributes = without_legacy_text(attributes);
                let stored = named(schema, texts);
                let snippet = if self.snippet == 0 { None } else { best_snippet(schema, index.analyzers(), &stored, &weights, self.snippet) };
                Hit { key, score, text_score, vector_score, attributes, text: if self.with_text { stored } else { Vec::new() }, snippet }
            })
            .collect();
        // Equal scores in key order, whichever retriever found them and wherever the documents
        // are stored. (Which of several exactly equal ones a full top k of a vector search
        // holds is as they were met; with real embeddings two scores are never bit for bit the
        // same unless the vectors are.)
        if self.order_by.is_none() {
            hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.key.cmp(&b.key)));
        }
        if let Some((name, descending)) = &self.order_by {
            if text.is_some() || self.vector.is_some() {
                let value = |h: &Hit| match h.attr(name) {
                    Some(Value::Int(x)) => *x,
                    Some(Value::Float(x)) => crate::attrs::float_code(*x),
                    _ => i64::MIN,
                };
                hits.sort_by(|a, b| if *descending { value(b).cmp(&value(a)) } else { value(a).cmp(&value(b)) });
            }
        }
        Ok((hits, stats))
    }
}

/// Attributes without the text that segments from before blob files kept among them.
fn without_legacy_text(attrs: Vec<(String, Value)>) -> Vec<(String, Value)> {
    attrs.into_iter().filter(|(n, _)| !n.starts_with(STORED)).collect()
}

/// Stored text by field name, in schema order, fields without text left out.
fn named(schema: &Schema, texts: Option<Vec<String>>) -> Vec<(String, String)> {
    texts.unwrap_or_default().into_iter().zip(&schema.text).filter(|(t, _)| !t.is_empty()).map(|(t, (n, _))| (n.clone(), t)).collect()
}

/// The snippet of the field that matches best (the weightiest distinct words, then matches, then
/// the higher boost, then schema order).
fn best_snippet(schema: &Schema, analyzers: &[crate::analysis::Analyzer], stored: &[(String, String)], weights: &[(u8, String, f32)], size: usize) -> Option<(String, Snippet)> {
    let boost = |field: &str| schema.text.iter().find(|(n, _)| n == field).map_or(1.0, |(_, b)| *b);
    let mut best: Option<((f32, usize), f32, String, Snippet)> = None;
    for (field, text) in stored {
        let Some(f) = schema.field(field) else { continue };
        let words: Vec<(String, f32)> = weights.iter().filter(|(wf, _, _)| *wf == f).map(|(_, t, w)| (t.clone(), *w)).collect();
        let (s, score) = snippet_analyzed(text, &words, size, analyzers.get(f as usize).copied().unwrap_or_default());
        let better = best.as_ref().is_none_or(|(bs, bb, _, _)| score.0 > bs.0 + 1e-6 || ((score.0 - bs.0).abs() <= 1e-6 && (score.1 > bs.1 || (score.1 == bs.1 && boost(field) > *bb))));
        if better {
            best = Some((score, boost(field), field.clone(), s));
        }
    }
    best.map(|(_, _, f, s)| (f, s))
}

// MARK: Filters

/// The start of a filter on an attribute: `attr("year").gte(2025)`, `attr("kind").eq("pdf")`.
pub fn attr(name: &str) -> Attr {
    Attr(name.to_string())
}

pub struct Attr(String);

impl From<&str> for Value {
    fn from(s: &str) -> Value {
        Value::Str(s.to_string())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Value {
        Value::Str(s)
    }
}

impl From<i64> for Value {
    fn from(x: i64) -> Value {
        Value::Int(x)
    }
}

impl From<i32> for Value {
    fn from(x: i32) -> Value {
        Value::Int(x as i64)
    }
}

impl From<f64> for Value {
    fn from(x: f64) -> Value {
        Value::Float(x)
    }
}

/// A number in a condition: `attr("year").gte(2025)`, `attr("score").gte(0.5)`. Either works
/// on a column of either kind (a column of integers has the integers in a range of floats).
/// From an `i64` or an `f64` and no other type, so a literal is one of the two whatever its
/// size (`attr("at").gte(1_700_000_000_000)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Number {
    Int(i64),
    Float(f64),
}

impl From<i64> for Number {
    fn from(x: i64) -> Number {
        Number::Int(x)
    }
}

impl From<f64> for Number {
    fn from(x: f64) -> Number {
        Number::Float(x)
    }
}

impl Number {
    fn float(self) -> f64 {
        match self {
            Number::Int(x) => x as f64,
            Number::Float(x) => x,
        }
    }
}

impl Attr {
    pub fn eq(self, v: impl Into<Value>) -> Filter {
        Filter::Eq(self.0, v.into())
    }

    pub fn ne(self, v: impl Into<Value>) -> Filter {
        Filter::Not(Box::new(Filter::Eq(self.0, v.into())))
    }

    pub fn one_of<V: Into<Value>>(self, vs: impl IntoIterator<Item = V>) -> Filter {
        Filter::In(self.0, vs.into_iter().map(Into::into).collect())
    }

    pub fn gte(self, x: impl Into<Number>) -> Filter {
        match x.into() {
            Number::Int(x) => Filter::Range(self.0, Some(x), None),
            Number::Float(x) => Filter::FloatRange(self.0, Some(x), None),
        }
    }

    pub fn lte(self, x: impl Into<Number>) -> Filter {
        match x.into() {
            Number::Int(x) => Filter::Range(self.0, None, Some(x)),
            Number::Float(x) => Filter::FloatRange(self.0, None, Some(x)),
        }
    }

    pub fn gt(self, x: impl Into<Number>) -> Filter {
        match x.into() {
            Number::Int(x) => Filter::above(self.0, x),
            // From the next float up (none is above infinity).
            Number::Float(x) if x == f64::INFINITY => Filter::nothing(),
            Number::Float(x) => Filter::FloatRange(self.0, Some(x.next_up()), None),
        }
    }

    pub fn lt(self, x: impl Into<Number>) -> Filter {
        match x.into() {
            Number::Int(x) => Filter::below(self.0, x),
            Number::Float(x) if x == f64::NEG_INFINITY => Filter::nothing(),
            Number::Float(x) => Filter::FloatRange(self.0, None, Some(x.next_down())),
        }
    }

    /// From `low` to `high`, both included.
    pub fn between(self, low: impl Into<Number>, high: impl Into<Number>) -> Filter {
        match (low.into(), high.into()) {
            (Number::Int(low), Number::Int(high)) => Filter::Range(self.0, Some(low), Some(high)),
            (low, high) => Filter::FloatRange(self.0, Some(low.float()), Some(high.float())),
        }
    }

    /// A tags attribute has this tag. (The same as `eq`, which on tags is "one of them is".)
    pub fn has(self, tag: impl Into<String>) -> Filter {
        Filter::Eq(self.0, Value::Str(tag.into()))
    }

    /// A tags attribute has at least one of these.
    pub fn has_any<S: Into<String>>(self, tags: impl IntoIterator<Item = S>) -> Filter {
        Filter::In(self.0, tags.into_iter().map(|t| Value::Str(t.into())).collect())
    }

    /// A tags attribute has every one of these.
    pub fn has_all<S: Into<String>>(self, tags: impl IntoIterator<Item = S>) -> Filter {
        Filter::And(tags.into_iter().map(|t| Filter::Eq(self.0.clone(), Value::Str(t.into()))).collect())
    }

    /// Starts with (ignoring case). On tags: one of them does.
    pub fn prefix(self, p: &str) -> Filter {
        Filter::Prefix(self.0, p.to_string())
    }

    /// Contains these characters (ignoring case). On tags: one of them does; for "has this
    /// tag", `has`.
    pub fn contains(self, p: &str) -> Filter {
        Filter::Contains(self.0, p.to_string())
    }

    /// The whole string matches a pattern: `*` is any run of characters, `?` any one, `\`
    /// takes the next as itself (`attr("path").glob("/home/*/taxes/*.pdf")`). On tags: one of
    /// them does.
    pub fn glob(self, pattern: &str) -> Filter {
        Filter::Glob(self.0, pattern.to_string(), false)
    }

    /// `glob`, whatever the case.
    pub fn iglob(self, pattern: &str) -> Filter {
        Filter::Glob(self.0, pattern.to_string(), true)
    }

    pub fn exists(self) -> Filter {
        Filter::Exists(self.0)
    }
}

impl std::ops::BitAnd for Filter {
    type Output = Filter;
    fn bitand(self, rhs: Filter) -> Filter {
        Filter::And(vec![self, rhs])
    }
}

impl std::ops::BitOr for Filter {
    type Output = Filter;
    fn bitor(self, rhs: Filter) -> Filter {
        Filter::Or(vec![self, rhs])
    }
}

impl std::ops::Not for Filter {
    type Output = Filter;
    fn not(self) -> Filter {
        Filter::Not(Box::new(self))
    }
}
