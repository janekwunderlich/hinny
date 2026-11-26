//! # hinny
//!
//! An embedded hybrid search database: full-text search (BM25), vector search and typed
//! attributes with filters, in your process, live as you write.
//!
//! - **Namespaces** of **documents** with a [`Schema`]: text fields, integer, float and string
//!   attributes, and a vector.
//! - **One query builder** ([`Namespace::query`]) for text, vectors, or both: hybrid queries fuse
//!   BM25 and vector scores (a convex combination, or rank fusion), with the same filter applied
//!   inside both searches. Vectors are compared by cosine, or by Euclidean distance or dot
//!   product ([`Schema::vector_with`], [`Metric`]).
//! - **Durable and live:** writes go to a write-ahead log and are searchable at once; checkpoints
//!   write immutable, memory-mapped segments; merges run in the background; a crash replays the log.
//! - **Fast:** BM25 with block-max MaxScore over bit-packed posting lists (scores as
//!   Tantivy's); vectors as 1-bit RaBitQ codes over clusters, reranked at full precision where the
//!   codes can't decide (as turbopuffer does at scale).
//!
//! ```no_run
//! use hinny::{Database, Document, Schema, attr};
//!
//! let db = Database::open("data")?;
//! let docs = db.namespace("docs", Schema::new().text("title", 2.0).text("body", 1.0).int("year").vector(384))?;
//! docs.upsert(Document::new(1).text("title", "Lease").text("body", "The apartment lease").int("year", 2025).vector(vec![0.0; 384]))?;
//! let hits = docs.query().text("apartment lease").vector(vec![0.0; 384]).filter(attr("year").gte(2025)).limit(10).run()?;
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! The modules below are the engine: [`index`] (the live index), [`segment`], [`search`] (BM25),
//! [`vector`], [`attrs`] (attributes and filters).

pub mod analysis;
pub mod api;
pub mod attrs;
pub mod bitmap;
pub mod blob;
mod codes;
pub mod hash;
pub mod bitpack;
#[doc(hidden)]
pub mod faults;
pub mod index;
pub mod norms;
pub mod search;
pub mod segment;
pub mod snippet;
pub mod stop;
mod substring;
pub mod tokenize;
pub mod vector;

/// BM25 term-frequency saturation and length normalization (Tantivy's and Lucene's defaults).
pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;

/// The file-name index (macOS): every file and folder name under a folder, searched as you type.
#[cfg(feature = "names")]
pub use hinny_names as names;

pub use analysis::{Analyzer, Language};
pub use api::{Attr, Change, Database, Document, Files, Fusion, FusionStats, Hit, Namespace, Number, Patch, Query, Retrieved, Schema, Snapshot, attr, valid_name};
pub use attrs::{Filter, Kind, Value};
pub use snippet::Snippet;
pub use stop::Cancel;
pub use vector::Metric;
