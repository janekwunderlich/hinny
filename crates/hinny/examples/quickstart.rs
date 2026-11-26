//! hinny in a minute: a namespace with a schema, documents with text, attributes and a vector,
//! then text, vector, hybrid and attribute queries.   cargo run --release -p hinny --example quickstart

use hinny::{Database, Document, Schema, attr};

fn main() -> std::io::Result<()> {
    let dir = std::env::temp_dir().join("hinny-quickstart");
    let db = Database::open(&dir)?;
    // Toy 3-dimensional "embeddings"; use your model's (384, 768, …) in practice.
    let notes = db.namespace("notes", Schema::new().text("title", 2.0).text("body", 1.0).int("year").string("kind").vector(3))?;

    notes.upsert_many([
        Document::new(1).text("title", "Lease agreement").text("body", "The apartment on Elm Street, from March").int("year", 2025).string("kind", "pdf").vector(vec![0.9, 0.1, 0.0]),
        Document::new(2).text("title", "Electricity invoice").text("body", "March, due in 14 days").int("year", 2026).string("kind", "pdf").vector(vec![0.1, 0.9, 0.0]),
        Document::new(3).text("title", "Viewing notes").text("body", "Bright apartment, quiet street").int("year", 2024).string("kind", "md").vector(vec![0.8, 0.0, 0.2]),
    ])?;

    let show = |label: &str, hits: Vec<hinny::Hit>| {
        println!("{label}:");
        for h in hits {
            println!("  {} score {:.3} (text {:?}, vector {:?})", h.key, h.score, h.text_score, h.vector_score);
        }
    };
    show("text", notes.query().text("apartment lease").run()?);
    show("vector", notes.query().vector(vec![1.0, 0.0, 0.1]).limit(2).run()?);
    show("hybrid, PDFs only", notes.query().text("apartment").vector(vec![1.0, 0.0, 0.1]).filter(attr("kind").eq("pdf")).run()?);
    show("newest first", notes.query().filter(attr("year").gte(2025)).order_by("year", true).run()?);

    // Writes are searchable on return; sync requests storage synchronization, flush also writes a segment.
    notes.flush()?;
    drop(notes);
    std::fs::remove_dir_all(dir)
}
