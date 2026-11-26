/** The whole API at a glance: a schema, a document, and one hybrid query with a filter. Shown on
 * the home page and at the top of the docs in each supported interface. The code is plain text;
 * components/highlight.tsx colours it. */
export const pythonShort = String.raw`import hinny
from hinny import attr

docs = hinny.Database("data").namespace("docs", text="body", ints=["year"], vector=384)
docs.upsert(1, text="The lease for the flat on Elm Street", attrs={"year": 2025}, vector=emb)

hits = docs.query(text="apartment lease", vector=q, filter=attr("year") >= 2025)  # one ranking`

export const rustShort = String.raw`let docs = Database::open("data")?
    .namespace("docs", Schema::new().text("body", 1.0).int("year").vector(384))?;
docs.upsert(Document::new(1).text("body", "The lease for the flat on Elm Street")
    .int("year", 2025).vector(emb))?;

let hits = docs.query().text("apartment lease").vector(q)
    .filter(attr("year").gte(2025)).run()?;   // one ranking`

export const nodeShort = String.raw`const { Database, attr } = require('hinny')

const docs = new Database('data').namespace('docs', { text: 'body', ints: ['year'], vector: 384 })
docs.upsert({ key: 1, text: 'The lease for the flat on Elm Street', attrs: { year: 2025 }, vector: emb })

const hits = docs.query({ text: 'apartment lease', vector: q, filter: attr('year').gte(2025) }) // one ranking`

/** The C interface (crates/hinny-c, hinny.h): documents, queries and results are JSON, vectors
 * plain arrays of float. */
export const cShort = String.raw`hinny_db *db = hinny_open("data", &err);
hinny_namespace *docs = hinny_namespace_open(db, "docs",
    "{\"text\": [{\"name\": \"body\"}], \"ints\": [\"year\"], \"vector\": 384}", &err);
hinny_upsert(docs, "{\"key\": 1, \"text\": {\"body\": \"The lease for the flat on Elm Street\"}, "
                   "\"ints\": {\"year\": 2025}}", emb, 384, &err);

char *hits = hinny_query(docs, "{\"text\": \"apartment lease\", "
                   "\"filter\": {\"range\": [\"year\", 2025, null]}}", q, 384, &err);   /* one ranking, as JSON */`

/** For the docs, Python first: it is the API the docs describe. */
export const firstQuery = [
  { name: 'Python', code: pythonShort },
  { name: 'Rust', code: rustShort },
  { name: 'Node.js', code: nodeShort },
  { name: 'C', code: cShort },
]

/** For the home page, in the order the four interfaces are described there. */
export const fourWays = [
  { name: 'Rust', code: rustShort },
  { name: 'Python', code: pythonShort },
  { name: 'Node.js', code: nodeShort },
  { name: 'C', code: cShort },
]
