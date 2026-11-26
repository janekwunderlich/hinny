"""hinny: an embedded hybrid search database.

Full-text search (BM25), vector search (cosine) and typed attributes with filters, in your
process, live as you write. Text and vectors are searched together and fused into one ranking.

    import hinny
    from hinny import attr

    db = hinny.Database("data")
    docs = db.namespace("docs", text={"title": 2.0, "body": 1.0}, ints=["year"], strings=["kind"], vector=384)
    docs.upsert(1, text={"title": "Lease", "body": "The apartment lease"}, attrs={"year": 2025, "kind": "pdf"}, vector=embedding)
    hits = docs.query(text="apartment lease", vector=query_embedding, filter=(attr("year") >= 2025) & (attr("kind") == "pdf"))
"""

from ._hinny import Attr, Database, Filter, Hit, Namespace, Snapshot, __version__, attr, key

__all__ = ["Database", "Namespace", "Snapshot", "Hit", "Filter", "Attr", "attr", "key", "__version__"]
