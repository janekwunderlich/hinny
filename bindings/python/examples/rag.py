"""Retrieval for RAG with hinny: chunk documents, embed them, store them, retrieve hybrid.

    python examples/rag.py "how does hinny decide which vectors to rerank?"

The corpus is this repository's own documentation. Embeddings come from sentence-transformers
when it's installed (any model works); otherwise from a small hashing stand-in, so the example
runs anywhere. Swap `embed` for your model.
"""

import hashlib
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

import hinny
from hinny import attr

ROOT = Path(__file__).resolve().parents[3]

try:
    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    DIMS = _model.get_sentence_embedding_dimension()

    def embed(texts):
        return _model.encode(texts, normalize_embeddings=True).astype(np.float32)

except ImportError:
    DIMS = 256

    def embed(texts):
        """A stand-in: hashed word counts (a real model understands meaning; this matches words)."""
        out = np.zeros((len(texts), DIMS), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[i, int(hashlib.blake2b(w.encode(), digest_size=4).hexdigest(), 16) % DIMS] += 1.0
        return out


def chunks(text, size=700):
    """Paragraphs, packed into chunks of about `size` characters."""
    out, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        if current and len(current) + len(para) > size:
            out.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        out.append(current.strip())
    return out


def main():
    question = " ".join(sys.argv[1:]) or "how does hinny decide which vectors to rerank?"
    with tempfile.TemporaryDirectory() as tmp:
        db = hinny.Database(tmp)
        store = db.namespace("chunks", text="text", ints=["chunk"], strings=["source"], vector=DIMS)

        files = [ROOT / "README.md", ROOT / "DESIGN.md", ROOT / "bindings/python/README.md"]
        docs = []
        for path in files:
            parts = chunks(path.read_text())
            for i, (part, vec) in enumerate(zip(parts, embed(parts))):
                docs.append({"key": hinny.key(f"{path.name}#{i}"), "text": part, "attrs": {"source": path.name, "chunk": i}, "vector": vec})
        store.upsert_many(docs)
        print(f"{len(docs)} chunks from {len(files)} files\n")

        # The question as both words and meaning; a filter could narrow the sources.
        hits = store.query(text=question, vector=embed([question])[0], limit=4)
        print(f"Q: {question}\n")
        for h in hits:
            print(f"[{h.attrs['source']} #{h.attrs['chunk']}] fused {h.score:.4f}  (BM25 {h.text_score}, cosine {h.vector_score})")

        only_design = store.query(text=question, vector=embed([question])[0], filter=attr("source") == "DESIGN.md", limit=2)
        print(f"\nfrom DESIGN.md only: {[h.attrs['chunk'] for h in only_design]}")

        # Hits carry their stored text (and a snippet, with the question's words marked).
        context = "\n\n---\n\n".join(h.text["text"] for h in hits)
        prompt = f"Answer from the context below.\n\n{context}\n\nQuestion: {question}"
        print(f"\nprompt: {len(prompt):,} characters from {len(hits)} chunks")
        best = hits[0]
        marked = best.snippet
        for a, b in reversed(best.highlights):
            marked = marked[:a] + "**" + marked[a:b] + "**" + marked[b:]
        print(f"best snippet: {marked}")

if __name__ == "__main__":
    main()
