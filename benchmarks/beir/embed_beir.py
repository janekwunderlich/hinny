"""Embeds BEIR corpora and test queries with EmbeddingGemma 2, as hinny-embed does: documents as
"title: {title} | text: {text}", queries with the search prompt, 512 tokens, the first 256
dimensions normalized. Saves emb/<name>/{corpus,queries}.npy and their ids.

SPLIT=dev (or train) embeds only that split's queries, as emb/<name>/queries-<split>.npy and
query_ids-<split>.json, for choosing settings without looking at the test queries."""
import json, os, sys, time
from pathlib import Path
import numpy as np

data, out = Path(sys.argv[1]), Path(sys.argv[2])


def mlx_encoder():
    """An MLX port of the model (mlx_embed.py: the same vectors, cosine 0.9999, 3.6x as fast), when
    this Python has mlx and tokenizers."""
    import importlib.util
    path = Path(__file__).resolve().with_name("mlx_embed.py")
    spec = importlib.util.spec_from_file_location("hinny_embed", path)
    fe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe)
    return fe.build_mlx()


try:
    import mlx.core  # noqa: F401
    encode = mlx_encoder()
    torch = None
except ImportError:
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("google/embeddinggemma-2", device="mps", model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = 512
    encode = lambda texts: model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

def enc(texts):
    # Longest first, 512 at a time with the MPS cache emptied between: in one call, batches of
    # every length grew the process past 29 GB on SciDocs.
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    parts = []
    for s in range(0, len(order), 512):
        parts.append(encode([texts[i] for i in order[s:s + 512]])[:, :256])
        if torch is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    res = np.empty((len(texts), 256), dtype=np.float32)
    res[order] = np.concatenate(parts)
    return res / np.linalg.norm(res, axis=1, keepdims=True).clip(min=1e-12)

for name in sys.argv[3:]:
    d = data / name / name
    o = out / name
    o.mkdir(parents=True, exist_ok=True)
    split = os.environ.get("SPLIT")
    if split:
        wanted = {l.split("\t")[0] for l in list(open(d / "qrels" / f"{split}.tsv"))[1:]}
        queries = [q for q in (json.loads(l) for l in open(d / "queries.jsonl")) if q["_id"] in wanted]
        t = time.time()
        np.save(o / f"queries-{split}.npy", enc(["task: search result | query: " + q["text"] for q in queries]))
        json.dump([q["_id"] for q in queries], open(o / f"query_ids-{split}.json", "w"))
        print(json.dumps({"dataset": name, "split": split, "queries": len(queries), "seconds": round(time.time() - t, 1)}), flush=True)
        continue
    docs = [json.loads(l) for l in open(d / "corpus.jsonl")]
    test = {l.split("\t")[0] for l in list(open(d / "qrels" / "test.tsv"))[1:]}
    queries = [q for q in (json.loads(l) for l in open(d / "queries.jsonl")) if q["_id"] in test]
    t = time.time()
    qv = enc(["task: search result | query: " + q["text"] for q in queries])
    np.save(o / "queries.npy", qv); json.dump([q["_id"] for q in queries], open(o / "query_ids.json", "w"))
    cv = enc([f"title: {(x.get('title') or '').strip() or 'none'} | text: {x.get('text', '')}" for x in docs])
    np.save(o / "corpus.npy", cv); json.dump([x["_id"] for x in docs], open(o / "corpus_ids.json", "w"))
    print(json.dumps({"dataset": name, "docs": len(docs), "queries": len(queries), "seconds": round(time.time() - t, 1), "docs_per_s": round(len(docs) / (time.time() - t), 1)}), flush=True)
