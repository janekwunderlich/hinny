"""Encodes BEIR corpora (title + text) and test queries with an inference-free learned sparse
model (OpenSearch's doc-v3-distill): documents through the model, queries without one, as the
model's tokenizer's tokens with its static weights. Saves emb/<name>/sparse_{corpus,queries}.jsonl,
one {term: weight} per line in the order of embed_beir.py's ids.

    python -I embed_sparse.py data emb scifact nfcorpus arguana fiqa

Documents go in batches of 32 in corpus order with the MPS cache emptied after each: batches of
varying length otherwise grew the process past 40 GB on FiQA. Output is appended line by line,
and a run that stops resumes where it left off.

SPLIT=dev (or train) encodes only that split's queries, in the order of embed_beir.py's
query_ids-<split>.json, as sparse_queries-<split>.jsonl, for choosing settings without the test
queries.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SparseEncoder

MODEL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"
data, out = Path(sys.argv[1]), Path(sys.argv[2])
device = "mps" if torch.backends.mps.is_available() else "cpu"
m = SparseEncoder(MODEL, device=device)
m.max_seq_length = 512


def rows(sp):
    # A sparse COO tensor per batch: (row, token) → weight.
    sp = sp.coalesce().cpu()
    idx, val = sp.indices(), sp.values()
    res = [dict() for _ in range(sp.shape[0])]
    for r, c, v in zip(idx[0].tolist(), idx[1].tolist(), val.tolist()):
        if v > 0:
            res[r][m.tokenizer.convert_ids_to_tokens(c)] = round(v, 4)
    return res


for name in sys.argv[3:]:
    o = out / name
    split = os.environ.get("SPLIT")
    if split:
        qids = json.load(open(o / f"query_ids-{split}.json"))
        queries = {x["_id"]: x["text"] for x in (json.loads(l) for l in open(data / name / name / "queries.jsonl"))}
        with open(o / f"sparse_queries-{split}.jsonl", "w") as f:
            for i in range(0, len(qids), 256):
                for r in rows(m.encode_query([queries[q] for q in qids[i:i + 256]], convert_to_sparse_tensor=True)):
                    f.write(json.dumps(r) + "\n")
        print(json.dumps({"dataset": name, "split": split, "queries": len(qids)}), flush=True)
        continue
    ids = json.load(open(o / "corpus_ids.json"))
    qids = json.load(open(o / "query_ids.json"))
    d = data / name / name
    docs = {x["_id"]: x for x in (json.loads(l) for l in open(d / "corpus.jsonl"))}
    queries = {x["_id"]: x["text"] for x in (json.loads(l) for l in open(d / "queries.jsonl"))}
    t = time.time()
    with open(o / "sparse_queries.jsonl", "w") as f:
        for i in range(0, len(qids), 256):
            for r in rows(m.encode_query([queries[q] for q in qids[i:i + 256]], convert_to_sparse_tensor=True)):
                f.write(json.dumps(r) + "\n")
    texts = [f"{(docs[i].get('title') or '').strip()} {docs[i].get('text', '')}".strip() for i in ids]
    part = o / "sparse_corpus.partial.jsonl"
    done = sum(1 for _ in open(part)) if part.exists() else 0
    with open(part, "a") as f:
        for s in range(done, len(texts), 32):
            with torch.inference_mode():
                enc = rows(m.encode_document(texts[s:s + 32], batch_size=32, convert_to_sparse_tensor=True))
            for r in enc:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if device == "mps":
                torch.mps.empty_cache()
    part.rename(o / "sparse_corpus.jsonl")
    print(json.dumps({"dataset": name, "docs": len(ids), "queries": len(qids), "seconds": round(time.time() - t, 1)}), flush=True)
