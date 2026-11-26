"""Dataset 1 "clustered": 1,000,000 documents + 1,000 queries, 256 dimensions.

1,000 topic centres ~ N(0, 1); each vector = a random topic centre + 0.8 * N(0, 1) noise,
then L2-normalized. Queries are drawn the same way from the same generator (fresh draws,
not copies of documents). Fixed seed. Writes float32 C-order .npy files into data/.

    .venv/bin/python gen_clustered.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEED, N, NQ, D, TOPICS, NOISE = 20261113, 1_000_000, 1_000, 256, 1_000, 0.8


def draw(rng, centres, n):
    out = np.empty((n, D), dtype=np.float32)
    for s in range(0, n, 100_000):
        e = min(n, s + 100_000)
        topic = rng.integers(0, TOPICS, size=e - s)
        v = centres[topic] + NOISE * rng.standard_normal((e - s, D), dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        out[s:e] = v
    return out


def main():
    os.makedirs(DATA, exist_ok=True)
    rng = np.random.default_rng(SEED)
    centres = rng.standard_normal((TOPICS, D), dtype=np.float32)
    docs = draw(rng, centres, N)
    queries = draw(rng, centres, NQ)
    assert docs.dtype == np.float32 and docs.flags.c_contiguous
    np.save(os.path.join(DATA, "clustered_docs.npy"), docs)
    np.save(os.path.join(DATA, "clustered_queries.npy"), queries)
    print("docs", docs.shape, docs.dtype, "queries", queries.shape,
          "norms", float(np.linalg.norm(docs[:5], axis=1).mean()))


if __name__ == "__main__":
    main()
