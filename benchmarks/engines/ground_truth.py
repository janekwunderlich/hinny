"""Exact top 10 by inner product on L2-normalized vectors (= cosine), shared by all Python engines.

    .venv/bin/python ground_truth.py DATASET

Reads data/DATASET_docs.npy and data/DATASET_queries.npy, writes data/DATASET_gt.npy (int64, NQ x 10).
Computed with faiss IndexFlatIP, cross-checked against plain NumPy on the first 100 queries.
Also reports how far an exact top 10 over float16-rounded documents (how Hinny stores its
full-precision rows) is from the float32 one, to size the "different ground truth" bias.
"""
import os
import sys
import numpy as np
import faiss

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main():
    ds = sys.argv[1]
    docs = np.load(os.path.join(DATA, f"{ds}_docs.npy"))
    q = np.load(os.path.join(DATA, f"{ds}_queries.npy"))
    print(ds, docs.shape, q.shape, "doc norm min/max",
          float(np.linalg.norm(docs[:10000], axis=1).min()), float(np.linalg.norm(docs[:10000], axis=1).max()))
    ix = faiss.IndexFlatIP(docs.shape[1])
    ix.add(docs)
    D, I = ix.search(q, 10)
    np.save(os.path.join(DATA, f"{ds}_gt.npy"), I.astype(np.int64))
    # NumPy cross-check (float64 accumulation) on the first 100 queries.
    agree = 0
    for i in range(100):
        s = docs @ q[i].astype(np.float32)
        top = np.argpartition(-s, 10)[:10]
        agree += len(set(top.tolist()) & set(I[i].tolist()))
    print(f"NumPy cross-check: {agree}/1000 neighbour slots agree with faiss IndexFlatIP")
    # Gap between rank 10 and rank 11 would tell how fragile the set is; cheap proxy: f16 rounding.
    d16 = docs.astype(np.float16).astype(np.float32)
    ix16 = faiss.IndexFlatIP(docs.shape[1])
    ix16.add(d16)
    _, I16 = ix16.search(q, 10)
    same = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(I, I16))
    print(f"float16-rounded documents: exact top 10 shares {same}/{I.size} slots with the float32 top 10 "
          f"({same / I.size:.5f})")
    with open(os.path.join(DATA, f"{ds}_gt_info.txt"), "w") as f:
        f.write(f"numpy_crosscheck {agree}/1000\nf16_overlap {same}/{I.size}\n")


if __name__ == "__main__":
    main()
