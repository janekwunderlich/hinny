"""Dataset 2 "glove": real embeddings from the ann-benchmarks collection, angular (cosine).

Source (public, no credentials): http://ann-benchmarks.com/glove-200-angular.hdf5 (963 MB)
    train: 1,183,514 x 200 float32   GloVe word vectors (Twitter, 200 dimensions)
    test:  10,000 x 200 float32      held-out query vectors (not rows of train)

Preparation: all 1,183,514 train rows in file order as documents; the first 1,000 test rows as
queries; both L2-normalized (angular distance becomes inner product) and written as float32
C-order .npy. Ground truth is recomputed with ground_truth.py on the normalized vectors (the
neighbours stored in the file are not used).

    .venv/bin/python -I prep_glove.py     # downloads into download/glove/, writes data/glove_*.npy
"""
import os
import sys
import urllib.request

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DL = os.path.join(HERE, "download", "glove")
URL = "http://ann-benchmarks.com/glove-200-angular.hdf5"
NQ = 1000


def normalized(a):
    a = np.asarray(a, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    assert (n > 0).all(), "zero vector"
    return np.ascontiguousarray(a / n)


def main():
    os.makedirs(DL, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DL, "glove-200-angular.hdf5")
    if not os.path.exists(path):
        print("downloading", URL, flush=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req) as r, open(path + ".part", "wb") as f:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(path + ".part", path)
    with h5py.File(path, "r") as f:
        print({k: f[k].shape for k in f.keys()}, dict(f.attrs), flush=True)
        q = normalized(f["test"][:NQ])
        np.save(os.path.join(DATA, "glove_queries.npy"), q)
        train = f["train"]
        n, d = train.shape
        out = np.lib.format.open_memmap(os.path.join(DATA, "glove_docs.npy"), mode="w+", dtype=np.float32, shape=(n, d))
        for s in range(0, n, 100_000):
            out[s:s + 100_000] = normalized(train[s:s + 100_000])
        out.flush()
        print("docs", out.shape, "queries", q.shape, flush=True)
    if "--keep" not in sys.argv:
        os.remove(path)  # the folder lives on a RAM-backed filesystem


if __name__ == "__main__":
    main()
