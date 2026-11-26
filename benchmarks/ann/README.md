# Vector search calibration (plan 011)

`examples/ann_calibration.rs` measures approximate against exact vector search on real queries,
by k, filter and selectivity, in an index of 4 segments, unflushed vectors and 5% deleted.
Inputs are the BEIR embeddings `benchmarks/beir/embed_beir.py` writes (256 dimensions); nothing
here contains query or document text.

- `calibration.jsonl`: the sweep of the depth exponent α ∈ {0, 0.1, 0.15, 0.2, 0.3} on dev and
  train queries (FiQA dev, NFCorpus dev, SciFact train, Quora dev; up to 1,000 each).
- `heldout.jsonl`: the one run on BEIR's test queries (the same datasets, SciDocs and ArguAna),
  at α = 0 and the chosen 0.1.
- `decide.py FILE`: plan 011's rule (cells on the approximate path, worst mean recall and 5th
  percentile per exponent, the choice).

Rerun one dataset (a bench build, for `--exponent`):

    cargo run --release -p hinny --offline --features bench --example ann_calibration -- \
        EMB/fiqa/corpus.npy EMB/fiqa/queries-dev.npy --name fiqa --queries 1000 --exponent 0.1

Measured on an Apple M-series laptop under other load: latencies are indicative, recall isn't
affected by load.
