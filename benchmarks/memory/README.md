# Conversational-memory retrieval: LongMemEval and LoCoMo

How plain hybrid retrieval in Hinny (BM25 + one small open embedding model, no LLM anywhere)
does on the two datasets that memory products (Supermemory, Mem0, Zep) publish numbers for.
Retrieval only: we measure whether the evidence the dataset labels is in the top k, not
whether an LLM answers correctly. Measured 2026-11-13 on a shared Apple M1 Max under Asahi
Linux, pinned to cores 0–3 (two efficiency, two performance cores) at `nice 19`.

Results: [RESULTS.md](RESULTS.md). This file is the protocol and how to rerun it.

## How the reference protocols measure retrieval

### LongMemEval (Wu et al., ICLR 2025; arXiv 2410.10813; github.com/xiaowu0162/LongMemEval)

Read from `src/retrieval/run_retrieval.py`, `src/retrieval/eval_utils.py` and
`src/evaluation/print_retrieval_metrics.py` at the repository's current head.

- **Haystack.** Each question has its own history: ~48 sessions (~115k tokens) in
  LongMemEval-S, ~500 sessions (~1.5M tokens) in LongMemEval-M. Retrieval runs per question
  over that question's haystack only.
- **Memory unit ("value granularity").** `session` or `turn`. The paper's §5.1: "when sessions
  or rounds are used as the key, we only keep the user-side utterances". In the code a
  session's key is its user messages joined with a space; a "round"/"turn" item is one user
  message (the assistant reply is not indexed). Item ids are `session_id` or
  `session_id_<1-based index of the user message in the session>`.
- **Labels.** Evidence sessions have ids containing `answer`; evidence turns carry
  `has_answer: true`. The code renames an evidence session (or turn) whose evidence is not on
  the user side to `noans`, so it is not counted as evidence. Consequence: the 56
  `single-session-assistant` questions, whose evidence is in assistant replies, have no
  evidence item at all under the code's labels for 51 of them, and `recall_all` over an
  empty set is 1.0 (Python's `all([])`), while NDCG is 0. Their contribution to the averages
  is fixed regardless of the retriever.
- **Metrics.** `recall_any@k`, `recall_all@k` (every evidence item in the top k) and
  `ndcg_any@k`, for k ∈ {1, 3, 5, 10, 30, 50}. Binary relevance; DCG = rel₁ + Σᵢ≥₂ relᵢ/log₂ i
  (positions 1 and 2 both weigh 1); ideal DCG over every relevant item in the corpus. The
  paper's tables report "Recall@k" = `recall_all@k` and "NDCG@k" = `ndcg_any@k` at k = 5
  and 10. For turn-level retrieval the code also scores at session level ("turn2session"):
  map turns to sessions, widen k until the top holds k distinct sessions.
- **Questions.** The 30 abstention questions are always skipped: 470 of 500.
- **Baselines.** `flat-bm25` is `rank_bm25.BM25Okapi` over whitespace-split text (no
  lowercasing, no stemming), query split the same way. `flat-contriever` is
  `facebook/contriever` (110M), mean pooling, tokenizer truncation at 512 tokens, **raw dot
  product** (no normalization). Stella V5 1.5B and gte-Qwen2-7B are the larger ones.
- **Where the paper's retrieval numbers come from.** Every retrieval table in the paper
  (Tables 3, 4, 9, 10) is on **LongMemEval-M**, not S. The paper publishes no retrieval
  figures for S.

### MemoryBench (github.com/supermemoryai/memorybench, commit 063b86b, 2026-10-20)

- **Pipeline.** ingest → index → search → answer → evaluate → report, per provider. The
  LongMemEval loader downloads `longmemeval_s_cleaned.json`, **deletes `has_answer`**, and
  renames sessions `<question_id>-session-<i>`, so the dataset's evidence labels never reach
  the evaluation. LoCoMo keeps `evidence` as metadata but the evaluation does not use it.
- **Memory unit.** Whatever the provider does with a session; MemoryBench hands it whole
  sessions (`provider.ingest([session])`). Supermemory, Mem0 and Zep extract memories with an
  LLM server-side; the built-in `rag` and `filesystem` providers call OpenAI for extraction
  and (rag) `text-embedding-3-small` embeddings.
- **Search.** `provider.search(question, {limit})`, `limit` = 10 for LongMemEval, 30 for
  LoCoMo.
- **"Retrieval" metrics** (`orchestrator/phases/retrieval-eval.ts`): the **judge LLM** is
  shown the question, the expected answer and the top-10 results as JSON, and labels each
  result relevant or not. From those labels: Hit@10, MRR, precision, and an NDCG whose ideal
  is `max(1, relevant retrieved)`; "recall" is defined as Hit@10 (`relevantRetrieved > 0`).
  So MemoryBench's retrieval numbers need an LLM and an API key, and measure something
  different from label-based Recall@k.
- **Answer accuracy** (what providers headline): an answering model (default `gpt-4o`) gets
  the question, the question date and the search results as raw JSON; a judge model
  (default `gpt-4o`) scores it 0/1 with LongMemEval's question-type-specific judge prompts.
  Needs API keys for both, and for the provider.
- **Without an LLM**: only ingest, search, and the latency and context-token counts. Nothing
  in MemoryBench's scoring runs without an LLM.

## The protocol used here

- **Data.** LongMemEval-S, the cleaned release (`longmemeval_s_cleaned.json`, 2025-09, the
  one the repository now points to and MemoryBench uses), 470 non-abstention questions. Also
  the original S and M files, text-only, for the sanity check. LoCoMo `locomo10.json` (10
  conversations, 1,540 non-adversarial questions; 4 have no evidence ids and are skipped).
- **Units.** `session` and `round`, each with two keys:
  - `key=user`: the paper's key (user text only). This is what the sanity check uses.
  - `key=full`: user and assistant text (for a round: the user message and the reply after
    it). Closer to what a memory product indexes.
  Every item carries `ts` (session timestamp), `session` (id) and `round` as attributes.
- **Labels.** `paper`: exactly the code's (above). `all`: evidence sessions =
  `answer_session_ids`; evidence rounds = rounds where either message has `has_answer`. The
  `all` labels are the headline: they do not hand the single-session-assistant questions a
  free 1.0.
- **Retrievers.** In Hinny, each question's haystack in a fresh database:
  `bm25` (`language="english"`: stemming and stopwords), `bm25-plain` (default analyzer),
  `vector` (Contriever, cosine), `hybrid` (default fusion: scores scaled to 0–1,
  `text_weight=0.5`), `hybrid-rrf` (`fusion="rrf"`);
  `vector-ip`, `hybrid-ip`, `hybrid-ip-rrf`: the same with vectors that rank by inner product
  under Hinny's cosine (each document vector gets an extra component `sqrt(M² − |x|²)`, the
  query a 0; M = the haystack's largest norm), because Contriever is a dot-product model and
  Hinny has no dot-product metric. Outside Hinny, as references:
  `ref-bm25` (the paper's rank_bm25 call), `ref-dot` (the paper's Contriever dot product,
  brute force), `ref-cos` (brute-force cosine). No parameter was tuned on these datasets.
- **Embedding model.** `facebook/contriever`, encoded exactly as the paper's code does.
  Chosen because it is the only small model the paper reports (Stella is 1.5B, GTE 7B), so
  the vector row has a published counterpart, and it runs on CPU. On these cores it embeds
  ~5 texts a second (512-token inputs: ~1.7/s), so the full S set (~18k user-side sessions,
  ~18k full sessions, ~94k rounds) would take about 10 hours: **vector and hybrid rows use a
  fixed random subset of 100 of the 470 questions** (`common.subset_order`, seed 0); text-only
  rows are given for all 470 and for the same 100. Round vectors exist only for `key=user`
  (round `key=full` would add ~95k long texts). Contriever truncates at 512 tokens, so a full
  session (mean ~2.6k tokens) is embedded from its first ~512 tokens only; user-side sessions
  (mean ~320 tokens) mostly fit.
- **Metrics.** As the paper's code (`common.py` reproduces `eval_utils.py`). `R@k` =
  `recall_all@k`, k ∈ {5, 10, 20}; `N@10` = NDCG@10. Round runs are also scored at session
  level (turn2session, as the paper's code). `recall_any` and the fraction of evidence
  recalled are in the raw output.
- **Timings.** Ingest = one `upsert_many` of a haystack into an empty namespace (text,
  attributes, vectors; WAL write included, no `optimize()`). Query = median of five runs of
  the same query, `snippet=0, with_text=False`, after a first ("cold") run that also builds
  scan caches; reported as p50/p99 across questions. Through Python, single thread.

## Files

- `common.py`: corpus construction, labels and metrics (the paper's), embedding cache.
- `embed.py`: Contriever embeddings, cached on disk by content hash; resumable.
- `run_lme.py`: LongMemEval runs (Hinny and reference retrievers) → per-question JSONL.
- `run_locomo.py`: the same for LoCoMo.
- `report.py`: tables from the JSONL.
- `results/`: the generated tables (`report.py` output) behind RESULTS.md.
- `RESULTS.md`: the numbers, the comparison with published figures, and what an end-to-end
  run would take.

## Rerun

Large files stay outside the repository (`MEMBENCH_WORK`, default
`/home/janek/.t3/scratch/2026-11-13-you-are-taking-over-a-d7cb416e/bench/memory`): datasets
in `data/`, model cache in `hf/` (`HF_HOME`), embeddings in `emb/`, raw results in
`results/`, databases in `tmp/`.

```sh
W=$MEMBENCH_WORK; export HF_HOME=$W/hf TMPDIR=$W/tmp
uv venv $W/venv --python 3.12 && VIRTUAL_ENV=$W/venv uv pip install maturin numpy rank_bm25 ijson \
    sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
VIRTUAL_ENV=$W/venv $W/venv/bin/maturin develop --profile python --manifest-path bindings/python/Cargo.toml
cd $W/data/lme && wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
    https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s \
    https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_m   # save as *.json
git clone https://github.com/snap-research/locomo $W/src/locomo
cd benchmarks/memory
taskset -c 0-3 nice -n 19 python embed.py --questions 100 --hours 3          # ~2 h here
python run_lme.py --questions 100 --out $W/results/s_cleaned_100.jsonl
python run_lme.py --no-vectors --out $W/results/s_cleaned_text.jsonl
python run_lme.py --data M_ORIG --no-vectors --configs session:user,round:user --out $W/results/m_orig_text.jsonl
python run_locomo.py --out $W/results/locomo.jsonl                            # embeds ~6k turns first
python report.py $W/results/s_cleaned_100.jsonl --labels all --by-type session:user --timing
```
