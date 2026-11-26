# Results: plain hybrid retrieval in Hinny on LongMemEval and LoCoMo

Measured 2026-11-13. Protocol, label sets and rerun instructions: [README.md](README.md). Full
tables (every mode, both label sets, per question type, timings): [results/](results/).

No LLM was called anywhere. Embeddings: `facebook/contriever` (110M) on CPU. Hinny at the
working tree of 2026-11-13 (`hinny 0.3.0`), Python binding built with `--profile python`.

**Read this first.**
- Vector and hybrid rows on LongMemEval-S are a **fixed random subset of 100 of the 470
  questions** (embedding all of S on four cores would take ~10 h). The ± is a 95% interval
  over questions: about ±0.05–0.09 at n = 100. Text-only rows cover all 470.
- `R@k` is the paper's "Recall": `recall_all@k`, every evidence item in the top k. Labels are
  `all` (answer sessions; rounds where either side holds evidence) unless marked `paper`.
- Hinny's vector search is cosine only. Contriever is trained for dot product, and cosine
  costs it a lot here (session R@10 0.84 vs 0.98). The `-ip` rows rank by inner product inside
  Hinny with the standard reduction (each document vector gets one extra component so all
  norms are equal; the query gets a 0); they reproduce the paper's dot-product ranking
  exactly (`vector-ip` = brute-force `ref-dot` on every question). This is a workaround in
  the benchmark scripts, not an engine change.

## 1. Sanity check against the paper (LongMemEval-M, BM25, all 470 questions)

Every retrieval number in the LongMemEval paper is on LongMemEval-**M** (~500 sessions per
question), so this is the only setting where a row can be checked against it. Text only.

| Value = session, key = user | R@5 | N@5 | R@10 | N@10 |
|---|---|---|---|---|
| Paper, Table 9, BM25 (K = V) | 0.634 | 0.516 | 0.710 | 0.540 |
| Paper's BM25 call, rerun here (`ref-bm25`) | 0.615 | 0.534 | 0.694 ±0.042 | 0.558 |
| Hinny `bm25` (english analyzer) | **0.768** | **0.680** | **0.845** ±0.033 | **0.701** |
| Hinny `bm25-plain` (default analyzer) | 0.728 | 0.652 | 0.796 | 0.673 |

| Value = round, key = user | R@5 | N@5 | R@10 | N@10 |
|---|---|---|---|---|
| Paper, Table 9, BM25 | 0.472 | 0.352 | 0.538 | 0.372 |
| `ref-bm25`, turn level | 0.426 | 0.361 | 0.489 ±0.045 | 0.384 |
| `ref-bm25`, turn level scored as sessions | 0.449 | – | 0.517 | 0.387 |
| Hinny `bm25` | **0.596** | **0.521** | **0.687** ±0.042 | **0.552** |

The rerun of the paper's own BM25 lands within 0.02 of Table 9 at session level (inside the
interval) and 0.05 low on round recall (NDCG 0.01 high). I could not find the cause of the
round gap; likely candidates are dataset revisions (the original M file was used) and
the paper's own spread: its Table 3 and Table 9 give different numbers for the same Stella
configuration (round R@5 0.582 vs 0.660). The pipeline (corpus, labels, metrics) is therefore
taken as reproducing the paper's. The gain of Hinny's BM25 over the paper's is tokenization:
the paper splits on spaces only (no lowercasing, punctuation stays attached).

**The vector row could not be checked against the paper.** The paper's Contriever numbers
are on M only (session R@5 0.723, R@10 0.823; round 0.589 / 0.747), and embedding M is ~15k
sessions and ~78k rounds per 40 questions, well past the CPU budget. What was verified
instead: the embedding is the paper's code path (same model, pooling, truncation), and
Hinny's vector rankings equal brute-force cosine, and its `-ip` rankings equal the paper's
brute-force dot product, on every question.

## 2. LongMemEval-S (cleaned), 100-question subset, labels = all

| Session granularity | key | R@5 | R@10 | R@20 | N@10 |
|---|---|---|---|---|---|
| `ref-bm25` (paper's BM25) | user | 0.690 | 0.760 ±0.084 | 0.850 | 0.783 |
| `ref-dot` (paper's Contriever) | user | 0.870 | 0.980 ±0.027 | 0.990 | 0.942 |
| Hinny `bm25` | user | 0.840 | 0.900 ±0.059 | 0.920 | 0.892 |
| Hinny `vector` (cosine) | user | 0.710 | 0.840 ±0.072 | 0.940 | 0.813 |
| Hinny `vector-ip` | user | 0.870 | 0.980 ±0.027 | 0.990 | 0.944 |
| Hinny `hybrid` (default fusion) | user | 0.880 | 0.930 ±0.050 | 0.960 | 0.931 |
| Hinny `hybrid-rrf` | user | 0.870 | 0.930 ±0.050 | 0.960 | 0.922 |
| Hinny `hybrid-ip` (default fusion) | user | 0.870 | 0.970 ±0.033 | 0.980 | 0.935 |
| Hinny `hybrid-ip-rrf` | user | 0.890 | 0.950 ±0.043 | 0.980 | 0.939 |
| Hinny `bm25` | full | 0.830 | 0.880 ±0.064 | 0.950 | 0.895 |
| Hinny `vector` (cosine) | full | 0.700 | 0.830 ±0.074 | 0.940 | 0.773 |
| Hinny `vector-ip` | full | 0.780 | 0.870 ±0.066 | 0.940 | 0.834 |
| Hinny `hybrid` | full | 0.860 | 0.940 ±0.047 | 0.960 | 0.919 |
| Hinny `hybrid-rrf` | full | 0.860 | 0.940 ±0.047 | 0.960 | 0.897 |
| Hinny `hybrid-ip` | full | 0.860 | 0.940 ±0.047 | 0.960 | 0.927 |

| Round granularity (turn level) | key | R@5 | R@10 | R@20 | N@10 | as sessions R@10 |
|---|---|---|---|---|---|---|
| `ref-bm25` | user | 0.510 | 0.610 ±0.096 | 0.650 | 0.507 | 0.640 |
| `ref-dot` | user | 0.650 | 0.780 ±0.081 | 0.850 | 0.672 | 0.850 |
| Hinny `bm25` | user | 0.690 | 0.720 ±0.088 | 0.820 | 0.687 | 0.760 |
| Hinny `vector` (cosine) | user | 0.590 | 0.740 ±0.086 | 0.820 | 0.606 | 0.820 |
| Hinny `vector-ip` | user | 0.650 | 0.780 ±0.081 | 0.850 | 0.672 | 0.850 |
| Hinny `hybrid` | user | 0.680 | 0.810 ±0.077 | 0.860 | 0.713 | 0.860 |
| Hinny `hybrid-rrf` | user | 0.680 | 0.800 ±0.078 | 0.870 | 0.712 | 0.870 |
| Hinny `hybrid-ip` | user | 0.690 | 0.790 ±0.080 | 0.870 | 0.726 | 0.870 |
| Hinny `bm25` (text only) | full | 0.670 | 0.770 ±0.082 | 0.820 | 0.677 | 0.820 |

R@10 by question type, session granularity, key = user (n per type in brackets):

| | KU (17) | MS (24) | SSA (13) | SSP (6) | SSU (10) | TR (30) |
|---|---|---|---|---|---|---|
| Hinny `bm25` | 0.941 | 0.833 | 1.000 | 1.000 | 1.000 | 0.833 |
| Hinny `vector` (cosine) | 0.941 | 0.792 | 0.923 | 1.000 | 1.000 | 0.700 |
| Hinny `vector-ip` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.933 |
| Hinny `hybrid` | 1.000 | 0.917 | 1.000 | 1.000 | 1.000 | 0.833 |
| Hinny `hybrid-ip` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 |

Per-type cells rest on 6–30 questions each; differences under ~0.1 are noise.

Text only, all 470 questions, labels = all (session, key = user): `ref-bm25` R@5 0.740,
R@10 0.813, R@20 0.885; Hinny `bm25` 0.864 / 0.915 / 0.945, N@10 0.915. Round, key = user:
`ref-bm25` 0.500 / 0.600 / 0.668; Hinny `bm25` 0.694 / 0.798 / 0.851. The subset's BM25 rows
are 0.01–0.08 below the full set's, so the subset is, if anything, slightly harder.

Under the paper's labels the numbers move by a few points (the 51 single-session-assistant
questions with no user-side evidence count as recall 1.0, NDCG 0); both are in `results/`.

What the numbers say, within their intervals:
- On S, sessions are easy: ~48 per haystack, so R@20 is close to half the haystack.
  Session-level hybrid and BM25 both find all evidence in the top 10 for 90–97% of questions.
- Hybrid ≥ the better of its two parts, except against inner-product vectors at session
  level (0.97 vs 0.98, within noise). Hinny's default fusion and RRF are indistinguishable
  here.
- Multi-session and temporal-reasoning questions are where retrieval misses; at round
  granularity, multi-session R@10 is ~0.55 for every method (several evidence rounds must all
  be in the top 10).
- With cosine Contriever, hybrid hides most of the vector side's weakness (0.93 vs 0.84).

## 3. LoCoMo (locomo10), 1,536 questions (categories 1–4), labels = dataset evidence

| Turn granularity | R@5 | R@10 | R@20 | N@10 | any@5 | session-level R@10 |
|---|---|---|---|---|---|---|
| `ref-bm25` | 0.270 | 0.342 ±0.024 | 0.409 | 0.283 | 0.326 | 0.713 |
| `ref-dot` (Contriever) | 0.408 | 0.501 ±0.025 | 0.581 | 0.428 | 0.511 | 0.813 |
| Hinny `bm25` | 0.484 | 0.552 ±0.025 | 0.609 | 0.513 | 0.590 | 0.857 |
| Hinny `vector` (cosine) | 0.249 | 0.332 ±0.024 | 0.426 | 0.268 | 0.317 | 0.708 |
| Hinny `vector-ip` | 0.408 | 0.501 ±0.025 | 0.581 | 0.428 | 0.511 | 0.813 |
| Hinny `hybrid` | 0.452 | 0.547 ±0.025 | 0.636 | 0.481 | 0.566 | 0.844 |
| Hinny `hybrid-rrf` | 0.389 | 0.480 ±0.025 | 0.556 | 0.412 | 0.483 | 0.805 |
| Hinny `hybrid-ip` | **0.522** | **0.598** ±0.025 | **0.673** | **0.545** | **0.641** | **0.869** |

| Session granularity (~27 sessions per conversation) | R@5 | R@10 | N@10 |
|---|---|---|---|
| Hinny `bm25` | 0.785 | 0.864 | 0.823 |
| Hinny `vector-ip` | 0.559 | 0.712 | 0.606 |
| Hinny `hybrid` | 0.785 | 0.877 | 0.817 |
| Hinny `hybrid-ip` | 0.784 | 0.876 | 0.818 |

Turn level, R@10 by category (names as Mem0 and most follow-up work map the dataset's
numbers: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop; the LoCoMo paper's own table
order is ambiguous): Hinny `hybrid-ip` multi-hop 0.188 (282), open-domain 0.261 (92),
single-hop 0.741 (841), temporal 0.682 (321).

LoCoMo's only published retrieval figures are the LoCoMo paper's (Maharana et al., ACL 2024,
arXiv 2402.17753, Table 3): DRAGON retriever, unit = dialog turn, "Recall Accuracy" overall
R@5 58.8, R@10 67.5, R@25 79.9. Not directly comparable: a different retriever, adversarial
questions included, and the recall definition is not stated (it could be any-hit or fraction
of evidence; our `any@5` for `hybrid-ip` is 0.641, fraction-of-evidence at 10 is 0.659).

## 4. Hinny timings (Hinny only)

Through the Python binding, one thread, on a shared machine under load (load average 7–15
during the runs; the machine runs two other agents' builds), cores 0–3 = two efficiency and
two performance cores of an M1 Max, `nice 19`. Ingest = one `upsert_many` of a whole haystack
into an empty namespace (text, three attributes, 768-d vectors; WAL included, no
`optimize()`). Query = median of five repeats after a cold first run, `limit` = min(items,
100), no snippets; p50/p99 across questions.

| Haystack | items | ingest p50 / p99 | BM25 p50 / p99 | vector p50 / p99 | hybrid p50 / p99 | hybrid-rrf p50 / p99 |
|---|---|---|---|---|---|---|
| S sessions, user key (100 q) | 47 | 2.6 / 5.5 ms | 0.017 / 0.043 ms | 0.033 / 0.042 ms | 0.085 / 0.199 ms | 0.086 / 0.183 ms |
| S sessions, full text (100 q) | 47 | 12.5 / 16.4 ms | 0.027 / 0.044 ms | 0.033 / 0.038 ms | 0.099 / 0.250 ms | 0.095 / 0.209 ms |
| S rounds, user key (100 q) | 243 | 3.9 / 8.2 ms | 0.028 / 0.077 ms | 0.081 / 0.107 ms | 0.166 / 0.378 ms | 0.164 / 0.352 ms |
| M sessions, text only (470 q) | 502 | 26 / 47 ms | 0.077 / 0.173 ms | – | – | – |
| M rounds, text only (470 q) | 2,431 | 26 / 53 ms | 0.093 / 0.255 ms | – | – | – |
| LoCoMo turns (10 conv., single runs) | 588 | 6.0 ms median | 0.126 / 0.384 ms | 0.165 / 0.687 ms | 0.216 / 1.01 ms | 0.187 / 0.723 ms |

Cold first queries (which also build per-segment scan caches) had p99s of 0.5–2 ms for hybrid;
those outliers are in `results/`. The `-ip` namespaces (769-d) time the same as cosine.
Embedding, not the database, is the cost: Contriever on these cores ran ~5 texts/s.

## 5. Next to published figures

| Source | Setting | Figure | What it measures |
|---|---|---|---|
| LongMemEval paper, Table 9 (arxiv.org/abs/2410.10813) | **M**, session, BM25 / Contriever / Stella 1.5B | R@10 0.710 / 0.823 / 0.794 | label-based recall_all, 470 q |
| same | **M**, round | R@10 0.538 / 0.747 / 0.784 | same |
| This run | **M**, session, Hinny BM25 | R@10 0.845 | same protocol |
| This run | **S** (cleaned), session, Hinny hybrid-ip / hybrid | R@10 0.970 / 0.930 (100 q) | same protocol |
| Supermemory research report, May 2026 (supermemory.ai/research/longmembench/, also supermemory.ai/research) | S, 500 q | "97% overall Recall@20 with aggregation"; per category SSU 97, SSA 100, SSP 95, KU 100, TR 95, MS 96 | **answer accuracy**: gpt-4o answers from the top-20 results, gpt-4o judges with LongMemEval's prompts. The same page's table is titled "LLM-as-judge evaluation … by answer model", and lists supermemory with gpt-5 at 84.6% and gemini-3-pro at 85.2% |
| Supermemory with gpt-4o, cited by third parties: Hindsight (arxiv.org/abs/2512.12818), Mastra (mastra.ai/research/observational-memory) | S | 81.6% | answer accuracy (older figure; not found on Supermemory's current pages; I did not verify its original source) |

**What is comparable.** The paper's BM25 row with ours on M, protocol for protocol (section
1). Our S rows with each other. Nothing else in the table.

**What is not.**
- Supermemory publishes no label-based retrieval figure. Its "Recall@20 with aggregation" is
  end-to-end QA accuracy judged by an LLM (it includes abstention questions, which have no
  evidence to retrieve), so retrieval recall cannot be compared with it. A retrieval R@20 of
  0.98 does not imply 98% answers, and a 97% answer score does not imply any particular recall.
- Supermemory's figures are self-reported and inconsistent across its own sources (97% with
  gpt-4o, 84.6% with gpt-5, 85.2% with gemini-3-pro on one page; 81.6% with gpt-4o cited
  elsewhere). Some per-category percentages do not correspond to whole numbers of questions
  (SSP 95% of 30 questions = 28.5; TR 95% of 133 = 126.4), so they are rounded or aggregated
  in a way the report does not describe.
- The paper's retrieval numbers are on M, ours mostly on S (~10× smaller haystacks): S numbers
  are higher for every method.
- MemoryBench's "retrieval metrics" are LLM-judged relevance of the top 10 (hit@10, MRR,
  an NDCG normalized by the hits it found), not the dataset's evidence labels; they are not
  the same quantity as R@k here either.
- LoCoMo: the LoCoMo paper's DRAGON numbers include adversarial questions and an unstated
  recall definition; Supermemory claims "#1" on LoCoMo with no figure.

## 6. What an end-to-end comparison would take (not run)

The way MemoryBench does it: per question, search (limit 10 for LongMemEval, 30 for LoCoMo),
then three LLM calls through the judge/answer models (defaults `gpt-4o` for both): one
answer, one 0/1 judgment, and one relevance labelling of the top-10 results (the
"retrieval metrics"). Sizes from the measured haystacks (~4 characters per token; results are
sent as pretty-printed JSON, which adds roughly 20%):

| Benchmark | Questions | LLM calls | Context per call | Input tokens, whole run |
|---|---|---|---|---|
| LongMemEval-S, Hinny returns rounds (user + reply, ~500 tokens) | 500 | 1,500 | ~6k | ~6M |
| LongMemEval-S, Hinny returns whole sessions (~2.6k tokens) | 500 | 1,500 | ~30k | ~30M |
| LoCoMo, 30 turns (~40 tokens each) | 1,986 (incl. adversarial) | ~6,000 | ~2k | ~9M |

Output is short (answers capped at 1,000 tokens; judgments a few dozen). The answering and
judging models should match whatever the compared provider's published figure used (gpt-4o
for Supermemory's 97% / 81.6%); with a different judge the numbers are not comparable. If
the Hinny provider used `text-embedding-3-small` like MemoryBench's `rag` provider, add the
embedding of every ingested haystack (~60M tokens for S: 500 haystacks of ~120k tokens, since MemoryBench ingests each question's haystack separately); with a local model there
is no API cost but the CPU time above (~10 h for S on four of these cores; a GPU or a smaller
model would cut it). Supermemory, Mem0 and Zep would also need their own API keys and run
their LLM extraction server-side on every ingested session.

**MemoryBench can take a new provider.** The interface (`src/types/provider.ts`) is
`initialize(config)`, `ingest(sessions, {containerTag})` → `{documentIds}`,
`awaitIndexing(result, containerTag)`, `search(query, {containerTag, limit})` → any array,
`clear(containerTag)`; optional custom answer/judge prompts. Registering one touches
`providers/index.ts`, the `ProviderName` union and `utils/config.ts` (which expects an API key
per provider; a local provider can pass a placeholder). Hinny has a Node binding (Node-API,
`bindings/node`), so a provider is one TypeScript file of roughly 100–150 lines: open a
`Database` per run, a namespace per `containerTag`, split sessions into rounds, embed, and
`upsertMany`; `search` maps to `query({text, vector, limit})` and returns text, session id and
date. Unverified: whether Bun loads the Hinny `.node` addon (Bun supports most of Node-API;
it was not tried), and which embedder to use without an API key (transformers.js would work
from TypeScript; the `-ip` trick above or a cosine-trained model avoids the dot-product
loss).

## 7. What I could not do

- **Contriever on LongMemEval-M**, so the vector row is not checked against the paper's
  (section 1). Time: ~2 h of CPU per 20 questions, too few to check anything.
- **All 470 S questions with vectors**: 100-question subset (seed 0).
- **Round vectors with the assistant reply** (`round`, key = full): BM25 only.
- **Stella or GTE** (the paper's other retrievers): too large for this CPU budget.
- **The paper's round gap** (our rerun of its BM25 is 0.05 below Table 9's round recall):
  not explained.
- **Supermemory's own retrieval**: its server is closed and running it needs an account and
  an API key, which were out of scope. Nothing was sent to any service.
- Binding gaps found: no inner-product (dot) vector metric, only cosine (worked around with
  an extra dimension); nothing else was missing.
