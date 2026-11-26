"""Shared pieces of the conversational-memory retrieval benchmark.

Corpus construction and metrics follow LongMemEval's own code
(github.com/xiaowu0162/LongMemEval, src/retrieval/run_retrieval.py and eval_utils.py) so the
numbers line up with the paper's; where this benchmark adds a variant (full-session keys,
labels that include assistant-side evidence) it is named and kept separate.
"""

import hashlib
import json
import os
import random
from datetime import datetime

import numpy as np

WORK = os.environ.get(
    "MEMBENCH_WORK",
    "/home/janek/.t3/scratch/2026-11-13-you-are-taking-over-a-d7cb416e/bench/memory",
)
DATA = os.path.join(WORK, "data")
S_CLEANED = os.path.join(DATA, "lme", "longmemeval_s_cleaned.json")
S_ORIG = os.path.join(DATA, "lme", "longmemeval_s.json")
M_ORIG = os.path.join(DATA, "lme", "longmemeval_m.json")

SEED = 0


def sha(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load(path):
    with open(path) as f:
        return json.load(f)


def retrieval_questions(data):
    """The paper always skips the 30 abstention questions for retrieval."""
    return [x for x in data if "_abs" not in x["question_id"]]


def subset_order(data):
    """A fixed random order of the non-abstention question ids; any prefix is the subset."""
    ids = sorted(x["question_id"] for x in retrieval_questions(data))
    random.Random(SEED).shuffle(ids)
    return ids


def parse_date(s):
    # "2023/05/20 (Sat) 02:32"
    return int(datetime.strptime(s[:10] + s[-6:], "%Y/%m/%d %H:%M").timestamp())


# ---------------------------------------------------------------------------- corpus
#
# A corpus item: (id, key_text, attrs). Ids carry the label the way the paper's code does:
# evidence items contain "answer", others do not ("noans" for an evidence session whose
# evidence is not on the side being indexed).

def corpus(entry, unit, key="user"):
    """unit: "session" or "round". key: "user" (the paper: user-side text only) or "full"
    (user and assistant text). Returns items [(id, text, attrs)] and two label sets:
    paper_labels (the paper's code: evidence on the user side only) and all_labels (any
    has_answer turn, either side; for sessions, answer_session_ids)."""
    items, paper, full = [], set(), set()
    for sid, sess, date in zip(entry["haystack_session_ids"], entry["haystack_sessions"],
                               entry["haystack_dates"]):
        ts = parse_date(date)
        is_ans = "answer" in sid
        if unit == "session":
            turns = sess if key == "full" else [t for t in sess if t["role"] == "user"]
            sep = "\n" if key == "full" else " "
            text = sep.join(t["content"] for t in turns)
            items.append((sid, text, {"ts": ts, "session": sid, "round": -1}))
            if is_ans:
                full.add(sid)
                if any(t.get("has_answer") for t in sess if t["role"] == "user"):
                    paper.add(sid)
        else:
            # A round is one user message and the assistant reply after it. Its id is the
            # paper's turn id: session id + "_" + 1-based index of the user message.
            for i, t in enumerate(sess):
                if t["role"] != "user":
                    continue
                reply = sess[i + 1] if i + 1 < len(sess) and sess[i + 1]["role"] == "assistant" else None
                rid = f"{sid}_{i + 1}"
                text = t["content"] if key == "user" or reply is None else t["content"] + "\n" + reply["content"]
                items.append((rid, text, {"ts": ts, "session": sid, "round": i + 1}))
                if is_ans:
                    if t.get("has_answer"):
                        paper.add(rid)
                    if t.get("has_answer") or (reply is not None and reply.get("has_answer")):
                        full.add(rid)
    return items, paper, full


def session_of(item_id):
    return "_".join(item_id.split("_")[:-1])


# ---------------------------------------------------------------------------- metrics
#
# Exactly eval_utils.py: binary relevance; DCG = rel[0] + sum(rel[i] / log2(i + 1)) for the
# 1-based position i >= 2; the ideal DCG uses every relevant item in the corpus; "recall" in
# the paper's tables is recall_all (every evidence item within the top k).

def _dcg(rels, k):
    rels = np.asarray(rels, dtype=float)[:k]
    if rels.size:
        return rels[0] + np.sum(rels[1:] / np.log2(np.arange(2, rels.size + 1)))
    return 0.0


def evaluate(ranked_ids, labels, n_relevant_in_corpus, k):
    top = ranked_ids[:k]
    found = set(top) & labels
    recall_any = float(len(found) > 0)
    recall_all = float(all(l in set(top) for l in labels))  # all([]) is 1.0, as in the paper
    ideal = _dcg([1] * n_relevant_in_corpus, k)
    ndcg = 0.0 if ideal == 0 else _dcg([1 if i in labels else 0 for i in top], k) / ideal
    return recall_any, recall_all, ndcg


KS = (5, 10, 20)


def metrics(ranked_ids, labels, all_ids):
    """Recall_all@5/10/20, recall_any@5/10/20, NDCG@10 (and @5) for one ranking."""
    n_rel = sum(1 for i in all_ids if i in labels)
    out = {}
    for k in KS + (50,):
        ra, rl, nd = evaluate(ranked_ids, labels, n_rel, k)
        out[f"recall_any@{k}"] = ra
        out[f"recall_all@{k}"] = rl
        out[f"ndcg@{k}"] = nd
        if labels:
            out[f"recall_frac@{k}"] = len(set(ranked_ids[:k]) & labels) / len(labels)
    return out


def metrics_as_sessions(ranked_ids, labels, all_ids):
    out = {}
    for k in KS:
        ra, rl, nd = evaluate_as_sessions_full(ranked_ids, labels, all_ids, k)
        out[f"s_recall_any@{k}"] = ra
        out[f"s_recall_all@{k}"] = rl
        out[f"s_ndcg@{k}"] = nd
    return out


def evaluate_as_sessions_full(ranked_ids, labels, all_ids, k):
    """eval_utils.evaluate_retrieval_turn2session, reproduced: the paper ranks the whole
    corpus, maps each round to its session, and widens k until the top holds k distinct
    sessions. In its ids a round of an evidence session that is not itself evidence is
    renamed ("noans"), so it maps to a different session id and does not count as a hit,
    and the ideal DCG counts one relevant entry per evidence round."""
    seen = set(ranked_ids)
    full_rank = list(ranked_ids) + [i for i in all_ids if i not in seen]
    sess_labels = {session_of(l) for l in labels}

    def key(i):
        s = session_of(i)
        return s if (i in labels or s not in sess_labels) else "noans:" + s

    sess_rank = [key(i) for i in full_rank]
    eff = k
    while eff <= len(sess_rank) and len(set(sess_rank[:eff])) < k:
        eff += 1
    n_rel = sum(1 for i in all_ids if i in labels)
    return evaluate(sess_rank, sess_labels, n_rel, eff)


# ---------------------------------------------------------------------------- embeddings

EMB_DIR = os.path.join(WORK, "emb", "contriever")


class EmbeddingCache:
    """sha1(text) -> float32[768], stored as npz shards under EMB_DIR."""

    def __init__(self, path=EMB_DIR):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.map = {}
        for name in sorted(os.listdir(path)):
            if name.endswith(".npz"):
                z = np.load(os.path.join(path, name))
                for h, v in zip(z["h"], z["v"]):
                    self.map[str(h)] = v

    def __contains__(self, text):
        return sha(text) in self.map

    def get(self, text):
        return self.map[sha(text)]

    def add(self, texts, vecs):
        hs = [sha(t) for t in texts]
        n = len([x for x in os.listdir(self.path) if x.endswith(".npz")])
        tmp = os.path.join(self.path, f"shard_{n:05d}.tmp.npz")
        np.savez(tmp, h=np.array(hs), v=np.asarray(vecs, dtype=np.float32))
        os.replace(tmp, os.path.join(self.path, f"shard_{n:05d}.npz"))
        for h, v in zip(hs, vecs):
            self.map[h] = v
