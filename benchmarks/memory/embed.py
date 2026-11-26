"""Embed LongMemEval texts with Contriever (facebook/contriever), cached on disk.

The encoding is the paper's (run_retrieval.py, flat-contriever): the tokenizer's own
truncation (512 tokens), mean pooling over the attention mask, no normalization. Texts are
deduplicated by content and embedded in length-sorted batches; each block of questions is
committed to the cache as it finishes, so the job can be stopped and resumed.

Questions are taken in the fixed random order of common.subset_order, so the questions
embedded after any amount of time are a random subset.

    taskset -c 0-3 nice -n 19 python embed.py --questions 100 --hours 2
"""

import argparse
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer

import common


def texts_for(entry, parts):
    out = [entry["question"]]
    for unit, key in parts:
        items, _, _ = common.corpus(entry, unit, key)
        out += [t for _, t, _ in items]
    return out


def encode(tok, model, texts, batch=16):
    """The paper's Contriever encoding: mean pooling over the mask, unnormalized."""
    import numpy as np
    vecs = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            inp = tok(texts[i:i + batch], padding=True, truncation=True, return_tensors="pt")
            out = model(**inp)[0]
            mask = inp["attention_mask"]
            out = out.masked_fill(~mask[..., None].bool(), 0.0)
            vecs.append((out.sum(dim=1) / mask.sum(dim=1)[..., None]).numpy())
    return np.concatenate(vecs)


def load_model(threads=4):
    torch.set_num_threads(threads)
    return (AutoTokenizer.from_pretrained("facebook/contriever"),
            AutoModel.from_pretrained("facebook/contriever").eval())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=common.S_CLEANED)
    ap.add_argument("--questions", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--parts", default="session:user,round:user,session:full")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    parts = [tuple(p.split(":")) for p in args.parts.split(",")]

    tok, model = load_model(args.threads)
    cache = common.EmbeddingCache()
    data = common.load(args.data)
    by_id = {x["question_id"]: x for x in data}
    order = common.subset_order(data)[args.start:args.start + args.questions]

    # Every question text first (all 500: cheap), then haystacks block by block.
    blocks = [[x["question"] for x in data]]
    for b in range(0, len(order), args.block):
        texts = []
        for qid in order[b:b + args.block]:
            texts += texts_for(by_id[qid], parts)
        blocks.append(texts)

    deadline = time.time() + args.hours * 3600
    t0 = time.time()
    done = 0
    for bi, texts in enumerate(blocks):
        todo = sorted({t for t in texts if t not in cache}, key=len)
        if not todo:
            continue
        if time.time() > deadline:
            print(f"deadline reached before block {bi}", flush=True)
            break
        cache.add(todo, encode(tok, model, todo, args.batch))
        done += len(todo)
        el = time.time() - t0
        print(f"block {bi}/{len(blocks) - 1}: +{len(todo)} texts, {done} total, "
              f"{el / 60:.1f} min, {done / el:.2f} texts/s", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    sys.exit(main())
