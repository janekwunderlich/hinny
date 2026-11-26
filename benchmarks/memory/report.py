"""Tables from run_lme.py / run_locomo.py output.

    python report.py RESULTS.jsonl [--labels paper|all] [--modes ...] [--by-type] [--timing]

Means over questions. "R@k" is recall_all@k (every evidence item in the top k: the paper's
"Recall"), "any@k" is recall_any@k, "N@10" is NDCG@10 with the paper's DCG. The ± after R@10
is a 95% normal-approximation half-width over questions.
"""

import argparse
import collections
import json
import math

import numpy as np

COLS = [("R@5", "recall_all@5"), ("N@5", "ndcg@5"), ("R@10", "recall_all@10"), ("R@20", "recall_all@20"),
        ("N@10", "ndcg@10"), ("any@5", "recall_any@5"), ("frac@10", "recall_frac@10")]
SCOLS = [("sR@5", "s_recall_all@5"), ("sR@10", "s_recall_all@10"), ("sR@20", "s_recall_all@20"),
         ("sN@10", "s_ndcg@10")]
MODE_ORDER = ["ref-bm25", "bm25-plain", "bm25", "ref-dot", "ref-cos", "vector", "vector-ip", "hybrid", "hybrid-rrf", "hybrid-ip", "hybrid-ip-rrf"]


def fmt(rows, cols, mkey):
    vals = []
    for _, key in cols:
        xs = [r[mkey][key] for r in rows if key in r[mkey]]
        vals.append(f"{np.mean(xs):.3f}" if xs else "–")
    return vals


def table(rows, labels, modes, title, only_qids=None):
    groups = collections.defaultdict(list)
    for r in rows:
        if only_qids is not None and r["qid"] not in only_qids:
            continue
        if modes and r["mode"] not in modes:
            continue
        groups[(r["unit"], r["key"], r["mode"])].append(r)
    out = [f"\n#### {title}\n"]
    has_s = any("s_recall_all@5" in g[0][labels] for g in groups.values())
    cols = COLS + (SCOLS if has_s else [])
    out.append("| unit | key | mode | n | " + " | ".join(c for c, _ in cols) + " |")
    out.append("|" + "---|" * (4 + len(cols)))
    for (unit, key, mode), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], MODE_ORDER.index(kv[0][2]) if kv[0][2] in MODE_ORDER else 99)):
        vals = fmt(g, cols, labels)
        p = np.mean([x[labels]["recall_all@10"] for x in g])
        vals[[c for c, _ in cols].index("R@10")] += f" ±{1.96 * math.sqrt(max(p * (1 - p), 1e-9) / len(g)):.3f}"
        out.append(f"| {unit} | {key} | {mode} | {len(g)} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def by_type(rows, labels, unit, key, modes, metric="recall_all@10"):
    types = sorted({r["qtype"] for r in rows})
    out = [f"\n#### {metric} by question type ({unit}, key={key}, labels={labels})\n",
           "| mode | " + " | ".join(f"{t} (n)" for t in types) + " |", "|" + "---|" * (1 + len(types))]
    for mode in [m for m in MODE_ORDER if m in modes]:
        cells = []
        for t in types:
            xs = [r[labels][metric] for r in rows if r["unit"] == unit and r["key"] == key
                  and r["mode"] == mode and r["qtype"] == t]
            cells.append(f"{np.mean(xs):.3f} ({len(xs)})" if xs else "–")
        if any(c != "–" for c in cells):
            out.append(f"| {mode} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def timing(path):
    rows = [json.loads(l) for l in open(path)]
    out = ["\n#### Hinny timings (one haystack per question; median of 5 runs per query)\n",
           "| unit | key | haystacks | items/haystack (median) | ingest ms p50 | ingest ms p99 | mode | query ms p50 | query ms p99 | cold p50 | cold p99 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["unit"], r["key"], r.get("vectors", True))].append(r)
    for (unit, key, vec), g in sorted(groups.items()):
        ing = np.array([r["ingest_en_s"] for r in g]) * 1000
        n = int(np.median([r["n"] for r in g]))
        modes = [m for m in ("bm25", "vector", "hybrid", "hybrid-rrf", "vector-ip", "hybrid-ip") if m in g[0].get("query_s", {})]
        for m in modes:
            q = np.array([r["query_s"][m] for r in g]) * 1000
            c = np.array([r["query_s"][m + ":cold"] for r in g]) * 1000
            out.append(f"| {unit} | {key}{'' if vec else ' (text only)'} | {len(g)} | {n} | {np.percentile(ing, 50):.2f} | {np.percentile(ing, 99):.2f} | {m} | "
                       f"{np.percentile(q, 50):.3f} | {np.percentile(q, 99):.3f} | {np.percentile(c, 50):.3f} | {np.percentile(c, 99):.3f} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--labels", default="paper")
    ap.add_argument("--modes", default="")
    ap.add_argument("--by-type", default="", help="unit:key[,unit:key]")
    ap.add_argument("--title", default="")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--qids-from", default="", help="restrict to the qids present in this results file")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.results)]
    rows = [r for r in rows if args.labels in r]
    modes = args.modes.split(",") if args.modes else MODE_ORDER
    only = None
    if args.qids_from:
        only = {json.loads(l)["qid"] for l in open(args.qids_from)}
    print(table(rows, args.labels, modes, args.title or f"{args.results} (labels={args.labels})", only))
    for c in filter(None, args.by_type.split(",")):
        unit, key = c.split(":")
        print(by_type([r for r in rows if only is None or r["qid"] in only], args.labels, unit, key, modes))
    if args.timing:
        print(timing(args.results.replace(".jsonl", ".timing.jsonl")))


if __name__ == "__main__":
    main()
