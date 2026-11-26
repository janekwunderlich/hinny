"""The README's tables from results.jsonl: nDCG@10, p50, Recall@100, and build time and size, with
the best per dataset and kind of search in bold. Prints Markdown.

    python3 tables.py [results.jsonl]          # current protocol only
    python3 tables.py --legacy [results.jsonl] # historical, mismatched request depths
"""

import argparse
import json
import sys

from run_bench import PROTOCOL, query_config

DATASETS = [("scifact", "SciFact"), ("nfcorpus", "NFCorpus"), ("arguana", "ArguAna"), ("scidocs", "SciDocs"), ("fiqa", "FiQA")]

# (kind, system, mode, label), in the tables' order.
ROWS = [
    ("Text", "hinny", "text", "hinny"),
    ("Text", "hinny (english)", "text", "hinny, English analyzer"),
    ("Text", "tantivy-py", "text", "Tantivy (tantivy-py)"),
    ("Text", "sqlite fts5 + sqlite-vec", "text", "SQLite (FTS5, sqlite-vec)"),
    ("Text", "lancedb", "text", "LanceDB"),
    ("Learned sparse", "hinny (sparse)", "sparse", "hinny, a sparse field"),
    ("Vector", "hinny", "vector", "hinny"),
    ("Vector", "sqlite fts5 + sqlite-vec", "vector", "SQLite (FTS5, sqlite-vec)"),
    ("Vector", "lancedb", "vector", "LanceDB"),
    ("Vector", "chroma", "vector", "Chroma"),
    ("Vector", "qdrant (local mode)", "vector", "Qdrant (local mode)"),
    ("Hybrid", "hinny", "hybrid", "hinny"),
    ("Hybrid", "hinny (english)", "hybrid", "hinny, English analyzer"),
    ("Hybrid", "hinny (sparse)", "sparse+vector", "hinny, sparse field + vectors"),
    ("Hybrid", "hinny (sparse)", "all three", "hinny, English + sparse + vectors"),
    ("Hybrid", "sqlite fts5 + sqlite-vec", "hybrid", "SQLite (FTS5, sqlite-vec)"),
    ("Hybrid", "lancedb", "hybrid", "LanceDB"),
]

BUILDS = [
    ("hinny", "hinny"),
    ("hinny (english)", "hinny, English analyzer"),
    ("hinny (sparse)", "hinny, English + sparse field"),
    ("tantivy-py", "Tantivy (tantivy-py)"),
    ("sqlite fts5 + sqlite-vec", "SQLite (FTS5, sqlite-vec)"),
    ("lancedb", "LanceDB"),
    ("chroma", "Chroma"),
    ("qdrant (local mode)", "Qdrant (local mode)"),
]


def ms(x):
    return f"{x:.2f} ms" if x < 10 else f"{x:.0f} ms"


def table(results, title, key, fmt, best):
    present = [r for r in ROWS if any((r[1], r[2], d) in results for d, _ in DATASETS)]
    out = [f"| {title} | " + " | ".join(n for _, n in DATASETS) + " |", "|---|" + "---|" * len(DATASETS)]
    for kind, system, mode, label in present:
        cells = []
        for d, _ in DATASETS:
            r = results.get((system, mode, d))
            if r is None or r.get(key) is None:
                cells.append("")
                continue
            peers = [results[(s, m, d)][key] for k, s, m, _ in present if k == kind and (s, m, d) in results and results[(s, m, d)].get(key) is not None]
            v = r[key]
            text = fmt(v)
            cells.append(f"**{text}**" if fmt(best(peers)) == text and len(peers) > 1 else text)
        out.append(f"| {kind}: {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def load_results(path, legacy=False):
    results = {}
    excluded = 0
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            r = json.loads(line)
            if "error" in r:
                continue
            selected = r.get("protocol") is None if legacy else r.get("protocol") == PROTOCOL
            if not selected:
                excluded += 1
                continue
            if not legacy and r.get("query_config") != query_config(r["system"], r["mode"]):
                raise ValueError(f"{path}:{line_number}: {PROTOCOL} query configuration is missing or incompatible")
            results[(r["system"], r.get("mode"), r["dataset"])] = r
    if not results:
        selected = "legacy" if legacy else PROTOCOL
        raise ValueError(f"No {selected} results in {path}; rerun the benchmark or use --legacy for historical rows")
    return results, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="results.jsonl")
    parser.add_argument("--legacy", action="store_true", help="show only unversioned historical rows")
    args = parser.parse_args()
    try:
        results, excluded = load_results(args.path, args.legacy)
    except ValueError as error:
        parser.error(str(error))
    if excluded:
        print(f"Excluded {excluded} rows from other protocols.", file=sys.stderr)
    if args.legacy:
        print("Legacy protocol: quality requested 100 results; latency requested 10. "
              "These are not paired quality/latency measurements.\n")
    else:
        print(f"Protocol {PROTOCOL}: nDCG@10 and latency request 11 results; "
              "Recall@100 separately requests 101. Self IDs are removed before trimming.\n")
    print(table(results, "nDCG@10" + (" (legacy request 100)" if args.legacy else " (request 11)"), "ndcg10", lambda v: f"{v:.3f}", max))
    print()
    print(table(results, "p50, request " + ("10 (legacy)" if args.legacy else "11, matching nDCG@10"), "p50_ms", ms, min))
    print()
    recall_title = "Recall@100 (legacy request 100)" if args.legacy else "Recall@100, separate request 101"
    print(table(results, recall_title, "recall100", lambda v: f"{v:.3f}", max))
    print()
    out = ["| Build, size | " + " | ".join(n for _, n in DATASETS) + " |", "|---|" + "---|" * len(DATASETS)]
    for system, label in BUILDS:
        cells = []
        for d, _ in DATASETS:
            r = next((v for (s, _, dd), v in results.items() if s == system and dd == d), None)
            cells.append("" if r is None else f"{r['build_s']:.1f} s, {r['size_mb']:.0f} MB")
        if any(cells):
            out.append(f"| {label} | " + " | ".join(cells) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
