"""Plan 011's decision rule over a sweep: cells on the approximate path (codes estimated at α=0),
the worst mean f32 recall and 5th percentile per exponent, and the choice."""
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
key = lambda r: (r["dataset"], r["filter"], r["per_mille"], r["k"])
base = {key(r): r for r in rows if r["exponent"] == 0}
approx = {k for k, r in base.items() if r["estimated"] > 0}
by = collections.defaultdict(dict)
for r in rows:
    if key(r) in approx:
        by[r["exponent"]][key(r)] = r
print(f"{len(approx)} approximate cells")
summary = {}
for x in sorted(by):
    cells = by[x]
    if len(cells) < len(approx):
        print(f"α={x}: incomplete ({len(cells)})"); continue
    worst = min(cells.values(), key=lambda r: r["recall_f32"])
    worst_p5 = min(c["p5_f32"] for c in cells.values())
    slow = {k: (c["p50_ms"] / base[k]["p50_ms"]) for k, c in cells.items()}
    summary[x] = (worst["recall_f32"], worst_p5)
    print(f"α={x}: worst mean {worst['recall_f32']:.4f} ({key(worst)}), worst p5 {worst_p5:.3f}, p50 vs α=0: median ×{sorted(slow.values())[len(slow)//2]:.2f}, max ×{max(slow.values()):.2f}")
ok = [x for x, (m, p) in summary.items() if m >= 0.99 and p >= 0.96]
if ok:
    choice = min(ok)
else:
    best = max(m for m, _ in summary.values())
    choice = min(x for x, (m, _) in summary.items() if m >= best - 0.002)
print("choice:", choice, "(meets the bar)" if ok else "(best worst cell)")
