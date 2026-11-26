"""Fast protocol checks; no databases, embeddings, or numpy required.

    python3 -m unittest discover -s benchmarks/beir -p 'test_*.py'
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_bench
import tables


def dataset():
    return {"ids": ["q"] + [f"d{i}" for i in range(1, 121)], "qids": ["q"],
            "qrels": {"q": {"d120": 1}}}


def current_row():
    return {"protocol": run_bench.PROTOCOL, "system": "hinny", "mode": "hybrid",
            "dataset": "scifact", "query_config": run_bench.query_config("hinny", "hybrid"),
            "ndcg10": 0.25, "recall100": 1.0, "p50_ms": 1.0, "p99_ms": 1.0,
            "build_s": 0, "size_mb": 0, "docs": 121, "queries": 1}


class RetrievalProtocolTests(unittest.TestCase):
    def test_quality_uses_shallow_ranking_and_timing_uses_identical_limit(self):
        calls = []

        def search(state, ds, i, mode, limit):
            calls.append((i, mode, limit))
            # A limit-sensitive hybrid engine ranks the relevant document first only
            # on the deeper request. Using that list for nDCG would wrongly score 1.
            return ([0, 120] + list(range(1, 100))) if limit == 101 else list(range(11))

        config = run_bench.query_config("hinny", "hybrid")
        with patch.object(run_bench.time, "perf_counter", side_effect=[1.0, 1.002]):
            result = run_bench.evaluate(None, dataset(), search, "hybrid", config)
        self.assertEqual(result["ndcg10"], 0.0)
        self.assertEqual(result["recall100"], 1.0)
        self.assertEqual(result["p50_ms"], 2.0)
        # Quality, deep recall, warm-up, timed query: only recall uses the deep limit.
        self.assertEqual(calls, [(0, "hybrid", k) for k in (11, 101, 11, 11)])
        self.assertEqual(result["timing"]["measured_queries"], 1)

    def test_self_exclusion_keeps_tenth_and_hundredth_nonself_results(self):
        ds = dataset()
        ds["qrels"] = {"q": {"d10": 1, "d100": 1}}

        def search(state, ds, i, mode, limit):
            return list(range(limit))

        result = run_bench.evaluate(None, ds, search, "text", run_bench.query_config("hinny", "text"))
        self.assertEqual(result["recall100"], 1.0)
        self.assertEqual(result["ndcg10"], round(run_bench.ndcg10([f"d{i}" for i in range(1, 11)], ds["qrels"]["q"]), 4))
        self.assertEqual(run_bench.exclude_self(ds, 0, list(range(11)), 10), [f"d{i}" for i in range(1, 11)])
        self.assertEqual(run_bench.exclude_self(ds, 0, list(range(1, 12)), 10), [f"d{i}" for i in range(1, 11)])

    def test_row_records_actual_profiles_and_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            systems = [("hinny", lambda ds, path: None, lambda state, ds, i, mode, k: list(range(k)), ["hybrid"])]
            with patch.object(run_bench, "load", return_value=dataset()), patch.object(run_bench, "SYSTEMS", systems), patch.object(run_bench, "OUT", Path(directory)), contextlib.redirect_stdout(io.StringIO()):
                row = run_bench.run("scifact")[0]
        self.assertEqual(row["protocol"], "beir-v2")
        shallow = row["query_config"]["quality_and_latency"]
        deep = row["query_config"]["recall100"]
        self.assertEqual((shallow["request_limit"], deep["request_limit"]), (11, 101))
        self.assertEqual(shallow["parameters"]["lexical_candidate_depth"], 200)
        self.assertEqual(shallow["parameters"]["vector_candidate_depth"], 100)
        self.assertEqual(deep["parameters"]["lexical_candidate_depth"], 200)
        self.assertEqual(deep["parameters"]["vector_candidate_depth"], 101)
        self.assertEqual(row["timing"]["scope"], "search_adapter_only")
        provenance = row["provenance"]
        self.assertEqual(set(provenance), {"commit", "uncommitted_changes", "packages", "python", "machine", "data", "conditions"})
        self.assertIn("sqlite", provenance["packages"])
        self.assertIn("cores", provenance["machine"])


class TableProtocolTests(unittest.TestCase):
    def write_rows(self, directory, rows):
        path = Path(directory) / "results.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_legacy_cannot_overwrite_current_table_row(self):
        current = current_row()
        legacy = {k: v for k, v in current.items() if k not in ("protocol", "query_config")}
        legacy["ndcg10"] = 0.99
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(directory, [current, legacy])
            rows, excluded = tables.load_results(path)
            self.assertEqual(next(iter(rows.values()))["ndcg10"], 0.25)
            self.assertEqual(excluded, 1)
            old_rows, excluded = tables.load_results(path, legacy=True)
            self.assertEqual(next(iter(old_rows.values()))["ndcg10"], 0.99)
            self.assertEqual(excluded, 1)

    def test_legacy_only_requires_explicit_opt_in(self):
        row = current_row()
        del row["protocol"]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(directory, [row])
            with self.assertRaisesRegex(ValueError, "No beir-v2 results"):
                tables.load_results(path)

    def test_version_without_matching_profile_is_rejected(self):
        for change in (lambda row: row.pop("query_config"),
                       lambda row: row["query_config"]["quality_and_latency"].update(request_limit=100)):
            row = current_row()
            change(row)
            with tempfile.TemporaryDirectory() as directory:
                path = self.write_rows(directory, [row])
                with self.assertRaisesRegex(ValueError, "configuration is missing or incompatible"):
                    tables.load_results(path)

    @unittest.skipUnless(shutil.which("node"), "node is required for the website generator check")
    def test_website_rejects_mixed_protocols_and_preserves_selected_metadata(self):
        generator = Path(__file__).resolve().parents[2] / "website/scripts/beir-data.mjs"
        row = current_row()
        legacy = copy.deepcopy(row)
        legacy.pop("protocol")
        legacy.pop("query_config")
        script = """
import assert from 'node:assert/strict';
const {selectResults} = await import(process.argv[1]);
const [current, legacy] = JSON.parse(process.argv[2]);
assert.throws(() => selectResults([current, legacy]), /multiple protocols/);
assert.deepEqual(selectResults([legacy, current], 'beir-v2')[0].query_config, current.query_config);
const old = selectResults([legacy])[0];
assert.equal(old.protocol, 'legacy-unversioned');
assert.equal(old.query_config.ndcg10.request_limit, 100);
assert.equal(old.query_config.latency.request_limit, 10);
assert.equal(old.ndcg10, legacy.ndcg10);
assert.throws(() => selectResults([{...current, query_config: null}]), /Invalid beir-v2/);
"""
        subprocess.run(["node", "--input-type=module", "-e", script, generator.as_uri(), json.dumps([row, legacy])], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
