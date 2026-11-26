"""hinny from Python: schemas, documents, text, vector and hybrid queries, filters, NumPy, threads."""

import tempfile
import subprocess
import sys
import threading
import unittest

import numpy as np

import hinny
from hinny import attr


class HinnyTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = hinny.Database(self.dir.name)
        self.ns = self.db.namespace("docs", text={"title": 2.0, "body": 1.0}, ints=["year"], strings=["kind"], vector=3)
        self.ns.upsert_many([
            {"key": 1, "text": {"title": "apartment lease", "body": "the lease for the apartment"}, "attrs": {"year": 2025, "kind": "pdf"}, "vector": [0.0, 0.0, 1.0]},
            {"key": 2, "text": {"title": "holiday photos", "body": "beach"}, "attrs": {"year": 2024, "kind": "jpg"}, "vector": np.array([1.0, 0.0, 0.0], dtype=np.float32)},
            {"key": 3, "text": {"title": "lease renewal", "body": "renewing it"}, "attrs": {"year": 2026, "kind": "pdf"}, "vector": np.array([0.9, 0.1, 0.0])},
            {"key": 4, "text": {"title": "groceries", "body": "milk"}, "attrs": {"year": 2023, "kind": "txt"}, "vector": (0.0, 1.0, 0.0)},
        ])

    def tearDown(self):
        self.dir.cleanup()

    def keys(self, hits):
        return [h.key for h in hits]

    def test_text_vector_hybrid(self):
        text = self.ns.query(text="apartment lease")
        self.assertEqual(self.keys(text)[:2], [1, 3])
        self.assertIsNotNone(text[0].text_score)
        self.assertIsNone(text[0].vector_score)

        near = self.ns.query(vector=np.array([1.0, 0.0, 0.0], dtype=np.float32), limit=2)
        self.assertEqual(self.keys(near), [2, 3])
        self.assertAlmostEqual(near[0].score, 1.0, places=5)

        # The best of each retriever and 3, good at both. (With two text matches, scaling gives the
        # weaker 0 from text; rank fusion counts its two good ranks.)
        both = self.ns.query(text="lease", vector=[1.0, 0.0, 0.0], limit=3)
        self.assertEqual(set(self.keys(both)), {1, 2, 3})
        self.assertIn(3, self.keys(self.ns.query(text="lease", vector=[1.0, 0.0, 0.0], limit=3, fusion="rrf"))[:2])
        three = next(h for h in both if h.key == 3)
        self.assertIsNotNone(three.text_score)
        self.assertIsNotNone(three.vector_score)
        self.assertEqual(three.attrs, {"year": 2026, "kind": "pdf"})

    def test_filters_and_order(self):
        pdfs = self.ns.query(text="lease", vector=[1.0, 0.0, 0.0], filter=attr("kind") == "pdf")
        self.assertTrue(all(h.attrs["kind"] == "pdf" for h in pdfs))
        self.assertEqual(self.keys(self.ns.query(filter=attr("year") >= 2024, order_by="-year")), [3, 1, 2])
        self.assertEqual(self.keys(self.ns.query(filter=attr("year").between(2024, 2025), order_by="year")), [2, 1])
        self.assertEqual(self.keys(self.ns.query(filter=(attr("kind") == "pdf") & ~(attr("year") < 2026))), [3])
        self.assertEqual(sorted(self.keys(self.ns.query(filter=attr("kind").one_of(["jpg", "txt"])))), [2, 4])
        self.assertEqual(self.keys(self.ns.query(filter=attr("kind").prefix("PD") & (attr("year") != 2026))), [1])
        with self.assertRaises(TypeError):
            bool(attr("year") >= 1)  # `and`/`or` would silently drop a filter

    def test_patterns(self):
        ns = self.db.namespace("files", text=["name"], strings=["path"], tags=["labels"])
        ns.upsert_many([
            {"key": 1, "text": "taxes", "attrs": {"path": "/home/mia/taxes/2025.pdf", "labels": ["Tax", "inbox"]}},
            {"key": 2, "text": "taxes", "attrs": {"path": "/home/leo/taxes/2024.PDF", "labels": ["family"]}},
            {"key": 3, "text": "photo", "attrs": {"path": "/home/mia/photos/2025.pdf.jpg"}},
        ])
        for where in ("memory", "segment"):
            keys = lambda f: sorted(self.keys(ns.query(filter=f)))
            self.assertEqual(keys(attr("path").glob("/home/*/taxes/*.pdf")), [1], where)
            self.assertEqual(keys(attr("path").iglob("/home/*/taxes/*.pdf")), [1, 2], where)
            self.assertEqual(keys(attr("path").glob("/home/???/taxes/20??.pdf")), [1], where)
            self.assertEqual(keys(attr("path").glob("*.pdf")), [1], where)  # the whole string
            self.assertEqual(keys(attr("path").glob("*.pdf*") & ~attr("labels").exists()), [3], where)
            self.assertEqual(keys(attr("labels").glob("T?x")), [1], where)
            self.assertEqual(keys(attr("labels").glob("t?x")), [], where)
            self.assertEqual(keys(attr("labels").iglob("t?x")), [1], where)
            ns.flush()

    def test_open_as_it_is(self):
        del self.ns
        ns = self.db.open("docs")
        self.assertEqual((len(ns), ns.schema["ints"], ns.schema["vector"]), (4, ["year"], 3))
        self.assertEqual(self.keys(ns.query(filter=attr("kind") == "jpg")), [2])
        with self.assertRaises(OSError):
            self.db.open("nothing")
        with self.assertRaises(ValueError):
            ns.upsert(9, attrs={"year": -2**63})  # the one integer an attribute can't be

    def test_fork(self):
        fork = self.ns.fork("sandbox")
        self.assertEqual((len(fork), fork.name, fork.schema), (4, "sandbox", self.ns.schema))
        self.assertEqual(self.keys(fork.query(text="lease")), self.keys(self.ns.query(text="lease")))
        # Apart from then on.
        fork.delete(1)
        fork.upsert(9, text={"title": "new in the fork"}, attrs={"year": 2030})
        self.assertEqual((len(fork), len(self.ns)), (4, 4))
        self.assertEqual(self.ns.get(1)["year"], 2025)
        self.assertIsNone(self.ns.get(9))
        self.ns.delete(2)
        self.assertEqual(fork.get(2)["kind"], "jpg")
        self.assertEqual(self.db.namespaces(), ["docs", "sandbox"])
        # What passes a filter, deleted at once: in the fork only.
        self.assertEqual(fork.delete_where(attr("kind") == "pdf"), 1)
        self.assertEqual(fork.delete_where(attr("kind") == "pdf"), 0)
        self.assertEqual((fork.count(attr("kind") == "pdf"), self.ns.count(attr("kind") == "pdf")), (0, 2))
        with self.assertRaises(TypeError):
            fork.delete_where(None)
        with self.assertRaises(FileExistsError):
            self.ns.fork("sandbox")
        with self.assertRaises(ValueError):
            self.ns.fork("not a name")

    def test_patch(self):
        ns = self.ns
        # One attribute changes; the text, the other attributes and the vector stay.
        self.assertTrue(ns.patch(1, attrs={"kind": "scan"}))
        self.assertEqual(ns.get(1), {"year": 2025, "kind": "scan"})
        self.assertEqual(ns.text(1), {"title": "apartment lease", "body": "the lease for the apartment"})
        self.assertEqual(self.keys(ns.query(vector=[0.0, 0.0, 1.0], limit=1, filter=attr("kind") == "scan")), [1])
        self.assertEqual(self.keys(ns.query(text="apartment", filter=attr("kind") == "scan")), [1])
        # Text, a vector, a version; and `remove` takes an attribute and a text field away.
        self.assertTrue(ns.patch(1, text={"body": "signed in May"}, vector=np.array([1.0, 0.0, 0.0], dtype=np.float32), version=5))
        self.assertEqual(ns.versions()[1], 5)
        self.assertEqual(self.keys(ns.query(text="signed")), [1])
        self.assertTrue(ns.patch(1, remove=["year", "body"]))
        self.assertEqual(ns.get(1), {"kind": "scan"})
        self.assertEqual(ns.text(1), {"title": "apartment lease"})
        self.assertEqual(ns.versions()[1], 5)
        # A key that isn't there: False, nothing written.
        self.assertFalse(ns.patch(99, attrs={"year": 1}))
        self.assertIsNone(ns.get(99))
        # With a condition: written when it holds, not when it doesn't.
        self.assertFalse(ns.patch(2, attrs={"year": 1999}, condition=attr("kind") == "pdf"))
        self.assertEqual(ns.get(2)["year"], 2024)
        self.assertTrue(ns.patch(2, attrs={"year": 1999}, condition=attr("kind") == "jpg"))
        self.assertEqual(ns.get(2)["year"], 1999)
        # Mistakes.
        with self.assertRaises(ValueError):
            ns.patch(2, attrs={"month": 3})
        with self.assertRaises(ValueError):
            ns.patch(2, remove=["nothing"])
        with self.assertRaises(ValueError):
            ns.patch(2, vector=[1.0, 2.0])
        with self.assertRaises(TypeError):
            ns.patch(2, text="a string")
        with self.assertRaises(TypeError):
            ns.patch(2, attrs={"year": [1]})
        with self.assertRaises(TypeError):
            ns.patch(2, condition="kind")
        self.assertEqual(ns.get(2)["year"], 1999)

    def test_patch_where(self):
        ns = self.ns
        self.assertEqual(ns.patch_where(attr("kind") == "pdf", attrs={"kind": "doc", "year": 2000}, remove=[]), 2)
        self.assertEqual(ns.patch_where(attr("kind") == "pdf", attrs={"year": 1}), 0)
        self.assertEqual((ns.count(attr("kind") == "doc"), ns.count(attr("year") == 2000), len(ns)), (2, 2, 4))
        self.assertEqual(ns.text(1)["title"], "apartment lease")
        self.assertEqual(ns.patch_where(attr("year") >= 2023, remove=["kind"]), 2)
        self.assertEqual((ns.count(attr("kind").exists()), ns.count(attr("kind") == "doc")), (2, 2))
        with self.assertRaises(TypeError):
            ns.patch_where(None, attrs={"year": 1})
        with self.assertRaises(ValueError):
            ns.patch_where(attr("year") > 0, attrs={"month": 1})

    def test_writes_with_a_condition(self):
        ns = self.ns
        newer = lambda rev: dict(text={"title": f"revision {rev}"}, attrs={"year": rev}, vector=[1.0, 1.0, 0.0])
        # Written where there is none; refused when the condition fails; written when it holds.
        self.assertTrue(ns.upsert_if(10, attr("year") < 5, **newer(5)))
        self.assertFalse(ns.upsert_if(10, attr("year") < 5, **newer(4)))
        self.assertEqual(ns.get(10), {"year": 5})
        self.assertEqual(ns.text(10)["title"], "revision 5")
        self.assertTrue(ns.upsert_if(10, attr("year") < 6, **newer(6), version=3))
        self.assertEqual((ns.get(10), ns.versions()[10]), ({"year": 6}, 3))
        self.assertEqual(len(ns), 5)
        # delete_if both ways; a key that isn't there is False.
        self.assertFalse(ns.delete_if(10, attr("year") == 5))
        self.assertIsNotNone(ns.get(10))
        self.assertTrue(ns.delete_if(10, attr("year") == 6))
        self.assertIsNone(ns.get(10))
        self.assertFalse(ns.delete_if(10, attr("year") == 6))
        with self.assertRaises(TypeError):
            ns.delete_if(1, None)
        with self.assertRaises(TypeError):
            ns.upsert_if(1, None)
        with self.assertRaises(ValueError):
            ns.upsert_if(1, attr("year") > 0, attrs={"month": 1})

    def test_typing(self):
        # The last word is still being typed: "lea" finds "lease", and not without `typing`.
        self.assertEqual(self.keys(self.ns.query(text="lea")), [])
        self.assertEqual(sorted(self.keys(self.ns.query(text="lea", typing=32))), [1, 3])
        self.assertEqual(self.keys(self.ns.query(text="lea", typing=0)), [])
        self.assertEqual(self.keys(self.ns.query(text="the apart", typing=32, all_words=True)), [1])
        self.assertEqual(self.keys(self.ns.query(text="lease ren", typing=32, all_words=True)), [3])
        self.assertEqual([h.key for h in self.ns.explain(text="renew", typing=32)["hits"]], [3])
        snap = self.ns.snapshot()
        self.assertEqual(self.keys(snap.query(text="lea")), [])
        self.assertEqual(sorted(self.keys(snap.query(text="lea", typing=32))), [1, 3])
        self.assertEqual(sorted(h.key for h in snap.explain(text="lea", typing=32)["hits"]), [1, 3])

    def test_delete_a_namespace(self):
        temp = self.db.namespace("temp", text="text")
        temp.upsert(1, text="short lived")
        snap = temp.snapshot()
        # Refused while a handle is open (the namespace, and a snapshot of it, count).
        with self.assertRaises(BlockingIOError):
            self.db.delete("temp")
        del temp
        with self.assertRaises(BlockingIOError):
            self.db.delete("temp")
        del snap
        self.assertEqual(self.db.namespaces(), ["docs", "temp"])
        self.db.delete("temp")
        self.assertEqual(self.db.namespaces(), ["docs"])
        with self.assertRaises(FileNotFoundError):
            self.db.open("temp")
        with self.assertRaises(FileNotFoundError):
            self.db.delete("temp")
        with self.assertRaises(ValueError):
            self.db.delete("../docs")
        self.assertEqual(len(self.ns), 4)
        # Made again, it starts empty.
        self.assertEqual(len(self.db.namespace("temp", text="text")), 0)
        # A fork is deleted apart from the original.
        fork = self.ns.fork("sandbox")
        del fork
        self.db.delete("sandbox")
        self.assertEqual((len(self.ns), self.db.namespaces()), (4, ["docs", "temp"]))

    def test_floats(self):
        ns = self.db.namespace("scored", text=["body"], floats=["score"], ints=["n"])
        self.assertEqual(ns.schema["floats"], ["score"])
        ns.upsert_many([
            {"key": 1, "text": "one", "attrs": {"score": 0.25, "n": 1}},
            {"key": 2, "text": "two", "attrs": {"score": 2, "n": 2}},      # an int for a float attribute is that number
            {"key": 3, "text": "three", "attrs": {"score": -1.5, "n": 3}},
            {"key": 4, "text": "four", "attrs": {"n": 4}},
        ])
        for where in ("memory", "segment"):
            self.assertEqual(ns.get(1), {"score": 0.25, "n": 1}, where)
            self.assertIsInstance(ns.get(2)["score"], float)
            self.assertEqual(self.keys(ns.query(filter=attr("score") >= 0.25, order_by="-score")), [2, 1], where)
            self.assertEqual(self.keys(ns.query(filter=attr("score") > 0.25)), [2], where)
            self.assertEqual(self.keys(ns.query(filter=attr("score") < 0, order_by="score")), [3], where)
            self.assertEqual(self.keys(ns.query(filter=attr("score").between(-2, 0.5), order_by="score")), [3, 1], where)
            self.assertEqual(self.keys(ns.query(filter=attr("score") == 2)), [2], where)
            self.assertEqual(self.keys(ns.query(filter=attr("score").one_of([0.25, -1.5, 7]), order_by="score")), [3, 1], where)
            self.assertEqual(self.keys(ns.query(filter=~attr("score").exists())), [4], where)
            # A fraction asked of integers: the integers in the range.
            self.assertEqual(self.keys(ns.query(filter=attr("n").between(1.5, 3.2), order_by="n")), [2, 3], where)
            self.assertEqual(self.keys(ns.query(text="one two three four", filter=attr("score") <= 0.25, order_by="score")), [3, 1], where)
            self.assertEqual(ns.count_by("score", filter=attr("n") <= 2), [(0.25, 1), (2.0, 1)], where)
            ns.flush()
        with self.assertRaises(ValueError):
            ns.upsert(5, text="five", attrs={"score": float("nan")})
        with self.assertRaises(ValueError):
            ns.upsert(5, text="five", attrs={"n": 0.5})

    def test_tags(self):
        ns = self.db.namespace("tagged", text=["body"], tags=["labels"], strings=["owner"])
        self.assertEqual(ns.schema["tags"], ["labels"])
        ns.upsert_many([
            {"key": 1, "text": "tax return", "attrs": {"labels": ["tax", "2025", "tax"], "owner": "mia"}},
            {"key": 2, "text": "tax letter", "attrs": {"labels": ("tax", "Inbox")}},
            {"key": 3, "text": "holiday", "attrs": {"labels": {"family"}}},
            {"key": 4, "text": "nothing", "attrs": {"labels": []}},
        ])
        for where in ("memory", "segment"):
            # A set: each once, in order.
            self.assertEqual(ns.get(1), {"labels": ["2025", "tax"], "owner": "mia"}, where)
            self.assertEqual(ns.get(4), {}, where)
            keys = lambda f: sorted(self.keys(ns.query(filter=f)))
            self.assertEqual(keys(attr("labels").has("tax")), [1, 2], where)
            self.assertEqual(keys(attr("labels") == "tax"), [1, 2], where)
            self.assertEqual(keys(attr("labels") != "tax"), [3, 4], where)
            self.assertEqual(keys(attr("labels").has_any(["family", "2025"])), [1, 3], where)
            self.assertEqual(keys(attr("labels").has_all(["tax", "Inbox"])), [2], where)
            self.assertEqual(keys(attr("labels").prefix("in")), [2], where)
            self.assertEqual(keys(~attr("labels").exists()), [4], where)
            self.assertEqual(self.keys(ns.query(text="tax", filter=attr("labels").has("Inbox"))), [2], where)
            self.assertEqual(ns.query(text="letter")[0].attrs["labels"], ["Inbox", "tax"], where)
            self.assertEqual(ns.count(attr("labels").has("tax")), 2, where)
            self.assertEqual(ns.count_by("labels", limit=1), [("tax", 2)], where)
            ns.flush()
        with self.assertRaises(TypeError):
            ns.upsert(5, text="five", attrs={"labels": [1, 2]})
        with self.assertRaises(ValueError):
            ns.upsert(5, text="five", attrs={"owner": ["a", "b"]})

    def test_counts(self):
        self.assertEqual((self.ns.count(), len(self.ns)), (4, 4))
        self.assertEqual(self.ns.count(attr("kind") == "pdf"), 2)
        self.assertEqual(self.ns.count(attr("year") > 2030), 0)
        self.assertEqual(self.ns.count_by("kind"), [("pdf", 2), ("jpg", 1), ("txt", 1)])
        self.assertEqual(self.ns.count_by("kind", limit=1), [("pdf", 2)])
        self.assertEqual(self.ns.count_by("year", filter=attr("kind") == "pdf"), [(2025, 1), (2026, 1)])
        self.assertEqual(self.ns.count_by("nothing"), [])
        # The same from segments, and a snapshot keeps its own.
        before = self.ns.snapshot()
        self.ns.flush()
        self.ns.delete(1)
        self.assertEqual((self.ns.count(), self.ns.count_by("kind")), (3, [("jpg", 1), ("pdf", 1), ("txt", 1)]))
        self.assertEqual((before.count(), before.count_by("kind", limit=1)), (4, [("pdf", 2)]))

    def test_documents(self):
        self.assertEqual(len(self.ns), 4)
        self.assertEqual(self.ns.get(2), {"year": 2024, "kind": "jpg"})
        self.ns.delete(2)
        self.assertIsNone(self.ns.get(2))
        self.assertNotIn(2, self.keys(self.ns.query(vector=[1.0, 0.0, 0.0])))
        self.ns.upsert(hinny.key("doc-17"), text={"title": "string ids"}, version=7)
        self.assertEqual(self.ns.versions()[hinny.key("doc-17")], 7)
        self.assertEqual(hinny.key("doc-17"), hinny.key("doc-17"))
        self.assertEqual(self.ns.schema["vector"], 3)

    def test_sync_without_flush_survives_process_exit(self):
        with tempfile.TemporaryDirectory() as path:
            # _exit bypasses destructors: recovery must use the synced WAL, not a final flush.
            script = """
import os, sys, hinny
ns = hinny.Database(sys.argv[1]).namespace("docs", text="body")
ns.upsert_many([{"key": 1, "text": "first"}, {"key": 2, "text": "second"}])
ns.delete(1)
ns.sync()
assert ns.stats()["segments"] == 0
os._exit(0)
"""
            subprocess.run([sys.executable, "-I", "-c", script, path], check=True, timeout=30)
            ns = hinny.Database(path).namespace("docs", text="body")
            self.assertEqual(set(ns.versions()), {2})
            self.assertEqual(ns.text(2), {"body": "second"})
            del ns

    def test_errors(self):
        with self.assertRaises(ValueError):
            self.ns.upsert(9, text={"nope": "x"})
        with self.assertRaises(ValueError):
            self.ns.upsert(9, vector=[1.0, 2.0])
        with self.assertRaises(ValueError):
            self.ns.upsert(9, attrs={"year": 2.5})  # a float for an integer attribute
        with self.assertRaises(TypeError):
            self.ns.upsert(9, attrs={"year": [2025]})
        with self.assertRaises(ValueError):
            self.ns.upsert(9, text="one field only")  # two text fields in this schema
        with self.assertRaises(ValueError):
            self.ns.query(vector=np.zeros((2, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            self.db.namespace("docs", text={"body": 1.0, "title": 2.0}, vector=3)

    def test_durable_and_reopened(self):
        before = self.keys(self.ns.query(text="lease", vector=[1.0, 0.0, 0.0], limit=3))
        del self.ns
        db = hinny.Database(self.dir.name)
        ns = db.namespace("docs", text={"title": 2.0, "body": 1.0}, ints=["year"], strings=["kind"], vector=3)
        self.assertEqual(self.keys(ns.query(text="lease", vector=[1.0, 0.0, 0.0], limit=3)), before)
        ns.flush()
        self.assertEqual(self.keys(ns.query(text="lease", vector=[1.0, 0.0, 0.0], limit=3)), before)
        self.ns = ns

    def test_optimize(self):
        for i in range(3):  # three segments
            self.ns.upsert(100 + i, text={"title": f"Extra {i}"}, attrs={"year": 2020}, vector=[0.0, 0.0, 1.0])
            self.ns.flush()
        before = self.keys(self.ns.query(text="lease", vector=[1.0, 0.0, 0.0], limit=5))
        self.assertGreater(self.ns.stats()["segments"], 1)
        self.ns.optimize()
        self.assertEqual(self.ns.stats()["segments"], 1)
        self.assertEqual(self.keys(self.ns.query(text="lease", vector=[1.0, 0.0, 0.0], limit=5)), before)

    def test_stored_text_and_snippets(self):
        filler = "Nothing about it here. " * 40
        body = f"{filler}Die Miete für März und die Wohnung in der Elm Street. {filler}"
        self.ns.upsert(10, text={"title": "Mietvertrag", "body": body}, attrs={"year": 2026})
        h = self.ns.query(text="miete märz wohnung", limit=1)[0]
        self.assertEqual(h.key, 10)
        self.assertEqual(h.text, {"title": "Mietvertrag", "body": body})
        self.assertEqual(h.snippet_field, "body")
        # Indices into the str, so non-ASCII text slices correctly.
        self.assertEqual([h.snippet[a:b] for a, b in h.highlights], ["Miete", "März", "Wohnung"])
        self.assertLessEqual(len(h.snippet), 245)
        self.assertNotIn("_fugu.text.0", h.attrs)
        self.assertEqual(self.ns.text(10)["title"], "Mietvertrag")
        lean = self.ns.query(text="miete", snippet=0, with_text=False, limit=1)[0]
        self.assertEqual((lean.text, lean.snippet, lean.highlights), ({}, None, []))
        bare = self.db.namespace("bare", text="body", store_text=False)
        bare.upsert(1, text="the lease")
        b = bare.query(text="lease")[0]
        self.assertEqual((b.text, b.snippet), ({}, None))

    def test_fusion(self):
        q = dict(text="lease", vector=[1.0, 0.0, 0.0], limit=4)
        self.assertEqual(self.ns.query(**q, text_weight=1.0)[0].key, self.keys(self.ns.query(text="lease", limit=1))[0])
        self.assertEqual(self.ns.query(**q, text_weight=0.0)[0].key, self.keys(self.ns.query(vector=[1.0, 0.0, 0.0], limit=1))[0])
        self.assertEqual(len(self.ns.query(**q, fusion="rrf")), 4)
        with self.assertRaises(ValueError):
            self.ns.query(**q, fusion="mean")

    def test_explain_and_candidate_budgets(self):
        r = self.ns.explain(text="lease", vector=[0.0, 0.0, 1.0], limit=3)
        self.assertEqual([h.key for h in r["hits"]], [h.key for h in self.ns.query(text="lease", vector=[0.0, 0.0, 1.0], limit=3)])
        f = r["fusion"]
        self.assertEqual((f["lexical"]["depth"], f["vector"]["depth"]), (200, 100))
        self.assertEqual(f["fused"], f["lexical"]["found"] + f["vector"]["found"] - f["both"])
        lo, hi = f["lexical"]["scores"]
        self.assertLessEqual(lo, hi)
        budgets = self.ns.explain(text="lease", vector=[0.0, 0.0, 1.0], limit=1, text_candidates=2, vector_candidates=3)["fusion"]
        self.assertEqual((budgets["lexical"]["depth"], budgets["vector"]["depth"]), (2, 3))
        self.assertIsNone(self.ns.explain(text="lease")["fusion"])
        self.assertIsNone(self.ns.snapshot().explain(vector=[0.0, 0.0, 1.0])["fusion"])

    def test_limits_and_timeouts(self):
        with self.assertRaises(TimeoutError):
            self.ns.query(text="lease", vector=[0.0, 0.0, 1.0], timeout=0)
        self.assertEqual(len(self.ns.query(text="lease", timeout=60)), 2)
        with self.assertRaises(ValueError):
            self.ns.query(text="lease", timeout=-1)
        with self.assertRaises(ValueError):
            self.ns.query(text=" ".join(f"w{i}" for i in range(5000)))
        self.ns.set_limits(max_batch_bytes=200)
        with self.assertRaises(ValueError):
            self.ns.upsert(9, text={"title": "x" * 500})
        self.assertNotIn(9, [h.key for h in self.ns.query(text="x" * 10, limit=10)])
        self.ns.set_limits(max_batch_bytes=1 << 30)
        self.ns.upsert(9, text={"title": "x" * 500})
        s = self.ns.stats()
        self.assertGreater(s["memtable_bytes"], 0)
        self.assertEqual(s["pinned_segments"], 0)

    def test_language(self):
        ns = self.db.namespace("en", text="body", language="english")
        ns.upsert(1, text="The tax returns were filed in March")
        ns.upsert(2, text="Running shoes for runners")
        self.assertEqual(self.keys(ns.query(text="taxed filings")), [1])
        h = ns.query(text="taxing", limit=1)[0]
        self.assertEqual([h.snippet[a:b] for a, b in h.highlights], ["tax"])
        self.assertEqual(self.keys(ns.query(text="the of and")), [])
        with self.assertRaises(ValueError):
            self.db.namespace("x", text="body", language="klingon")
        del ns, h
        with self.assertRaises(ValueError):
            self.db.namespace("en", text="body")  # analysis is fixed once there's data

    def test_text_that_isnt_unicode(self):
        # Lone surrogates (from JSON escapes, in real mail) are replaced, not refused.
        ns = self.db.namespace("u", text="body")
        ns.upsert(1, text="bad \ud83d surrogate lease")
        ns.upsert_many([{"key": 2, "text": {"body": "also \udc00 lease"}}])
        self.assertEqual(sorted(self.keys(ns.query(text="lease"))), [1, 2])
        self.assertIn("\ufffd", ns.text(1)["body"])

    def test_sparse(self):
        ns = self.db.namespace("sp", text="body", sparse=["terms"], vector=2)
        ns.upsert_many([
            {"key": 1, "text": "a lease", "sparse": {"terms": {"lease": 1.4, "rent": 0.6}}, "vector": [1.0, 0.0]},
            {"key": 2, "text": "a rent", "sparse": {"terms": {"rent": 1.0}}, "vector": [0.0, 1.0]},
        ])
        ns.upsert(3, text="nothing", sparse={"terms": {"beach": 2.0}}, vector=[0.7, 0.7])
        # Dot product of query and document weights: 1 → 2·1.4 + 1·0.6 = 3.4, 2 → 1·1.0.
        hits = ns.query(sparse={"terms": {"lease": 2.0, "rent": 1.0}})
        self.assertEqual(self.keys(hits), [1, 2])
        self.assertAlmostEqual(hits[0].text_score, 3.4, places=1)
        self.assertAlmostEqual(hits[1].text_score, 1.0, places=1)
        # Stays after a flush, and fuses with vectors.
        ns.flush()
        self.assertEqual(self.keys(ns.query(sparse={"terms": {"beach": 1.0}})), [3])
        self.assertEqual(self.keys(ns.query(sparse={"terms": {"rent": 1.0}}, vector=[0.0, 1.0], limit=1)), [2])
        # Sparse terms aren't words: text queries don't see them.
        self.assertEqual(self.keys(ns.query(text="beach")), [])
        with self.assertRaises(ValueError):
            ns.query(sparse={"body": {"lease": 1.0}})

    def test_metrics(self):
        # Against the query [1, 0]: by cosine the short [0.5, 0] in the query's direction comes
        # first and [3, 1] second; by Euclidean distance [1, 1] is nearer than [3, 1]; by dot
        # product the long [3, 1] comes first and the short one last.
        vectors = [{"key": 1, "vector": [0.5, 0.0]}, {"key": 2, "vector": [3.0, 1.0]}, {"key": 3, "vector": [1.0, 1.0]}]
        cases = {
            "cosine": [(1, 1.0), (2, 3 / 10 ** 0.5), (3, 1 / 2 ** 0.5)],
            "euclidean": [(1, -0.25), (3, -1.0), (2, -5.0)],  # minus the squared distance
            "dot": [(2, 3.0), (3, 1.0), (1, 0.5)],  # the product
        }
        for metric, want in cases.items():
            ns = self.db.namespace(metric, vector=2, metric=metric)
            self.assertEqual((ns.schema["vector"], ns.schema["metric"]), (2, metric))
            ns.upsert_many(vectors)
            hits = ns.query(vector=[1.0, 0.0])
            self.assertEqual(self.keys(hits), [k for k, _ in want], metric)
            for h, (_, score) in zip(hits, want):
                self.assertAlmostEqual(h.score, score, places=4)
            del ns
            self.assertEqual(self.db.open(metric).schema["metric"], metric)
            other = "euclidean" if metric == "dot" else "dot"
            with self.assertRaisesRegex(ValueError, "another schema"):
                self.db.namespace(metric, vector=2, metric=other)
        self.assertEqual(self.db.namespace("plain", vector=2).schema["metric"], "cosine")
        self.assertEqual(self.ns.schema["metric"], "cosine")
        self.assertIsNone(self.db.namespace("novector", text="body").schema["metric"])
        with self.assertRaisesRegex(ValueError, "cosine, euclidean or dot"):
            self.db.namespace("other", vector=2, metric="manhattan")
        with self.assertRaises(ValueError):
            self.db.namespace("other", text="body", metric="dot")

    def test_patch_with_sparse_terms(self):
        ns = self.db.namespace("sp", text="body", sparse=["terms"], ints=["year"])
        ns.upsert(1, text="a lease", sparse={"terms": {"lease": 1.4}}, attrs={"year": 2025})
        found = lambda: self.keys(ns.query(sparse={"terms": {"lease": 1.0}}))
        self.assertEqual(found(), [1])
        # The terms aren't kept as given: a patch that leaves them out is refused.
        with self.assertRaisesRegex(ValueError, "sparse"):
            ns.patch(1, attrs={"year": 2026})
        with self.assertRaisesRegex(ValueError, "sparse"):
            ns.patch_where(attr("year") == 2025, attrs={"year": 2026})
        self.assertEqual(ns.get(1)["year"], 2025)
        self.assertTrue(ns.patch(1, attrs={"year": 2026}, sparse={"terms": {"lease": 1.4}}))
        self.assertEqual((ns.get(1)["year"], found()), (2026, [1]))
        self.assertEqual(ns.patch_where(attr("year") == 2026, attrs={"year": 2027}, sparse={"terms": [("lease", 1.4)]}), 1)
        self.assertEqual((ns.get(1)["year"], found()), (2027, [1]))
        # Or that removes the field.
        self.assertTrue(ns.patch(1, attrs={"year": 2028}, remove=["terms"]))
        self.assertEqual((ns.get(1)["year"], found()), (2028, []))
        with self.assertRaises(TypeError):
            ns.patch(1, sparse=["lease"])

    def test_snapshots_and_atomic_batches(self):
        snap = self.ns.snapshot()
        self.ns.upsert(99, text={"title": "Later lease"}, attrs={"year": 2030})
        self.ns.delete(1)
        self.ns.flush()
        # The snapshot: as before the writes.
        self.assertEqual(sorted(snap.versions()), [1, 2, 3, 4])
        self.assertIsNotNone(snap.get(1))
        self.assertNotIn(99, self.keys(snap.query(text="lease", limit=10)))
        self.assertGreater(self.ns.snapshot().seq, snap.seq)
        # Now.
        self.assertIsNone(self.ns.get(1))
        self.assertIn(99, self.keys(self.ns.query(text="lease", limit=10)))
        # Upserts and deletes in one batch.
        before = self.ns.snapshot()
        self.assertEqual(self.ns.apply(upsert=[{"key": 300, "text": {"title": "moved lease"}}], delete=[3, 99]), 3)
        self.assertIsNone(self.ns.get(3))
        self.assertIsNotNone(self.ns.get(300))
        self.assertIsNotNone(before.get(3))
        self.assertEqual(self.ns.delete_many([2, 4]), 2)
        self.assertEqual(sorted(self.ns.versions()), [300])
        # A batch with a bad document changes nothing.
        with self.assertRaises(ValueError):
            self.ns.upsert_many([{"key": 200, "text": {"title": "ok"}}, {"key": 201, "text": {"nofield": "x"}}])
        self.assertIsNone(self.ns.get(200))

    def test_one_writer_process(self):
        import subprocess, sys
        code = f"import hinny; hinny.Database({self.dir.name!r}).namespace('docs', text={{'title': 2.0, 'body': 1.0}}, ints=['year'], strings=['kind'], vector=3)"
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already open for writing", r.stderr)

    def test_threads(self):
        ns = self.db.namespace("many", text="text", ints=["n"], vector=8)
        rng = np.random.default_rng(1)
        vecs = rng.standard_normal((2000, 8)).astype(np.float32)
        ns.upsert_many({"key": i, "text": f"item {i} of the batch", "attrs": {"n": i}, "vector": vecs[i]} for i in range(2000))
        errors = []

        def search():
            try:
                for i in range(200):
                    hits = ns.query(vector=vecs[i], limit=5)
                    assert hits[0].key == i, (hits[0].key, i)
                    # Hybrid too (every document has "batch", so the text side is uninformative
                    # here and only the concurrency is checked).
                    assert len(ns.query(text="batch", vector=vecs[i], limit=5)) == 5
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=search) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
