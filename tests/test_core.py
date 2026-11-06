"""核心逻辑的单元测试。跑法：python -m unittest discover -s tests -v"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starscout import fixtures, profile, rank, store  # noqa: E402
from starscout.features import Vectorizer  # noqa: E402


def repo(rid, name, lang, topics, desc, stars=1000, created="2025-01-01", quality=0.5):
    return {
        "id": rid,
        "full_name": f"demo/{name}",
        "owner": "demo",
        "name": name,
        "description": desc,
        "language": lang,
        "topics": topics,
        "stars": stars,
        "forks": stars // 10,
        "open_issues": 3,
        "created_at": f"{created}T00:00:00Z",
        "pushed_at": "2026-11-01T00:00:00Z",
        "license": "MIT",
        "archived": False,
        "homepage": None,
        "source": "test",
        "quality": quality,
    }


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.repos = [
            repo(1, "agent-framework", "Python", ["llm", "agents"], "framework for building agents"),
            repo(2, "yolo-trainer", "Python", ["computer-vision", "yolo"], "train object detection models"),
            repo(3, "rust-editor", "Rust", ["editor"], "a fast code editor written in rust"),
        ]
        self.vec = Vectorizer(512).fit(self.repos)

    def test_vectors_are_l2_normalized(self):
        v, _ = self.vec.transform(self.repos[0])
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=6)

    def test_same_repo_same_vector(self):
        a, _ = self.vec.transform(self.repos[1])
        b, _ = self.vec.transform(self.repos[1])
        self.assertTrue(np.allclose(a, b))

    def test_similar_repo_closer_than_unrelated(self):
        base, _ = self.vec.transform(self.repos[0])
        near, _ = self.vec.transform(
            repo(9, "agent-tools", "Python", ["llm", "agents"], "more agent framework tooling")
        )
        far, _ = self.vec.transform(
            repo(10, "yolo-tools", "Python", ["computer-vision", "yolo"], "object detection trainer")
        )
        self.assertGreater(float(base @ near), float(base @ far))

    def test_quality_score_in_range(self):
        for r in fixtures.load():
            q = rank.quality_score(r)
            self.assertGreaterEqual(q, 0.0)
            self.assertLessEqual(q, 1.0)


class TestProfile(unittest.TestCase):
    def setUp(self):
        self.repos = [
            repo(1, "agent-framework", "Python", ["llm", "agents"], "framework for building llm agents"),
            repo(2, "llm-serving", "Python", ["llm", "inference"], "serve large language models fast"),
            repo(3, "game-engine", "C++", ["game-engine", "graphics"], "a 3d game engine with renderer"),
        ]
        self.by_id = {r["id"]: r for r in self.repos}
        self.vec = Vectorizer(1024).fit(self.repos)

    def test_liking_llm_raises_llm_score(self):
        events = [
            {"repo_id": 1, "action": "like", "created_at": "2026-11-01T00:00:00+00:00"},
            {"repo_id": 2, "action": "save", "created_at": "2026-11-02T00:00:00+00:00"},
            {"repo_id": 3, "action": "dislike", "created_at": "2026-11-03T00:00:00+00:00"},
        ]
        w = profile.build(self.vec, self.by_id, events)

        llm_new, _ = self.vec.transform(
            repo(7, "llm-router", "Python", ["llm"], "route prompts between language models")
        )
        game_new, _ = self.vec.transform(
            repo(8, "renderer", "C++", ["graphics", "game-engine"], "vulkan renderer for a game engine")
        )
        self.assertGreater(float(llm_new @ w), float(game_new @ w))

    def test_dislike_lowers_score(self):
        events = [
            {"repo_id": 3, "action": "dislike", "created_at": "2026-11-03T00:00:00+00:00"},
        ]
        w = profile.build(self.vec, self.by_id, events)
        game_new, _ = self.vec.transform(
            repo(8, "renderer", "C++", ["graphics", "game-engine"], "vulkan renderer for a game engine")
        )
        self.assertLess(float(game_new @ w), 0.0)

    def test_prior_topics_apply_without_events(self):
        w = profile.build(self.vec, self.by_id, [], prior_topics=["llm"])
        llm_new, _ = self.vec.transform(repo(7, "llm-router", "Python", ["llm"], "route prompts"))
        self.assertGreater(float(llm_new @ w), 0.0)

    def test_taste_summary_counts(self):
        events = [
            {"repo_id": 1, "action": "like", "created_at": "2026-11-01T00:00:00+00:00"},
            {"repo_id": 2, "action": "like", "created_at": "2026-11-01T00:00:00+00:00"},
            {"repo_id": 3, "action": "dislike", "created_at": "2026-11-01T00:00:00+00:00"},
        ]
        taste = profile.taste_summary(self.by_id, events)
        names = [t["name"] for t in taste["topics"]]
        self.assertIn("llm", names)
        self.assertLess(taste["topics"][0]["score"], 0) if names[0] == "game-engine" else None
        llm = next(t for t in taste["topics"] if t["name"] == "llm")
        self.assertGreater(llm["score"], 0)


class TestRanking(unittest.TestCase):
    def setUp(self):
        self.repos = [
            repo(1, "py-llm", "Python", ["llm"], "language model toolkit", stars=5000),
            repo(2, "py-agent", "Python", ["llm", "agents"], "agent framework", stars=4000),
            repo(3, "py-rag", "Python", ["rag", "llm"], "retrieval augmented generation", stars=3000),
            repo(4, "py-cv", "Python", ["computer-vision"], "vision models", stars=2000),
            repo(5, "rs-cli", "Rust", ["cli"], "command line tool", stars=1500),
            repo(6, "go-server", "Go", ["server"], "http server", stars=1200),
        ]
        self.by_id = {r["id"]: r for r in self.repos}
        self.vec = Vectorizer(1024).fit(self.repos)
        self.w = np.zeros(1024)

    def test_returns_k_items(self):
        picks = rank.recommend(self.repos, self.w, self.vec, k=4)
        self.assertEqual(len(picks), 4)
        self.assertEqual([p["rank"] for p in picks], [1, 2, 3, 4])

    def test_excludes_rated(self):
        picks = rank.recommend(self.repos, self.w, self.vec, k=6, exclude={1, 2})
        ids = {p["repo"]["id"] for p in picks}
        self.assertNotIn(1, ids)
        self.assertNotIn(2, ids)

    def test_language_cap(self):
        picks = rank.recommend(self.repos, self.w, self.vec, k=6, lang_cap=2)
        langs = [p["repo"]["language"] for p in picks]
        self.assertLessEqual(langs.count("Python"), 2)

    def test_reason_is_non_empty(self):
        picks = rank.recommend(self.repos, self.w, self.vec, k=3)
        for p in picks:
            self.assertTrue(p["reason"])


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "t.db"
        store.init_db(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_upsert_is_idempotent(self):
        store.upsert_repos([repo(1, "a", "Python", ["x"], "d")], self.path)
        store.upsert_repos([repo(1, "a", "Python", ["x"], "d2")], self.path)
        self.assertEqual(store.count_repos(self.path), 1)
        self.assertEqual(store.all_repos(self.path)[0]["description"], "d2")

    def test_same_day_rating_overwrites(self):
        store.upsert_repos([repo(1, "a", "Python", ["x"], "d")], self.path)
        rid = store.all_repos(self.path)[0]["id"]
        store.add_event(rid, "like", self.path)
        store.add_event(rid, "dislike", self.path)
        events = store.all_events(self.path)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "dislike")

    def test_kv_roundtrip(self):
        store.kv_set("prior_topics", ["llm", "rag"], self.path)
        self.assertEqual(store.kv_get("prior_topics", [], self.path), ["llm", "rag"])


class TestServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "web.db"
        from starscout.server import create_app

        store.init_db(self.path)
        store.upsert_repos(fixtures.load(), self.path)
        self.app = create_app(self.path)
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def test_state(self):
        r = self.client.get("/api/state")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(r.get_json()["stats"]["repos"], 10)

    def test_feed_returns_cards(self):
        r = self.client.get("/api/feed?k=5")
        self.assertEqual(r.status_code, 200)
        items = r.get_json()["items"]
        self.assertEqual(len(items), 5)
        self.assertTrue(items[0]["url"].startswith("https://github.com/"))

    def test_rate_updates_stats(self):
        items = self.client.get("/api/feed?k=2").get_json()["items"]
        rid = items[0]["id"]
        r = self.client.post("/api/rate", json={"repo_id": rid, "action": "like"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["stats"]["ratings"], 1)

    def test_rate_rejects_bad_action(self):
        r = self.client.post("/api/rate", json={"repo_id": 1, "action": "nuke"})
        self.assertEqual(r.status_code, 400)

    def test_index_renders(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("StarScout", r.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
