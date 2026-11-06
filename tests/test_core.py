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

    def test_df_never_exceeds_doc_count(self):
        # 维度故意设小，逼出哈希冲突：同一文档里两个 token 撞进同一个桶时，
        # 这个桶的 df 只能 +1，否则会出现「1 篇文档、df=2」
        vec = Vectorizer(8).fit([
            repo(1, "x", "Python", ["llm"], "alpha beta gamma delta epsilon zeta eta theta iota kappa")
        ])
        self.assertLessEqual(float(vec.df.max()), 1.0)
        self.assertEqual(vec.n_docs, 1)

    def test_df_counts_multiple_docs(self):
        vec = Vectorizer(64).fit([
            repo(1, "a", "Python", ["llm"], "agent framework"),
            repo(2, "b", "Python", ["llm"], "agent toolkit"),
        ])
        self.assertLessEqual(float(vec.df.max()), 2.0)

    def test_chinese_description_produces_tokens(self):
        # 只切 [a-z0-9] 的话，纯中文简介一个 token 都提取不到
        tokens = Vectorizer.tokens(
            repo(1, "中文项目", "Python", [], "一个用于检索增强生成的框架")
        )
        cjk = [t for t in tokens if any("\u4e00" <= ch <= "\u9fff" for ch in t)]
        self.assertTrue(cjk, f"没提取到中文 token：{tokens}")

    def test_chinese_bigram_shared_between_similar_descriptions(self):
        vec = Vectorizer(256).fit([
            repo(1, "a", "Python", [], "检索增强生成"),
            repo(2, "b", "Python", [], "向量检索"),
        ])
        near, _ = vec.transform(repo(3, "c", "Python", [], "检索增强"))
        far, _ = vec.transform(repo(4, "d", "Python", [], "图像分割"))
        self.assertGreater(float(near @ far), 0.0)


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

    def test_feedback_progress_caps_at_one(self):
        events = [{"repo_id": 1, "action": "like", "created_at": "2026-11-01T00:00:00+00:00"}] * 30
        self.assertEqual(profile.feedback_progress(events), 1.0)

    def test_feedback_progress_counts_skip_too(self):
        # skip 也算一条反馈——这正是它不能叫「置信度」的原因
        events = [{"repo_id": 1, "action": "skip", "created_at": "2026-11-01T00:00:00+00:00"}] * 5
        self.assertGreater(profile.feedback_progress(events), 0.0)


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

    def test_recent_velocity_overrides_lifetime_average(self):
        # 老仓库 + 昨天爆火：生命周期均值会严重低估，快照增量才对
        old = repo(1, "old", "Python", [], "d", stars=5000, created="2021-01-01")
        lifetime = rank.star_velocity(old)
        self.assertLess(lifetime, 5.0)

        hot = dict(old, recent_velocity=200.0)
        self.assertEqual(rank.star_velocity(hot), 200.0)
        self.assertTrue(rank.velocity_is_recent(hot))
        self.assertFalse(rank.velocity_is_recent(old))

    def test_quality_reacts_to_recent_velocity(self):
        old = repo(1, "old", "Python", ["llm"], "d", stars=5000, created="2021-01-01")
        base = rank.quality_score(old)
        hot = dict(old, recent_velocity=400.0)
        self.assertGreater(rank.quality_score(hot), base)

    def test_explain_says_recent_when_using_snapshot(self):
        item = {
            "repo": repo(1, "hot", "Python", ["llm"], "d", stars=5000, created="2021-01-01",
                         ) | {"recent_velocity": 300.0},
            "detail": [],
            "taste": 0.0,
            "quality": 0.3,
            "score": 0.3,
            "rank": 1,
        }
        reason = rank.explain(item, np.zeros(16))
        self.assertIn("最近日均涨星", reason)


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

    def test_repo_exists(self):
        store.upsert_repos([repo(1, "a", "Python", ["x"], "d")], self.path)
        rid = store.all_repos(self.path)[0]["id"]
        self.assertTrue(store.repo_exists(rid, self.path))
        self.assertFalse(store.repo_exists(999999, self.path))

    def test_single_snapshot_has_no_velocity(self):
        store.upsert_repos([repo(1, "a", "Python", ["x"], "d", stars=100)], self.path)
        self.assertEqual(store.recent_velocities(self.path), {})
        self.assertIsNone(store.all_repos(self.path)[0]["recent_velocity"])

    def test_recent_velocity_from_snapshots(self):
        from datetime import date

        store.upsert_repos([repo(1, "a", "Python", ["x"], "d", stars=900)], self.path)
        rid = store.all_repos(self.path)[0]["id"]
        store.record_stars(rid, 100, "2026-01-01", self.path)
        store.record_stars(rid, 400, "2026-01-03", self.path)

        span = (date.fromisoformat(store.today()) - date(2026, 1, 3)).days
        expected = (900 - 400) / span
        self.assertAlmostEqual(store.recent_velocities(self.path)[rid], expected, places=4)

    def test_all_repos_survives_missing_star_history(self):
        # 旧库没有这张表：读取路径要降级，不能整个崩掉
        store.upsert_repos([repo(1, "a", "Python", ["x"], "d")], self.path)
        with store.db(self.path) as conn:
            conn.execute("DROP TABLE star_history")
        repos = store.all_repos(self.path)
        self.assertEqual(len(repos), 1)
        self.assertIsNone(repos[0]["recent_velocity"])

    def test_negative_growth_clamped_to_zero(self):
        from datetime import date

        store.upsert_repos([repo(1, "a", "Python", ["x"], "d", stars=100)], self.path)
        rid = store.all_repos(self.path)[0]["id"]
        store.record_stars(rid, 5000, "2026-01-01", self.path)
        store.record_stars(rid, 5000, "2026-01-02", self.path)
        span = (date.fromisoformat(store.today()) - date(2026, 1, 2)).days
        self.assertGreater(span, 0)
        self.assertEqual(store.recent_velocities(self.path)[rid], 0.0)


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

    # ---- 写接口的防护 ----
    def test_onboard_rejects_non_json(self):
        r = self.client.post("/api/onboard", data="topics=llm", content_type="text/plain")
        self.assertEqual(r.status_code, 415)

    def test_onboard_rejects_json_array(self):
        r = self.client.post("/api/onboard", json=["llm", "rag"])
        self.assertEqual(r.status_code, 400)

    def test_onboard_rejects_cross_origin(self):
        r = self.client.post(
            "/api/onboard",
            json={"topics": ["llm"]},
            headers={"Origin": "https://evil.example"},
        )
        self.assertEqual(r.status_code, 403)

    def test_onboard_rejects_cross_site_fetch_metadata(self):
        r = self.client.post(
            "/api/onboard",
            json={"topics": ["llm"]},
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        self.assertEqual(r.status_code, 403)

    def test_bad_body_does_not_wipe_preferences(self):
        self.client.post("/api/onboard", json={"topics": ["llm", "rag"], "languages": ["Python"]})
        self.client.post("/api/onboard", data="junk", content_type="text/plain")
        self.client.post("/api/onboard", json=["llm"])
        state = self.client.get("/api/state").get_json()
        self.assertEqual(state["prior_topics"], ["llm", "rag"])
        self.assertEqual(state["prior_languages"], ["Python"])

    def test_collect_rejects_cross_origin(self):
        r = self.client.post("/api/collect", headers={"Origin": "https://evil.example"})
        self.assertEqual(r.status_code, 403)

    # ---- 评分校验 ----
    def test_rate_missing_repo_returns_404(self):
        r = self.client.post("/api/rate", json={"repo_id": 999999, "action": "like"})
        self.assertEqual(r.status_code, 404)
        self.assertEqual(self.client.get("/api/state").get_json()["stats"]["ratings"], 0)

    def test_rate_rejects_json_array(self):
        r = self.client.post("/api/rate", json=[1, 2, 3])
        self.assertEqual(r.status_code, 400)

    def test_rate_rejects_cross_origin(self):
        r = self.client.post(
            "/api/rate",
            json={"repo_id": 1, "action": "like"},
            headers={"Origin": "https://evil.example"},
        )
        self.assertEqual(r.status_code, 403)

    # ---- 榜单数量 ----
    def test_feed_rejects_bad_k(self):
        self.assertEqual(self.client.get("/api/feed?k=abc").status_code, 400)

    def test_feed_fills_up_to_k_after_small_request(self):
        first = self.client.get("/api/feed?k=3").get_json()["items"]
        self.assertEqual(len(first), 3)
        second = self.client.get("/api/feed?k=12").get_json()["items"]
        self.assertEqual(len(second), 12)

    def test_feed_never_exceeds_k(self):
        for k in (1, 3, 12, 30):
            items = self.client.get(f"/api/feed?k={k}").get_json()["items"]
            self.assertLessEqual(len(items), k, f"k={k}")

    def test_feed_after_rating_does_not_exceed_k(self):
        items = self.client.get("/api/feed?k=12").get_json()["items"]
        for it in items[:10]:
            self.client.post("/api/rate", json={"repo_id": it["id"], "action": "like"})
        again = self.client.get("/api/feed?k=12").get_json()["items"]
        self.assertLessEqual(len(again), 12)
        self.assertGreater(len(again), 0)

    def test_feed_does_not_repeat_rated_repos(self):
        items = self.client.get("/api/feed?k=5").get_json()["items"]
        rated = {it["id"] for it in items[:3]}
        for rid in rated:
            self.client.post("/api/rate", json={"repo_id": rid, "action": "dislike"})
        again = self.client.get("/api/feed?k=5").get_json()["items"]
        self.assertFalse(rated & {it["id"] for it in again})

    # ---- 指标命名 ----
    def test_stats_reports_progress_not_confidence(self):
        stats = self.client.get("/api/state").get_json()["stats"]
        self.assertIn("progress", stats)
        self.assertNotIn("confidence", stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
