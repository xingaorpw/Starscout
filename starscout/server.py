"""Flask 服务：一个页面 + 一组 JSON API。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from . import collect as collector
from . import config, profile, rank, store
from .features import Vectorizer

RATE_ACTIONS = {"like", "dislike", "save", "skip"}

_collect_state: dict[str, Any] = {
    "running": False,
    "message": "",
    "added": 0,
    "error": None,
}


def _pipeline(db_path=None):
    """重建向量器 + 用户画像。数据量小（几千条），每次现算最简单也最不容易出错。"""
    repos = store.all_repos(db_path)
    vec = Vectorizer(config.FEATURE_DIM).fit(repos)
    by_id = {r["id"]: r for r in repos}
    events = store.all_events(db_path)
    prior_topics = store.kv_get("prior_topics", [], db_path) or []
    prior_languages = store.kv_get("prior_languages", [], db_path) or []
    w = profile.build(vec, by_id, events, prior_topics, prior_languages)

    # 只有"引导页勾选的"和"实际点赞过的"才算偏好，避免把噪声说成兴趣
    taste = profile.taste_summary(by_id, events)
    preferred_topics = {t.lower() for t in prior_topics}
    preferred_topics |= {t["name"].lower() for t in taste["topics"] if t["score"] > 0}
    preferred_languages = {l.lower() for l in prior_languages}
    preferred_languages |= {l["name"].lower() for l in taste["languages"] if l["score"] > 0}
    return repos, by_id, vec, events, w, preferred_topics, preferred_languages


def _card(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "name": row.get("name") or row["full_name"].split("/")[-1],
        "owner": row.get("owner") or row["full_name"].split("/")[0],
        "url": f"https://github.com/{row['full_name']}",
        "description": row.get("description") or "",
        "language": row.get("language"),
        "topics": (row.get("topics") or [])[:6],
        "stars": row.get("stars") or 0,
        "forks": row.get("forks") or 0,
        "pushed_at": row.get("pushed_at"),
        "created_at": row.get("created_at"),
        "license": row.get("license"),
        "quality": row.get("quality") or 0,
        "rank": row.get("rank"),
        "score": row.get("score"),
        "reason": row.get("reason") or "",
    }


def _stats(db_path=None) -> dict[str, Any]:
    events = store.all_events(db_path)
    by_id = {r["id"]: r for r in store.all_repos(db_path)}
    likes = sum(1 for e in events if e["action"] in ("like", "save"))
    dislikes = sum(1 for e in events if e["action"] == "dislike")
    saves = sum(1 for e in events if e["action"] == "save")
    today_events = [e for e in events if e["day"] == store.today()]
    return {
        "repos": store.count_repos(db_path),
        "ratings": len(events),
        "likes": likes,
        "dislikes": dislikes,
        "saves": saves,
        "today": len(today_events),
        "confidence": profile.confidence(events),
        "taste": profile.taste_summary(by_id, events),
        "days": len({e["day"] for e in events}),
    }


def create_app(db_path: Path | str | None = None) -> Flask:
    app = Flask(__name__)
    app.json.ensure_ascii = False
    store.init_db(db_path)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/state")
    def api_state():
        return jsonify(
            {
                "onboarded": bool(store.kv_get("onboarded", False, db_path)),
                "prior_topics": store.kv_get("prior_topics", [], db_path),
                "prior_languages": store.kv_get("prior_languages", [], db_path),
                "stats": _stats(db_path),
                "collect": _collect_state,
            }
        )

    @app.post("/api/onboard")
    def api_onboard():
        body = request.get_json(silent=True) or {}
        topics = [t for t in (body.get("topics") or []) if isinstance(t, str)][:8]
        languages = [l for l in (body.get("languages") or []) if isinstance(l, str)][:6]
        store.kv_set("prior_topics", topics, db_path)
        store.kv_set("prior_languages", languages, db_path)
        store.kv_set("onboarded", True, db_path)
        # 偏好变了，当天的榜单作废，下次取 feed 时按新画像重排
        store.clear_recommendations(store.today(), db_path)
        return jsonify({"ok": True, "stats": _stats(db_path)})

    @app.get("/api/feed")
    def api_feed():
        k = max(1, min(int(request.args.get("k", config.TOP_K)), 30))
        day = store.today()
        rated = store.rated_repo_ids(db_path)

        stored = [r for r in store.recommendations_for(day, db_path) if r["id"] not in rated]
        if len(stored) >= min(k, 3):
            return jsonify({"items": [_card(r) for r in stored[:k]], "fresh": False})

        repos, by_id, vec, events, w, pref_topics, pref_langs = _pipeline(db_path)
        if not repos:
            return jsonify({"items": [], "fresh": True, "empty": True})

        shown = {r["id"] for r in store.recommendations_for(day, db_path)}
        picks = rank.recommend(
            repos,
            w,
            vec,
            k=k,
            exclude=rated | shown,
            liked_vecs=rank.liked_vectors(vec, by_id, events),
            preferred_topics=pref_topics,
            preferred_languages=pref_langs,
        )
        offset = len(shown)
        store.save_recommendations(
            day,
            [
                {
                    "repo_id": it["repo"]["id"],
                    "rank": offset + it["rank"],
                    "score": it["score"],
                    "reason": it["reason"],
                }
                for it in picks
            ],
            db_path,
        )
        items = [
            _card(
                {
                    **it["repo"],
                    "rank": offset + it["rank"],
                    "score": it["score"],
                    "reason": it["reason"],
                }
            )
            for it in picks
        ]
        return jsonify({"items": [_card(r) for r in stored] + items, "fresh": True})

    @app.post("/api/rate")
    def api_rate():
        body = request.get_json(silent=True) or {}
        try:
            repo_id = int(body.get("repo_id"))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "repo_id 不合法"}), 400
        action = str(body.get("action", "")).strip()
        if action not in RATE_ACTIONS:
            return jsonify({"ok": False, "error": "action 不合法"}), 400

        store.add_event(repo_id, action, db_path)
        return jsonify({"ok": True, "stats": _stats(db_path)})

    @app.get("/api/stats")
    def api_stats():
        return jsonify(_stats(db_path))

    @app.post("/api/collect")
    def api_collect():
        if _collect_state["running"]:
            return jsonify({"ok": False, "error": "正在采集，请稍等"}), 409

        topics = store.kv_get("prior_topics", [], db_path) or []
        _collect_state.update(running=True, message="正在连接 GitHub…", added=0, error=None)

        def worker() -> None:
            try:
                repos = collector.collect(topics=topics, verbose=False)
                added = store.upsert_repos(repos, db_path)
                _collect_state.update(added=added, message=f"已更新 {added} 个仓库")
            except Exception as exc:  # noqa: BLE001 - 采集失败不该拖垮服务
                _collect_state.update(error=str(exc), message="采集失败")
            finally:
                _collect_state["running"] = False

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True})

    @app.get("/api/collect/status")
    def api_collect_status():
        return jsonify(_collect_state)

    return app


def main() -> None:
    app = create_app()
    app.run(host="127.0.0.1", port=8000, debug=False)
