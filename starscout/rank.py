"""排序：质量分 + 口味分 + 新鲜度，再用 MMR 保证一屏之内有变化。"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np

from . import config
from .features import Vectorizer


# --------------------------------------------------------------------------
# 质量分
# --------------------------------------------------------------------------
def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _days_since(ts: str | None, now: datetime) -> float:
    dt = _parse_ts(ts)
    if dt is None:
        return 9999.0
    return max((now - dt).total_seconds() / 86400.0, 0.0)


def _norm_log(x: float, cap: float) -> float:
    if x <= 0:
        return 0.0
    return min(1.0, math.log1p(x) / math.log1p(cap))


def quality_score(repo: dict[str, Any], now: datetime | None = None) -> float:
    """0~1 的"值得看"程度。star 绝对值只占四成，涨星速度同样重要。"""
    now = now or datetime.now(timezone.utc)
    stars = float(repo.get("stars") or 0)
    forks = float(repo.get("forks") or 0)
    velocity = star_velocity(repo, now)
    fresh = 1.0 - min(_days_since(repo.get("pushed_at"), now) / 120.0, 1.0)

    q = (
        0.38 * _norm_log(stars, config.STAR_CAP)
        + 0.22 * _norm_log(velocity, config.VELOCITY_CAP)
        + 0.10 * _norm_log(forks, config.FORK_CAP)
        + 0.10 * fresh
        + 0.06 * bool(repo.get("description"))
        + 0.06 * bool(repo.get("topics"))
        + 0.04 * bool(repo.get("license"))
        + 0.04 * (not repo.get("archived"))
    )
    return round(min(1.0, q), 4)


def star_velocity(repo: dict[str, Any], now: datetime | None = None) -> float:
    """日增 star。

    优先用近期快照算出的真实增量；没有快照才退回「总 star ÷ 存活天数」——
    后者是生命周期均值，会严重低估「存在五年、昨天刚爆火」的项目。
    """
    recent = repo.get("recent_velocity")
    if recent is not None:
        return max(0.0, float(recent))
    now = now or datetime.now(timezone.utc)
    age = max(_days_since(repo.get("created_at"), now), 1.0)
    return float(repo.get("stars") or 0) / age


def velocity_is_recent(repo: dict[str, Any]) -> bool:
    """涨星速度是不是来自近期快照（而不是生命周期均值）。"""
    return repo.get("recent_velocity") is not None


def repo_age_days(repo: dict[str, Any], now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    return _days_since(repo.get("created_at"), now)


# --------------------------------------------------------------------------
# 排序
# --------------------------------------------------------------------------
def _novelty(vec: np.ndarray, liked_vecs: Sequence[np.ndarray]) -> float:
    """跟已经收藏/点赞过的东西越不像，越新鲜。"""
    if not liked_vecs:
        return 0.5
    sims = [float(vec @ v) for v in liked_vecs]
    return float(max(0.0, 1.0 - max(sims)))


def recommend(
    candidates: Iterable[dict[str, Any]],
    w: np.ndarray,
    vectorizer: Vectorizer,
    k: int = config.TOP_K,
    exclude: Iterable[int] = frozenset(),
    liked_vecs: Sequence[np.ndarray] | None = None,
    lam: float = config.MMR_LAMBDA,
    lang_cap: int = config.MAX_PER_LANGUAGE,
    preferred_topics: Iterable[str] = frozenset(),
    preferred_languages: Iterable[str] = frozenset(),
) -> list[dict[str, Any]]:
    exclude = set(exclude)
    preferred_topics = {t.lower() for t in preferred_topics}
    preferred_languages = {l.lower() for l in preferred_languages}
    pool: list[dict[str, Any]] = []
    for repo in candidates:
        if int(repo["id"]) in exclude:
            continue
        vec, detail = vectorizer.transform(repo)
        pool.append(
            {
                "repo": repo,
                "vec": vec,
                "detail": detail,
                "taste": float(vec @ w),
                "quality": float(repo.get("quality") or quality_score(repo)),
            }
        )
    if not pool:
        return []

    tastes = np.array([p["taste"] for p in pool])
    mu = float(tastes.mean())
    sd = float(tastes.std()) or 1.0

    for p in pool:
        z = (p["taste"] - mu) / sd
        p["novelty"] = _novelty(p["vec"], liked_vecs or [])
        p["score"] = (
            config.W_TASTE * math.tanh(z)
            + config.W_QUALITY * p["quality"]
            + config.W_NOVELTY * p["novelty"]
        )

    # MMR：既看分数，也看跟已选内容的差异
    selected: list[dict[str, Any]] = []
    remaining = pool
    lang_count: Counter[str] = Counter()
    while len(selected) < k and remaining:
        best: dict[str, Any] | None = None
        best_val = -1e18
        for p in remaining:
            lang = (p["repo"].get("language") or "").lower()
            if lang and lang_count[lang] >= lang_cap:
                continue
            sim = max((float(p["vec"] @ s["vec"]) for s in selected), default=0.0)
            val = lam * p["score"] - (1.0 - lam) * sim
            if val > best_val:
                best, best_val = p, val
        if best is None:
            break
        selected.append(best)
        remaining = [x for x in remaining if x is not best]
        lang_count[(best["repo"].get("language") or "").lower()] += 1

    out = []
    for rank, item in enumerate(selected, start=1):
        out.append(
            {
                "repo": item["repo"],
                "score": round(item["score"], 4),
                "taste": round(item["taste"], 4),
                "quality": round(item["quality"], 4),
                "rank": rank,
                "reason": explain(item, w, preferred_topics, preferred_languages),
            }
        )
    return out


# --------------------------------------------------------------------------
# 解释
# --------------------------------------------------------------------------
def explain(
    item: dict[str, Any],
    w: np.ndarray,
    preferred_topics: set[str] = frozenset(),
    preferred_languages: set[str] = frozenset(),
) -> str:
    """一句话说明"为什么推给你"。

    关键约束：只有出现在用户口味画像里的 topic / 语言，才允许被说成"命中偏好"。
    否则小样本下逻辑回归会把描述里的稀有词（tailored、fancy）当成兴趣，理由就会很荒唐。
    """
    repo = item["repo"]
    topic_hits: list[tuple[str, float]] = []
    lang_hits: list[tuple[str, float]] = []

    for label, bucket, value in item["detail"]:
        weight = float(w[bucket])
        if weight <= 1e-6:
            continue
        contribution = weight * value
        if label.startswith("topic:"):
            name = label[6:]
            if name in preferred_topics:
                topic_hits.append((name, contribution))
        elif label.startswith("lang:"):
            name = label[5:]
            if name in preferred_languages:
                lang_hits.append((name, contribution))

    topic_hits.sort(key=lambda x: -x[1])
    lang_hits.sort(key=lambda x: -x[1])

    tags = [t for t, _ in topic_hits[:2]]
    if not tags and lang_hits:
        tags = [lang_hits[0][0]]

    parts: list[str] = []
    if tags:
        parts.append("命中你的偏好 " + " / ".join(tags))

    stars = int(repo.get("stars") or 0)
    age = repo_age_days(repo)
    velocity = star_velocity(repo)
    if age <= 90 and stars >= 300:
        parts.append(f"{int(age)} 天涨到 {stars:,} star")
    elif velocity >= 40:
        prefix = "最近日均涨星约" if velocity_is_recent(repo) else "日均涨星约"
        parts.append(f"{prefix} {velocity:.0f}")
    elif float(item["quality"]) >= 0.62:
        parts.append(f"综合质量 {float(item['quality']):.2f}")

    if not parts:
        parts.append("和你现有口味互补，值得看看")
    return "，".join(parts)


def liked_vectors(
    vectorizer: Vectorizer,
    repos: dict[int, dict[str, Any]],
    events: Iterable[dict[str, Any]],
) -> list[np.ndarray]:
    vecs = []
    for e in events:
        if e.get("action") not in ("like", "save"):
            continue
        repo = repos.get(int(e["repo_id"]))
        if repo is None:
            continue
        vec, _ = vectorizer.transform(repo)
        vecs.append(vec)
    return vecs
