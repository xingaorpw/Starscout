"""用户画像：把"喜欢/不喜欢"变成一个权重向量。

做法很朴素，但足够解释：
1. 先验——你在引导页勾选的 topic / 语言，直接在对应维度上加分；
2. 在线学习——把每条评分当成一个二分类样本，用 SGD 逻辑回归拟合；
3. 时间衰减——三个月前的口味不该和昨天一样重。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

from . import config
from .features import Vectorizer

# 评分动作 -> (是否正样本, 样本权重)
ACTIONS: dict[str, tuple[bool, float]] = {
    "like": (True, 1.0),
    "save": (True, 1.6),
    "dislike": (False, 1.0),
    "skip": (False, 0.25),
}

TASTE_WEIGHT = {"like": 1.0, "save": 1.6, "dislike": -1.0, "skip": 0.0}


def recency_weight(iso: str | None, halflife: float = config.RECENCY_HALFLIFE_DAYS) -> float:
    if not iso:
        return 0.6
    try:
        t = datetime.fromisoformat(iso)
    except ValueError:
        return 0.6
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    days = max((datetime.now(timezone.utc) - t).total_seconds() / 86400.0, 0.0)
    return max(0.35, 0.5 ** (days / halflife))


def _sigmoid(z: np.ndarray | float):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def train_logreg(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    w0: np.ndarray,
    epochs: int = 220,
    lr: float = 0.30,
    l2: float = 1e-3,
    seed: int = 7,
) -> np.ndarray:
    """小样本友好的 SGD 逻辑回归。样本只有几十条时也能收敛。"""
    w = w0.astype(np.float64).copy()
    n = len(X)
    if n == 0:
        return w
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    for _ in range(epochs):
        rng.shuffle(order)
        for i in order:
            x = X[i]
            p = float(_sigmoid(x @ w))
            grad = (p - float(y[i])) * float(sample_weight[i])
            w -= lr * (grad * x + l2 * w)
    return w


def build(
    vectorizer: Vectorizer,
    repos: dict[int, dict[str, Any]],
    events: Iterable[dict[str, Any]],
    prior_topics: Iterable[str] = (),
    prior_languages: Iterable[str] = (),
    dim: int | None = None,
) -> np.ndarray:
    """由评分事件训练出用户权重向量 w。"""
    w = np.zeros(dim or vectorizer.dim, dtype=np.float64)
    for t in prior_topics:
        w[vectorizer.bucket(f"topic:{str(t).strip().lower()}")] += config.PRIOR_TOPIC_WEIGHT
    for lang in prior_languages:
        w[vectorizer.bucket(f"lang:{str(lang).strip().lower()}")] += config.PRIOR_LANG_WEIGHT

    X, y, sw = [], [], []
    for e in events:
        repo = repos.get(int(e["repo_id"]))
        if repo is None:
            continue
        action = e.get("action", "")
        if action not in ACTIONS:
            continue
        positive, base = ACTIONS[action]
        vec, _ = vectorizer.transform(repo)
        X.append(vec)
        y.append(1.0 if positive else 0.0)
        sw.append(base * recency_weight(e.get("created_at")))

    if X:
        w = train_logreg(np.vstack(X), np.asarray(y), np.asarray(sw), w)
    return w


def taste_summary(
    repos: dict[int, dict[str, Any]],
    events: Iterable[dict[str, Any]],
    top_n: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """给界面用的可读画像：哪些 topic / 语言在加分，哪些在减分。"""
    topics: dict[str, float] = defaultdict(float)
    languages: dict[str, float] = defaultdict(float)

    for e in events:
        repo = repos.get(int(e["repo_id"]))
        if repo is None:
            continue
        delta = TASTE_WEIGHT.get(e.get("action", ""), 0.0) * recency_weight(e.get("created_at"))
        if delta == 0.0:
            continue
        for t in repo.get("topics") or []:
            topics[t] += delta
        if repo.get("language"):
            languages[repo["language"]] += delta

    def _rank(d: dict[str, float]) -> list[dict[str, Any]]:
        items = [{"name": k, "score": round(v, 2)} for k, v in d.items()]
        items.sort(key=lambda x: (-x["score"], x["name"]))
        pos = [i for i in items if i["score"] > 0][:top_n]
        neg = [i for i in items if i["score"] < 0][-3:][::-1]
        return pos + neg

    return {"topics": _rank(topics), "languages": _rank(languages)}


def confidence(events: Iterable[dict[str, Any]]) -> float:
    """0~1，告诉用户"画像学得有多准"。20 条评分大概到 0.8。"""
    n = sum(1 for e in events if e.get("action") in ACTIONS)
    return round(min(1.0, n / 25.0), 2)
