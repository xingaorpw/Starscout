"""把仓库变成一个稀疏向量（hashing + TF-IDF）。

不依赖 sklearn：词表用 blake2b 哈希到固定维度，省掉保存 vocab 的麻烦，
代价是会有哈希冲突，但对"推荐"这种粗粒度任务完全够用。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Iterable

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    the and for with that this from your you our their its are was were will can
    have has had not but all any can use used using into out over more most other
    than then them these those such very just also about after before between
    based build built tool tools project projects simple small fast easy new
    """.split()
)


class Vectorizer:
    """哈希 TF-IDF。先 fit 语料统计 idf，再 transform 单条。"""

    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim
        self._bucket_cache: dict[str, int] = {}
        self.df = np.zeros(dim, dtype=np.float64)
        self.n_docs = 0
        self.idf = np.ones(dim, dtype=np.float64)

    # -- 基础 ------------------------------------------------------------
    def bucket(self, token: str) -> int:
        b = self._bucket_cache.get(token)
        if b is None:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            b = int.from_bytes(digest, "big") % self.dim
            self._bucket_cache[token] = b
        return b

    @staticmethod
    def tokens(repo: dict[str, Any]) -> list[str]:
        out: list[str] = []
        lang = (repo.get("language") or "").strip().lower()
        if lang:
            out.append(f"lang:{lang}")
        for t in repo.get("topics") or []:
            t = str(t).strip().lower()
            if t:
                out.append(f"topic:{t}")
        text = f"{repo.get('name') or ''} {repo.get('description') or ''}".lower()
        for w in TOKEN_RE.findall(text):
            if len(w) < 2 or w in STOPWORDS:
                continue
            out.append(f"w:{w}")
        return out

    # -- 拟合 ------------------------------------------------------------
    def fit(self, repos: Iterable[dict[str, Any]]) -> "Vectorizer":
        self.df[:] = 0.0
        self.n_docs = 0
        for r in repos:
            for tok in set(self.tokens(r)):
                self.df[self.bucket(tok)] += 1.0
            self.n_docs += 1
        self.idf = np.log((1.0 + self.n_docs) / (1.0 + self.df)) + 1.0
        return self

    # -- 转换 ------------------------------------------------------------
    def transform(self, repo: dict[str, Any]) -> tuple[np.ndarray, list[tuple[str, int, float]]]:
        """返回 (L2 归一化向量, [(token, bucket, 权重), ...])。"""
        tf: Counter[int] = Counter()
        labels: dict[int, str] = {}
        for tok in self.tokens(repo):
            b = self.bucket(tok)
            tf[b] += 1
            labels.setdefault(b, tok)
        if not tf:
            return np.zeros(self.dim), []

        idx = np.fromiter(tf.keys(), dtype=np.int64, count=len(tf))
        counts = np.fromiter(tf.values(), dtype=np.float64, count=len(tf))
        vals = (1.0 + np.log(counts)) * self.idf[idx]
        norm = float(np.linalg.norm(vals))
        if norm > 0:
            vals = vals / norm

        vec = np.zeros(self.dim, dtype=np.float64)
        vec[idx] = vals
        detail = [(labels[int(b)], int(b), float(v)) for b, v in zip(idx, vals)]
        return vec, detail


def pretty_token(token: str) -> str:
    """把内部 token 变成人看得懂的样子。"""
    for prefix in ("topic:", "lang:", "w:"):
        if token.startswith(prefix):
            return token[len(prefix):]
    return token
