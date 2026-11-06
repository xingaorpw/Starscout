"""采集器：GitHub Search API + Trending 页面。

为什么要两个来源？
- Search API：结构化、字段全，能按 created/pushed 时间切片，适合"最近冒头"；
- Trending 页：没有官方 API，但它反映的是"今天大家在点星"，是很好的补充信号。
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from . import config
from .rank import quality_score

API = "https://api.github.com"
TRENDING = "https://github.com/trending?since=daily"

_ARTICLE_RE = re.compile(r'<article class="Box-row">(.*?)</article>', re.S)
_NAME_RE = re.compile(r'<h2[^>]*>\s*<a[^>]*href="/([^"/]+/[^"/]+)"')
_STARS_TODAY_RE = re.compile(r"([\d,]+)\s+stars?\s+today")
_STARS_TOTAL_RE = re.compile(r'href="/[^"]+/stargazers"[^>]*>\s*([\d,]+)')


class RateLimitExceeded(RuntimeError):
    pass


class GitHub:
    """带限速与重试的极简客户端。不带 token 也能跑，就是额度小（搜索 10 次/分钟）。"""

    def __init__(self, token: str | None = None, session: requests.Session | None = None):
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        self.session = session or requests.Session()
        self._last_search = 0.0

    def _headers(self) -> dict[str, str]:
        h = {"User-Agent": "starscout/0.1", "Accept": "application/vnd.github+json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = path if path.startswith("http") else f"{API}{path}"
        for attempt in range(3):
            resp = self.session.get(url, params=params, headers=self._headers(), timeout=25)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (403, 429):
                if attempt == 2:
                    raise RateLimitExceeded(f"GitHub 限流：{resp.text[:120]}")
                reset = resp.headers.get("X-RateLimit-Reset")
                wait = 30.0
                if reset and reset.isdigit():
                    wait = max(5.0, min(60.0, int(reset) - time.time() + 1))
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            if attempt == 2:
                raise RuntimeError(f"GitHub {resp.status_code}: {resp.text[:160]}")
            time.sleep(2)
        return None

    def search(self, query: str, per_page: int = 100, pages: int = 2) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for page in range(1, pages + 1):
            # 未认证时搜索接口 10 次/分钟，强行拉开间隔
            gap = 6.5 if not self.token else 1.0
            elapsed = time.time() - self._last_search
            if elapsed < gap:
                time.sleep(gap - elapsed)
            data = self.get(
                "/search/repositories",
                {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": per_page,
                    "page": page,
                },
            )
            self._last_search = time.time()
            if not data:
                break
            items = data.get("items") or []
            out.extend(items)
            if len(items) < per_page:
                break
        return out


def normalize(item: dict[str, Any], source: str) -> dict[str, Any]:
    lic = (item.get("license") or {}).get("spdx_id")
    if lic in (None, "NOASSERTION"):
        lic = None
    owner = (item.get("owner") or {}).get("login")
    return {
        "full_name": item["full_name"],
        "owner": owner,
        "name": item.get("name"),
        "description": (item.get("description") or "").strip() or None,
        "language": item.get("language"),
        "topics": list(item.get("topics") or []),
        "stars": int(item.get("stargazers_count") or 0),
        "forks": int(item.get("forks_count") or 0),
        "open_issues": int(item.get("open_issues_count") or 0),
        "created_at": item.get("created_at"),
        "pushed_at": item.get("pushed_at"),
        "license": lic,
        "archived": bool(item.get("archived")),
        "homepage": item.get("homepage") or None,
        "source": source,
    }


def fetch_trending(session: requests.Session | None = None) -> list[dict[str, Any]] | None:
    """抓 https://github.com/trending 的今日榜。纯正则解析，不引额外依赖。

    返回 None 表示站点不可达（国内网络下 github.com 常被墙，但 api.github.com 通常能通），
    返回 [] 表示可达但没解析到条目。两者要分开，否则用户不知道是没数据还是没连上。
    """
    s = session or requests.Session()
    try:
        resp = s.get(TRENDING, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    out = []
    for block in _ARTICLE_RE.findall(resp.text):
        m = _NAME_RE.search(block)
        if not m:
            continue
        stars_today = 0
        m2 = _STARS_TODAY_RE.search(block)
        if m2:
            stars_today = int(m2.group(1).replace(",", ""))
        stars_total = 0
        m3 = _STARS_TOTAL_RE.search(block)
        if m3:
            stars_total = int(m3.group(1).replace(",", ""))
        out.append({"full_name": m.group(1), "stars": stars_total, "stars_today": stars_today})
    return out


def collect(
    days: int = config.SEARCH_DAYS,
    min_stars: int = config.SEARCH_MIN_STARS,
    token: str | None = None,
    topics: list[str] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """采一批候选仓库。返回已算好 quality 的列表。"""
    gh = GitHub(token)
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    found: dict[str, dict[str, Any]] = {}

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    queries = [
        (f"created:>{since} stars:>{min_stars}", "new"),
        (f"pushed:>{since} stars:>2000", "active"),
    ]
    # 按用户口味定向挖：这是"每日采集"里唯一真正个性化的部分
    for topic in (topics or [])[:4]:
        queries.append((f"topic:{topic} pushed:>{since} stars:>50", f"topic:{topic}"))

    for query, tag in queries:
        try:
            items = gh.search(query)
        except RateLimitExceeded as exc:
            log(f"  ! {exc}，跳过剩余查询")
            break
        log(f"  · {tag:14s} 拉到 {len(items)} 个")
        for it in items:
            rec = normalize(it, tag)
            prev = found.get(rec["full_name"])
            if prev:
                prev["source"] = f"{prev['source']}+{tag}"
            else:
                found[rec["full_name"]] = rec

    # Trending 补充：榜单上的项目 API 未必搜得到
    trending = fetch_trending()
    if trending is None:
        log("  ! github.com 不可达，跳过 Trending 源（API 源不受影响）")
        trending = []
    else:
        log(f"  · trending      拉到 {len(trending)} 个")
    fetched = 0
    for t in trending:
        name = t["full_name"]
        if name in found:
            found[name]["source"] += "+trending"
            continue
        if fetched >= config.MAX_DETAIL_FETCH:
            break
        try:
            detail = gh.get(f"/repos/{name}")
        except (RateLimitExceeded, RuntimeError):
            break
        fetched += 1
        if not detail:
            continue
        rec = normalize(detail, "trending")
        found[name] = rec

    repos = list(found.values())
    for r in repos:
        r["quality"] = quality_score(r)
    repos.sort(key=lambda r: r["quality"], reverse=True)
    log(f"  合计 {len(repos)} 个候选，已计算质量分")
    return repos
