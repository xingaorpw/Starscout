"""SQLite 存储层：仓库、评分事件、推荐记录、键值配置。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name   TEXT NOT NULL UNIQUE,
    owner       TEXT,
    name        TEXT,
    description TEXT,
    language    TEXT,
    topics      TEXT,
    stars       INTEGER DEFAULT 0,
    forks       INTEGER DEFAULT 0,
    open_issues INTEGER DEFAULT 0,
    created_at  TEXT,
    pushed_at   TEXT,
    license     TEXT,
    archived    INTEGER DEFAULT 0,
    homepage    TEXT,
    source      TEXT,
    quality     REAL DEFAULT 0,
    first_seen  TEXT,
    last_seen   TEXT
);

CREATE INDEX IF NOT EXISTS idx_repos_quality ON repos(quality DESC);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id    INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    action     TEXT NOT NULL,
    day        TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (repo_id, day)
);

CREATE INDEX IF NOT EXISTS idx_events_day ON events(day);

-- star 快照：判断"正在冒头"必须看近期增量，只看 star/存活天数会低估新近爆火的项目
CREATE TABLE IF NOT EXISTS star_history (
    repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    day     TEXT NOT NULL,
    stars   INTEGER NOT NULL,
    PRIMARY KEY (repo_id, day)
);

CREATE TABLE IF NOT EXISTS recommendations (
    day     TEXT NOT NULL,
    repo_id INTEGER NOT NULL,
    rank    INTEGER NOT NULL,
    score   REAL NOT NULL,
    reason  TEXT,
    PRIMARY KEY (day, repo_id)
);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


@contextmanager
def db(path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    p = Path(path or config.DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: Path | str | None = None) -> None:
    with db(path) as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------
# 仓库
# --------------------------------------------------------------------------
_REPO_COLS = (
    "full_name", "owner", "name", "description", "language", "topics",
    "stars", "forks", "open_issues", "created_at", "pushed_at", "license",
    "archived", "homepage", "source", "quality",
)


def upsert_repos(repos: Iterable[dict[str, Any]], path=None) -> int:
    now = utcnow()
    day = today()
    sql = f"""
        INSERT INTO repos ({", ".join(_REPO_COLS)}, first_seen, last_seen)
        VALUES ({", ".join("?" * len(_REPO_COLS))}, ?, ?)
        ON CONFLICT(full_name) DO UPDATE SET
            description = excluded.description,
            language    = excluded.language,
            topics      = excluded.topics,
            stars       = excluded.stars,
            forks       = excluded.forks,
            open_issues = excluded.open_issues,
            pushed_at   = excluded.pushed_at,
            license     = excluded.license,
            archived    = excluded.archived,
            homepage    = excluded.homepage,
            quality     = excluded.quality,
            last_seen   = excluded.last_seen
    """
    n = 0
    with db(path) as conn:
        for r in repos:
            conn.execute(sql, _row_values(r) + (now, now))
            row = conn.execute(
                "SELECT id, stars FROM repos WHERE full_name = ?", (r["full_name"],)
            ).fetchone()
            if row is not None:
                conn.execute(
                    "INSERT INTO star_history (repo_id, day, stars) VALUES (?, ?, ?) "
                    "ON CONFLICT(repo_id, day) DO UPDATE SET stars = excluded.stars",
                    (int(row["id"]), day, int(row["stars"] or 0)),
                )
            n += 1
    return n


def _row_values(r: dict[str, Any]) -> tuple:
    return (
        r["full_name"],
        r.get("owner"),
        r.get("name"),
        r.get("description"),
        r.get("language"),
        json.dumps(r.get("topics") or [], ensure_ascii=False),
        int(r.get("stars") or 0),
        int(r.get("forks") or 0),
        int(r.get("open_issues") or 0),
        r.get("created_at"),
        r.get("pushed_at"),
        r.get("license"),
        1 if r.get("archived") else 0,
        r.get("homepage"),
        r.get("source"),
        float(r.get("quality") or 0.0),
    )


def _hydrate(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    try:
        d["topics"] = json.loads(d.get("topics") or "[]")
    except json.JSONDecodeError:
        d["topics"] = []
    d["archived"] = bool(d.get("archived"))
    return d


def all_repos(path=None) -> list[dict[str, Any]]:
    with db(path) as conn:
        rows = conn.execute("SELECT * FROM repos ORDER BY quality DESC").fetchall()
    repos = [_hydrate(r) for r in rows]
    velocities = recent_velocities(path)
    for r in repos:
        r["recent_velocity"] = velocities.get(int(r["id"]))
    return repos


def recent_velocities(path=None) -> dict[int, float]:
    """用最近两次 star 快照算日增。

    只有一个快照的仓库不会出现在结果里——那时只能退回生命周期均值，
    调用方靠 `recent_velocity is None` 区分这两种情况。
    """
    with db(path) as conn:
        try:
            rows = conn.execute(
                "SELECT repo_id, day, stars FROM star_history ORDER BY repo_id, day"
            ).fetchall()
        except sqlite3.OperationalError:
            # 旧库还没建这张表：当成「没有快照」处理，不要连累整个读取路径
            return {}

    points: dict[int, list[tuple[str, int]]] = {}
    for r in rows:
        points.setdefault(int(r["repo_id"]), []).append((r["day"], int(r["stars"])))

    out: dict[int, float] = {}
    for repo_id, pts in points.items():
        if len(pts) < 2:
            continue
        (day1, stars1), (day2, stars2) = pts[-2], pts[-1]
        try:
            span = (date.fromisoformat(day2) - date.fromisoformat(day1)).days
        except ValueError:
            continue
        if span <= 0:
            continue
        out[repo_id] = max(0.0, (stars2 - stars1) / span)
    return out


def record_stars(repo_id: int, stars: int, day: str | None = None, path=None) -> None:
    """手工写一条 star 快照（测试和回填用）。"""
    with db(path) as conn:
        conn.execute(
            "INSERT INTO star_history (repo_id, day, stars) VALUES (?, ?, ?) "
            "ON CONFLICT(repo_id, day) DO UPDATE SET stars = excluded.stars",
            (int(repo_id), day or today(), int(stars)),
        )


def repo_exists(repo_id: int, path=None) -> bool:
    with db(path) as conn:
        row = conn.execute("SELECT 1 FROM repos WHERE id = ?", (int(repo_id),)).fetchone()
    return row is not None


def repo_map(path=None) -> dict[int, dict[str, Any]]:
    return {r["id"]: r for r in all_repos(path)}


def count_repos(path=None) -> int:
    with db(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM repos").fetchone()[0])


# --------------------------------------------------------------------------
# 评分事件
# --------------------------------------------------------------------------
def add_event(repo_id: int, action: str, path=None) -> None:
    """同一天对同一个仓库重复打分，覆盖旧值。"""
    with db(path) as conn:
        conn.execute(
            """
            INSERT INTO events (repo_id, action, day, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(repo_id, day) DO UPDATE SET action = excluded.action,
                                                   created_at = excluded.created_at
            """,
            (repo_id, action, today(), utcnow()),
        )


def all_events(path=None) -> list[dict[str, Any]]:
    with db(path) as conn:
        rows = conn.execute("SELECT * FROM events ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]


def rated_repo_ids(path=None) -> set[int]:
    with db(path) as conn:
        rows = conn.execute("SELECT DISTINCT repo_id FROM events").fetchall()
    return {int(r[0]) for r in rows}


# --------------------------------------------------------------------------
# 推荐记录
# --------------------------------------------------------------------------
def save_recommendations(day: str, items: Iterable[dict[str, Any]], path=None) -> None:
    with db(path) as conn:
        for it in items:
            conn.execute(
                """
                INSERT INTO recommendations (day, repo_id, rank, score, reason)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(day, repo_id) DO UPDATE SET rank = excluded.rank,
                                                        score = excluded.score,
                                                        reason = excluded.reason
                """,
                (day, it["repo_id"], it["rank"], float(it["score"]), it.get("reason", "")),
            )


def clear_recommendations(day: str, path=None) -> int:
    """偏好变了以后，当天的榜单要作废重排。"""
    with db(path) as conn:
        cur = conn.execute("DELETE FROM recommendations WHERE day = ?", (day,))
        return cur.rowcount


def recommendations_for(day: str, path=None) -> list[dict[str, Any]]:
    with db(path) as conn:
        rows = conn.execute(
            """
            SELECT r.*, rec.rank AS rank, rec.score AS score, rec.reason AS reason
            FROM recommendations rec
            JOIN repos r ON r.id = rec.repo_id
            WHERE rec.day = ?
            ORDER BY rec.rank
            """,
            (day,),
        ).fetchall()
    return [_hydrate(r) for r in rows]


# --------------------------------------------------------------------------
# 键值
# --------------------------------------------------------------------------
def kv_get(key: str, default: Any = None, path=None) -> Any:
    with db(path) as conn:
        row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return row[0]


def kv_set(key: str, value: Any, path=None) -> None:
    with db(path) as conn:
        conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )
