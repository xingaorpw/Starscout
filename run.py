"""命令行入口。

    python run.py init                初始化数据库
    python run.py demo                灌入离线演示数据
    python run.py collect             从 GitHub 抓取
    python run.py serve               启动网页
    python run.py stats               看看现在学成什么样
"""

from __future__ import annotations

import argparse
import sys

from starscout import collect as collector
from starscout import config, fixtures, profile, rank, store


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def cmd_init(_: argparse.Namespace) -> None:
    store.init_db()
    print(f"数据库已就绪：{config.DB_PATH}")


def cmd_demo(_: argparse.Namespace) -> None:
    store.init_db()
    repos = fixtures.load()
    n = store.upsert_repos(repos)
    print(f"已灌入 {n} 个演示仓库。现在运行：python run.py serve")


def cmd_collect(args: argparse.Namespace) -> None:
    store.init_db()
    topics = store.kv_get("prior_topics", []) or []
    print(f"开始采集（最近 {args.days} 天，最低 {args.min_stars} star）…")
    repos = collector.collect(
        days=args.days,
        min_stars=args.min_stars,
        token=args.token,
        topics=topics,
    )
    n = store.upsert_repos(repos)
    print(f"已写入 {n} 个仓库，库内共 {store.count_repos()} 个。")


def cmd_serve(args: argparse.Namespace) -> None:
    from starscout.server import create_app

    store.init_db()
    if store.count_repos() == 0:
        print("库是空的，先灌入演示数据…")
        store.upsert_repos(fixtures.load())
    app = create_app()
    print(f"StarScout 已启动 → http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=args.debug)


def cmd_reset(args: argparse.Namespace) -> None:
    store.init_db()
    with store.db() as conn:
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM recommendations")
        conn.execute("DELETE FROM kv")
        if args.all:
            conn.execute("DELETE FROM repos")
    scope = "全部数据" if args.all else "评分记录、推荐缓存和偏好设置"
    print(f"已清空{scope}。")
    if args.all:
        print("接着可以跑：python run.py demo  或  python run.py collect")


def cmd_stats(_: argparse.Namespace) -> None:
    repos = store.all_repos()
    by_id = {r["id"]: r for r in repos}
    events = store.all_events()
    if not repos:
        print("库是空的。先跑 python run.py demo 或 python run.py collect")
        return

    from starscout.features import Vectorizer

    vec = Vectorizer(config.FEATURE_DIM).fit(repos)
    w = profile.build(
        vec,
        by_id,
        events,
        store.kv_get("prior_topics", []) or [],
        store.kv_get("prior_languages", []) or [],
    )
    taste = profile.taste_summary(by_id, events)

    print(f"仓库 {len(repos)} 个 · 评分 {len(events)} 条 · 画像置信度 {profile.confidence(events):.0%}")
    if taste["topics"]:
        print("\n口味（topic）：")
        for t in taste["topics"]:
            bar = "#" * max(1, int(abs(t["score"]) * 2))
            sign = "+" if t["score"] > 0 else "-"
            print(f"  {t['name']:<24} {sign}{abs(t['score']):>5.1f}  {bar}")

    picks = rank.recommend(
        repos,
        w,
        vec,
        k=8,
        exclude=store.rated_repo_ids(),
        preferred_topics={t.lower() for t in (store.kv_get("prior_topics", []) or [])}
        | {t["name"].lower() for t in taste["topics"] if t["score"] > 0},
        preferred_languages={l.lower() for l in (store.kv_get("prior_languages", []) or [])}
        | {l["name"].lower() for l in taste["languages"] if l["score"] > 0},
    )
    print("\n如果现在推给你，会推这 8 个：")
    for it in picks:
        r = it["repo"]
        print(f"  {it['rank']:>2}. {r['full_name']:<42} q={it['quality']:.2f}  {it['reason']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="starscout", description="每日 GitHub 精选，越用越懂你")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="初始化数据库").set_defaults(func=cmd_init)
    sub.add_parser("demo", help="灌入离线演示数据").set_defaults(func=cmd_demo)
    sub.add_parser("stats", help="查看画像与推荐结果").set_defaults(func=cmd_stats)

    r = sub.add_parser("reset", help="清空评分记录（--all 连仓库一起删）")
    r.add_argument("--all", action="store_true")
    r.set_defaults(func=cmd_reset)

    c = sub.add_parser("collect", help="从 GitHub 采集")
    c.add_argument("--days", type=int, default=config.SEARCH_DAYS)
    c.add_argument("--min-stars", type=int, default=config.SEARCH_MIN_STARS)
    c.add_argument("--token", default=None, help="GitHub token，不填就读环境变量 GITHUB_TOKEN")
    c.set_defaults(func=cmd_collect)

    s = sub.add_parser("serve", help="启动网页")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--debug", action="store_true")
    s.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    _utf8()
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
