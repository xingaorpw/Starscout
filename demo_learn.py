"""验证学习闭环：同样的候选池，评分前后推荐结果应该明显不同。"""

import os
import tempfile
from pathlib import Path

tmp = tempfile.TemporaryDirectory()
os.environ["STARSCOUT_DATA"] = tmp.name

from starscout import config, fixtures, profile, rank, store  # noqa: E402
from starscout.features import Vectorizer  # noqa: E402

db = Path(tmp.name) / "learn.db"
store.init_db(db)
store.upsert_repos(fixtures.load(), db)


def top(k=6):
    repos = store.all_repos(db)
    by_id = {r["id"]: r for r in repos}
    events = store.all_events(db)
    vec = Vectorizer(config.FEATURE_DIM).fit(repos)
    w = profile.build(vec, by_id, events, ["llm", "rag"], ["Python"])
    taste = profile.taste_summary(by_id, events)
    pref_topics = {"llm", "rag"} | {t["name"].lower() for t in taste["topics"] if t["score"] > 0}
    pref_langs = {"python"} | {l["name"].lower() for l in taste["languages"] if l["score"] > 0}
    picks = rank.recommend(
        repos, w, vec, k=k,
        exclude=store.rated_repo_ids(db),
        preferred_topics=pref_topics,
        preferred_languages=pref_langs,
    )
    return [(p["repo"]["full_name"], round(p["score"], 3), p["reason"]) for p in picks], by_id, events, vec, w


print("=== 初始（只有引导页偏好：llm / rag / Python）===")
before, by_id, events, vec, w = top()
for i, (n, s, r) in enumerate(before, 1):
    print(f"{i}. {n:<42} {s:+.3f}  {r}")

# 模拟 12 次反馈：喜欢大模型基础设施，讨厌前端/编辑器
likes = [
    "vllm-project/vllm", "ggml-org/llama.cpp", "ollama/ollama", "hiyouga/LLaMA-Factory",
    "unslothai/unsloth", "sgl-project/sglang", "BerriAI/litellm", "infiniflow/ragflow",
]
dislikes = ["lobehub/lobe-chat", "open-webui/open-webui", "neovim/neovim", "zed-industries/zed"]

name_to_id = {r["full_name"]: r["id"] for r in by_id.values()}
for n in likes:
    store.add_event(name_to_id[n], "like", db)
for n in dislikes:
    store.add_event(name_to_id[n], "dislike", db)

print("\n=== 12 次反馈之后 ===")
after, by_id, events, vec, w = top()
for i, (n, s, r) in enumerate(after, 1):
    print(f"{i}. {n:<42} {s:+.3f}  {r}")

print("\n=== 口味画像 ===")
taste = profile.taste_summary(by_id, events)
for t in taste["topics"][:6]:
    print(f"  {t['name']:<22} {t['score']:+.2f}")
print(f"  置信度 {profile.confidence(events):.0%}")

before_names = [n for n, _, _ in before]
after_names = [n for n, _, _ in after]
print(f"\n前后重合 {len(set(before_names) & set(after_names))}/{len(before_names)} 个")
print("新进入榜单：", [n for n in after_names if n not in before_names])
print("被挤出去：", [n for n in before_names if n not in after_names])
