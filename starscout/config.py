"""全局配置。所有可调参数集中在这里，方便做实验对比。"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("STARSCOUT_DATA", str(ROOT / "data")))
DB_PATH = DATA_DIR / "starscout.db"

# ---- 特征 ----
FEATURE_DIM = 4096

# ---- 推荐 ----
TOP_K = 12                 # 每次推几张卡
MMR_LAMBDA = 0.72          # 相关性 vs 多样性的权衡
MAX_PER_LANGUAGE = 3       # 同一语言最多几张，防止一整屏都是 Python

W_TASTE = 0.58             # 口味匹配
W_QUALITY = 0.27           # 项目质量
W_NOVELTY = 0.15           # 新鲜度（别老推你收藏过的同类）

# ---- 采集 ----
SEARCH_DAYS = 7            # 只看最近 N 天冒头的项目
SEARCH_MIN_STARS = 40
MAX_DETAIL_FETCH = 20      # Trending 补详情时的上限，保护 API 额度

# ---- 质量分标定 ----
# 这几个 cap 决定"多少算顶格"，按当前 GitHub 生态标定：
# 40 万 star 基本是人类天花板，1500 star/天 是病毒式传播级别。
STAR_CAP = 400_000
VELOCITY_CAP = 1_500      # star / 天
FORK_CAP = 20_000

# ---- 画像 ----
RECENCY_HALFLIFE_DAYS = 45  # 旧评分按这个半衰期衰减
PRIOR_TOPIC_WEIGHT = 0.9
PRIOR_LANG_WEIGHT = 0.7
