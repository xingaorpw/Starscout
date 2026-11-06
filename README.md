# StarScout · 每日 GitHub 星探

[![tests](https://github.com/xingaorpw/Starscout/actions/workflows/ci.yml/badge.svg)](https://github.com/xingaorpw/Starscout/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

每天从 GitHub 捞一批正在冒头的项目，按**你的口味**排序推给你；
你点 👍 / 👎 / 🔖，它当晚就调整明天的榜单。

> 一个完整的"采集 → 画像 → 排序 → 反馈 → 再画像"闭环，约 1800 行代码，单机可跑，不需要 GPU。

![StarScout 界面](docs/screenshot.png)

---

## 30 秒上手

```bash
pip install -r requirements.txt

python run.py demo          # 灌入离线演示数据（没网也能看效果）
python run.py serve         # 打开 http://127.0.0.1:8000
```

想用真实数据：

```bash
set GITHUB_TOKEN=ghp_xxx    # Windows；macOS/Linux 用 export
python run.py collect       # 抓最近 7 天的新项目 + 今日 Trending
python run.py serve
```

没有 token 也能跑，只是搜索接口限 10 次/分钟，采集会慢一些。

其他命令：

```bash
python run.py stats         # 命令行看画像和推荐结果
python run.py reset         # 清空评分、推荐缓存和偏好（保留仓库）
python run.py reset --all   # 连仓库一起清空
python demo_learn.py        # 对照实验：证明它真的会学
```

> **国内网络提示**：`api.github.com` 一般能通，`github.com` 网页版经常超时。
> 抓不到 Trending 时会自动跳过并提示，只靠 API 源，功能不受影响。

---

## 它到底怎么判断"你爱看什么"

不是让大模型猜，是一个能解释的小模型。四步：

### 1. 把仓库变成向量

`name + description + topics + language` 切词，用 blake2b 哈希到 4096 维，再乘 IDF、做 L2 归一化。

不用 sklearn 的 `TfidfVectorizer`，是为了省掉"保存词表"的麻烦——哈希冲突对推荐这种粗粒度任务无所谓。见 [`features.py`](starscout/features.py)。

### 2. 把评分变成权重

每条 👍 是一个正样本、👎 是负样本，跑一个带 L2 的 SGD 逻辑回归，得到权重向量 `w`。

- **先验**：引导页勾选的 topic / 语言直接在对应维度上加分，解决冷启动；
- **时间衰减**：三个月前的口味权重按 45 天半衰期打折；
- **不用深度模型**：几十条样本就能收敛，而且每个维度都能反查成可读的 token。

见 [`profile.py`](starscout/profile.py)。

### 3. 三路打分，再做多样性

```
final = 0.58 × tanh(口味 z 分) + 0.27 × 质量分 + 0.15 × 新鲜度
```

- **质量分**：star 绝对值只占 0.38，**涨星速度**占 0.22——只看 star 总数会永远推那几个巨无霸；
- **新鲜度**：跟你已收藏内容的余弦相似度越低分越高，防止越推越窄；
- **MMR + 同语言上限 3**：一屏 12 张卡不会全是 Python。

见 [`rank.py`](starscout/rank.py)。

### 4. 理由来自模型本身

"命中你的偏好 llm / fine-tuning" 不是模板，是把 `w` 里对这条贡献最大的 topic 反查出来。

这里有一条硬约束：**只有你真的点过赞（或引导页勾选过）的 topic，才允许出现在理由里**。
不然小样本下逻辑回归会把描述里的稀有词（`tailored`、`fancy`）当成你的兴趣，理由就会变得很荒唐——
这是实测踩出来的坑，也是最好的调试手段：理由离谱，说明模型学歪了。

---

## 采集了什么

两个来源互补：

| 来源 | 拿什么 | 为什么 |
|---|---|---|
| Search API | `created:>7天 stars:>50` | 刚冒头、还没被 star 数淹没的新项目 |
| Search API | `pushed:>7天 stars:>2000` | 持续活跃的老项目 |
| Search API | `topic:<你的偏好>` | 定向挖你感兴趣的领域 |
| Trending 页 | 今日榜单 | 没有官方 API，但它反映"今天大家在点什么" |

质量分再过滤一遍：没简介、没 topic、没 license、已归档的都会掉分。

---

## 目录结构

```
starscout/
├── run.py                  命令行入口（init / demo / collect / serve / stats）
├── requirements.txt
├── starscout/
│   ├── config.py           所有可调参数集中在这
│   ├── store.py            SQLite：repos / events / recommendations / kv
│   ├── collect.py          GitHub 采集（含限速重试）
│   ├── features.py         哈希 TF-IDF
│   ├── profile.py          逻辑回归画像 + 可读口味统计
│   ├── rank.py             质量分、MMR 排序、推荐理由
│   ├── server.py           Flask API
│   ├── fixtures.py         47 个离线演示仓库
│   └── templates/index.html  单页前端（无 CDN 依赖）
└── tests/test_core.py      20 个单元测试
```

## API

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/state` | 引导状态、统计、口味画像 |
| POST | `/api/onboard` | 保存兴趣方向 |
| GET | `/api/feed?k=12` | 今日推荐（未评分的） |
| POST | `/api/rate` | `{repo_id, action}`，action ∈ like/dislike/save/skip |
| POST | `/api/collect` | 后台采集，配合 `/api/collect/status` 轮询 |

---

## 跑测试

```bash
python -m unittest discover -s tests -v
```

20 个用例覆盖：向量归一化、相似度、质量分边界、画像学习方向、排除已评分、语言配额、同日评分覆盖、API 全链路。

---

## 想看它真的会学

```bash
python demo_learn.py
```

用同一批候选做对照：只给引导偏好 vs 12 次反馈之后，前 6 名会换掉 4 个，
被喜欢的是 `vllm / llama.cpp / unsloth` 这类推理与微调项目，被讨厌的编辑器与聊天前端会掉出去。

---

## 可以继续做的方向

1. **A/B 两套排序**：把 `config.py` 的权重做成两组，记录点击率对比；
2. **Thompson Sampling**：现在的探索是固定权重，改成按 topic 的 Beta 分布采样，收敛更快；
3. **README 语义化**：现在只用描述和 topics，抓 README 前 2000 字做 embedding，判断会更准；
4. **周报**：每周日生成"本周你错过的 5 个项目"，用 LLM 写一段中文导语；
5. **多用户**：`events` 表加 `user_id`，画像按人隔离，就能做成一个小网站。

---

## 已知限制

- 演示数据的 star 数是手写快照，只用于看效果，不代表仓库真实状态；
- 未认证时 Search API 限 10 次/分钟，采集一轮约 1 分钟；
- 哈希向量有碰撞，极少数情况下两个不相关项目会互相"串味"；
- 单机 SQLite，没有并发写保护，别开多个采集进程。

---

## License

[MIT](LICENSE) © 2026 xingaorpw
