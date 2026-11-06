# Changelog

所有值得注意的变更都记在这里。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.1.1] - 2026-11-06

一次外部审计按 `ec77e34` 复现了 7 个问题，全部确认并修复。每条都有回归测试。

### Fixed

- **安全：写接口可被跨站触发。** `/api/onboard` 不校验来源，且把非 JSON 请求当成空配置处理，
  任意外部页面用普通表单 POST 就能清空用户的兴趣偏好。现在校验 `Origin` / `Sec-Fetch-Site`，
  非 JSON 请求返回 `415`。
- **安全：可以给不存在的仓库评分。** `/api/rate` 只校验 `repo_id` 能否转成整数，
  不检查仓库是否存在。现在返回 `404`，`events` 表补上外键约束。
- **榜单数量不一致。** 「缓存够 3 条就返回」+「不足时生成完整 k 条再拼上缓存」导致：
  先要 3 条再要 12 条只得 3 条；先要 12 条、评 10 条、再要 12 条会返回 14 条。
  现在只补差额，最后截到 `k`。
- **「画像置信度」名不副实。** 实际计算是 `min(1.0, 评分条数 / 25)`，
  不评估推荐准确率、反馈一致性或预测不确定性，25 条评分（连 skip 也算）就显示 100%。
  改名为「反馈进度」，代码注释写清它不代表准确率。
- **TF-IDF 哈希冲突计数错误。** 对 token 去重了，却没对映射后的桶去重。
  同一文档里两个 token 撞进一个桶，该桶的文档频率会 +2。现在先映射到桶再统计。
- **涨星速度用了生命周期均值。** `总 star ÷ 存活天数` 会把「存在五年、昨天刚爆火」的项目
  算成 2.34 star/天，而实际是 250。现在优先用 `star_history` 快照的真实增量。
  采集器早就解析了 Trending 的 `stars_today`，此前从未使用。
- **异常输入返回 500。** `k=abc`、评分请求传 JSON 数组都会崩。现在返回 `400`。
- **中文简介提取不到特征。** 分词正则只有 `[a-z0-9]+`，纯中文简介 token 数为 0。补上 CJK bigram。

### Added

- `star_history` 表，以及 `record_stars()` / `recent_velocities()`：
  记录每次采集的 star 快照，用最近两次算日增。
- 29 个回归测试（20 → 49），覆盖上面每一项。
- README 新增「一次外部审计，以及修了什么」一节。

### Changed

- `run.py` 每个命令执行前先跑 `init_db()`，旧数据库自动补建新增的表。

## [0.1.0] - 2026-10

首个可用版本。

### Added

- GitHub 采集：Search API + Trending 页面双来源，带限速与重试，无 token 也能跑。
- 特征：blake2b 哈希 TF-IDF（4096 维），不保存词表。
- 画像：SGD 逻辑回归在线学习，45 天半衰期，引导页先验解决冷启动。
- 排序：`0.58 × tanh(口味 z 分) + 0.27 × 质量分 + 0.15 × 新鲜度`，MMR 重排（λ=0.72），
  同语言上限 3。
- 推荐理由反查：只有真正点赞过或勾选过的 topic 才允许出现在理由里。
- Flask 单页前端（无 CDN 依赖）、47 个离线演示仓库、`demo_learn.py` 学习效果对照脚本。

[Unreleased]: https://github.com/xingaorpw/Starscout/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/xingaorpw/Starscout/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/xingaorpw/Starscout/releases/tag/v0.1.0
