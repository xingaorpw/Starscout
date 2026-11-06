# 参与贡献

感谢你愿意花时间。这个项目不大，规则也简单。

## 先跑起来

```bash
git clone https://github.com/xingaorpw/Starscout.git
cd Starscout
pip install -r requirements.txt
python run.py demo      # 离线数据，不联网也能看效果
python -m unittest discover -s tests -v
```

测试必须全绿再提交。

## 提 Issue

有用的 bug 报告长这样：

```
环境：Windows 11 / Python 3.12 / 无 GitHub token
命令：python run.py collect
现象：采集到 0 个仓库，日志显示 "GitHub 限流"
预期：至少能拿到 Trending 那部分
```

- 带上**完整命令**和**完整报错**
- 说清楚有没有配 `GITHUB_TOKEN`
- 如果是推荐结果不对，附上你的评分动作和期望结果

## 提 PR

- 一个 PR 只做一件事
- 改了逻辑就加测试，改了行为就更新 README
- 提交信息用 `feat:` / `fix:` / `docs:` / `refactor:` 开头
- 中文注释和中文文档都可以，跟现有代码保持一致就行

## 几个约定

**不要引入重型依赖。** 现在只有 `requests` + `numpy` + `flask`。
想加 sklearn / pandas / torch 之前先开 issue 讨论——这个项目的卖点之一就是「单机可跑、不需要 GPU」。

**不要为了分数改测试。** 如果某个测试挂了，先确认是实现错了还是测试写错了。
把断言放宽到「能过」不是修复。

**保持推荐理由可解释。** 理由必须能反查到具体的 topic 或语言。
引入任何黑盒打分之前，先想清楚它怎么解释给用户看。

## 可以做的事

见 README 末尾的「可以继续做的方向」。其中比较适合上手的：

- 中文/英文混合的描述分词还能更好
- 推荐结果的分页和无限滚动
- 导出/导入评分记录（换机器不丢画像）

## License

提交的代码默认按 [MIT](LICENSE) 授权。
