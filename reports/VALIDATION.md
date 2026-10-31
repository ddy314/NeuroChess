# 验证记录 — 2026-10-29

## 实际完成

- 本地 CPU：7 项测试通过；随机合法走子编码、升变／易位／吃过路兵、镜像视角、重复状态、分片、短样本过拟合、Q 掩码梯度、泄漏检测、一步将杀。
- 本地集成：真实训练保存后恢复，权重与不中断的同配置训练逐元素完全相同；policy/Q/oneply 三种入口返回合法动作。
- 对局接口 smoke：本地 Stockfish 18，正反颜色共两局，故意设置最大 12 ply；两局都正确报告为截断，未计成和棋。这不是棋力测试。
- T4：同一套 7 项测试通过；155M 主模型真实棋谱训练 4 步、恢复至 8 步、从该权重切换 Stockfish 蒸馏 3 步；另作 20 步真实数据吞吐测试。
- 棋谱数据 audit：4060 train / 36 val，88 个 split 内重复，精确局面跨 split 泄漏为零。
- Stockfish 数据 audit：255 train / 1 val，无重复、无精确局面跨 split 泄漏。只有一个验证样本，不能据此评价蒸馏泛化。
- 修复了 Stockfish 多进程退出时引擎线程未关闭导致的挂起，改用 multiprocessing Finalize；重新运行 256 局面标注成功结束。

## GPU 性能

Colab Tesla T4；PyTorch 2.11.0+cu130；主模型 155,000,581 参数。每个合成基准预热 3 步，计时 10 个完整优化步，包含前向、反向、AdamW。

| 模式 | batch | 局面/秒 | 峰值 allocated GiB |
|---|---:|---:|---:|
| BF16 模拟路径（错误选择，已修正） | 128 | 40.49 | 7.30 |
| 原生 FP16 | 32 | 224.86 | 3.10 |
| 原生 FP16 | 128 | 261.52 | 6.22 |
| 原生 FP16 | 256 | 258.84 | 10.38 |
| 原生 FP16 + compile | 128 | 267.83 | 6.13 |

PyTorch 的 `is_bf16_supported()` 在该环境会接受模拟支持；代码现在还检查 GPU compute capability >= 8，防止 T4 误选 BF16。相同 batch=128，实测约快 6.46 倍。

compile 稳态只改善约 2.4%，且上表不含首次编译时间。默认选择不编译、batch=128。不能把这些 10 步基准当作长时间稳定吞吐或所有 T4 的保证。

真实 mmap 数据训练，batch=128、accum=1、2 个 loader worker，20 步累计约 **255.39 局面/秒**，含取数据与训练；该条日志位于最终验证／保存之前，因此不包含末次验证与 checkpoint 写入，且 checkpoint 写在 VM 本地。正式 Drive 保存及更多分片可能更慢。

## 数据来源与结果解释

Smoke 数据：[Lichess 2013-01 标准棋谱](https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst)，实际下载文件 SHA256：

`aa40b3671fa3cf1072eb182892cd90b0e1e003a4a5943492f64b77e7f3fd1635`

双方等级分 >=1800，提取 4096 局面；教师抽样 256 局面，Stockfish 18，每局面 1000 nodes、MultiPV=4，仅做流程检验。引擎二进制哈希在 `teacher-manifest.json`。

20 步训练的监控验证 top-1 为 13/36；样本小、数据相关且模型已经见过训练 split，该数字不能用于任何 Elo 推断。没有执行正式棋力匹配、独立战术题评测或 2500 评级验证。

## 资源与保存

A100 按用户要求停止，没有用于训练。随后 T4 实测账户使用率 1.07 计算单位/小时。测试前后账户余额显示 197.87 → 197.69，约消耗 0.18 单位（账户显示精度有限）。结束后：无活动实例，0.00 单位/小时。

本轮没有进行长训练。远端 smoke 检查点只用于流程测试，未下载数 GB 的优化器文件，已随测试实例释放；日志、性能 JSON、数据 manifest 已保存到本仓库。正式训练命令必须把 checkpoint 放到持久目录。

## 原始证据

- `local-tests.txt`、`local-integration.txt`
- `t4-pilot.log`、`t4-final-validation.log`
- `t4-fp16-*.json`；`t4-batch*.json` 保留错误 BF16 路径的对照
- `human-audit.json`、`teacher-audit.json`
- `human-manifest.json`、`teacher-manifest.json`

本地 PyTorch 2.15.0+cpu；云端版本见基准 JSON。未运行 GitHub CI，未提交或推送 Git。

## GUI 与近期数据增量验证

后续按用户要求新增后台启动、只读 Web GUI、周期评测、近期数据 profile 及持久数据缓存。当前本地 **13 项测试通过**（`gui-tests.txt`）：新增 HTTP 实时读取、截断 JSONL、续训去重、路径/软链接越界拒绝、数据质量筛选、多源去重、缓存复制恢复，以及周期对局+断点恢复。周期测试通过超短预算确认未开局／截断不被计为和棋。

`runs/gui-check` 为真实 tiny 模型 24 步训练，step 8/16/24 分别进行了 4 局 Stockfish 对局后继续训练。结果存于 `gui-arena.jsonl`，只用于验证调度和页面，不属于棋力结论。

Playwright 检查了桌面和 390px 手机布局、EMA、暂停刷新；HTTP 测试验证 PGN 下载与动态日志更新。截图在 `output/playwright/dashboard-desktop.png` 与 `dashboard-mobile.png`（忽略目录，不提交）。未发布外部网站。

云端另使用 **CPU 实例**验证后台“远程近期 PGN → 过滤/去重 → audit → tiny 训练 → GUI API”。首次严格要求赛事有用时标签，仅得到 19 个局面且没有验证集，训练正确拒绝；该失败保留在 `colab-gui-check.json`。随后对赛事允许未知用时并统计，成功结果见 `colab-gui-check-v2.json`：2026-09 赛事源读取 1125 盘，保留 512 局面，去重 9 局面，509 train / 3 val；两步训练完成，监控 API 返回 complete。保留的 26 盘均缺失用时，未将其声称为已验证的慢棋。

2026-09 标准棋源另做小样本流式验证：读取 443 盘，保留 64 局面，双方等级分 ≥2300、基础用时 ≥180 秒，无未知用时；结果在 `recent-rated-prepare.log`。这是过滤器检查，64 个样本未形成 val，不用于训练。完整默认 profile（四个来源、最高 275 万局面）尚未全量处理。

从云端实际安装的 Colab 包检查，`serve_kernel_port_as_window` 已弃用，启动说明已改用 `serve_kernel_port_as_iframe(8765, height=900)`。本次验证了云端 HTTP API 和本地浏览器；没有替用户登录并渲染其 Colab iframe。长期训练、正式棋力验证仍未执行。
