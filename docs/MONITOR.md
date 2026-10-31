# 实时 GUI、默认近期数据与周期评测

## 默认数据（已按本次要求修改）

启动器缺少预处理数据时，会读取 `configs/data/elite-recent.json`，在 Colab 内通过 HTTPS 流式读取压缩 PGN，边解压边过滤、写入分片，完成审核后才开始训练。达到每个来源的上限会停止读取该来源，不需要先下载整个大月份文件。

| 来源 | 门槛 | 局面上限 |
|---|---|---:|
| 2026-09、08、07 Lichess 赛事转播，各一份 | 双方 PGN Elo ≥2400；已知基础用时 ≥600 秒；缺失用时单独统计并保留 | 各 250,000 |
| 2026-09 Lichess 标准等级分对局 | 双方 Lichess Elo ≥2300；基础用时 ≥180 秒；缺失用时不收 | 2,000,000 |

这些是上限，不是承诺能提取到的数量。日期固定以便复现，不会悄悄换成另一个月份。线上等级分与赛事 PGN 等级分不能直接视为同一评分池。赛事源不是天然“全部精英”：筛掉低／缺失等级分、未结束、机器人、非标准棋、已知用时太短和解析错误的棋局。

多个来源精确棋盘去重，优先保留先出现的赛事源；同一棋盘只保留第一份策略／结果标签，减少重复但也丢失了同局面不同人类选择的分布。数据转换器会在 manifest 中记录每源实际数量、重复剔除、未知用时和各拒绝原因。

来源与许可：[Lichess 数据库](https://database.lichess.org/)。标准棋 CC0；赛事转播 CC BY-SA 4.0。`SOURCES.json` 和 manifest 保留来源、署名和许可；分发赛事派生数据时须保留其许可信息。2013 年数据只用于早期 smoke 测试，不是启动器默认。

## 推荐启动方式

先在本机打包源码（无需打包训练数据）：

```bash
cd /path/to/NeuroChess
tar -czf /tmp/neurochess-source.tar.gz neurochess configs scripts tests pyproject.toml
colab new -s neurochess-train --gpu T4
colab upload -s neurochess-train /tmp/neurochess-source.tar.gz /content/neurochess-source.tar.gz
colab drivemount -s neurochess-train /content/drive
```

最后一条需要你在自己的终端完成 Drive 授权。然后启动后台流程：

```bash
colab exec -s neurochess-train -f scripts/colab_job.py --timeout 180 \
  --env NC_BACKGROUND=1 \
  --env NC_DATA_CACHE=/content/drive/MyDrive/neurochess-data/elite-recent-2026q3 \
  --env 'NC_JOB=["--data","data/elite-recent","--out","/content/drive/MyDrive/neurochess/human-recent","--steps","10000","--stop-after","100","--batch-size","128","--accum","8","--workers","2","--save-every","100","--warmup","500","--log-every","1"]'
```

这是 **100 步 pilot**，学习率按计划的 10000 总步计算。流程会自动准备近期数据、审核、缓存到 Drive、启动 GUI、安装缺少的 Stockfish，并训练。`colab exec` 很快返回仅表示后台流程已启动；完成情况以页面或 `status.json` 为准。

注意首轮过滤大量棋谱可能花较长 CPU 时间。要更省钱，可以先在**不带 GPU 的 CPU 实例**运行同一命令并额外加 `--env NC_PREPARE_ONLY=1`，等数据缓存写完再停止 CPU 实例；之后申请 T4、重新上传源码并挂载 Drive，运行上面的命令，从缓存复制到本地训练。后台状态显示 `prepared` 才表示数据准备结束；复制过程中不要释放实例。

脚本不在训练步骤中在线解析 PGN；训练读取 VM 本地分片，避免网络抖动卡住 GPU。缓存路径应专属于该 profile；不完整／不匹配缓存需要检查后换新路径。失败的转换目录不会被自动删除或静默合并。

## 打开 GUI

在本机连接当前正在运行的 Colab 训练并自动打开浏览器：

```bash
cd /path/to/NeuroChess
python scripts/monitor.py
```

页面地址为 `http://127.0.0.1:18765/`。脚本把监控连接放在用户 systemd 服务中，关闭终端后连接继续运行；连接失败时会重试。本机重启后再次执行上述命令即可。SSH 密钥保存在 `~/.local/state/neurochess-monitor/`，不依赖临时目录。脚本只连接已有的 `neurochess-train` 会话，不启动训练；没有活动会话时会明确报错。可用 `--no-open` 只建立连接，或用 `--session NAME --port PORT` 指定其他会话和本地端口。

停止本机的监控连接：

```bash
systemctl --user stop neurochess-monitor-neurochess-train-18765.service
```

停止此连接不会停止远程训练。

在 Colab 网页内查看已有实例的监控：

```bash
colab url -s neurochess-train
```

在返回的 Colab 地址中连接**已有实例**，运行一个单元格：

```python
from google.colab import output
output.serve_kernel_port_as_iframe(8765, height=900)
```

使用 iframe；当前 Colab 的 `serve_kernel_port_as_window` 已弃用且可能受浏览器安全策略影响。后台模式让内核空闲，可以执行这个单元格。

本机查看本地训练日志：

```bash
.venv/bin/python -m neurochess.dashboard --root runs --port 8765
```

打开 `http://127.0.0.1:8765`。默认只绑定本机回环地址，不需要公开网站、付费监控平台或外部 JS/CDN。

页面每 5 秒读取一次日志；训练日志频率由 `--log-every` 控制。提供总／策略／Q／WDL loss、验证 loss、学习率、吞吐、训练状态、数据来源数量、周期对局表、PGN 下载、窗口筛选和可选 EMA 平滑。悬停显示原始数值。长日志只读取最近 5000 条，重复 step 显示最新记录。

## 周期棋力评测

通过 `colab_job.py` 默认开启：每 1000 步以及本轮结束时评测，4 个开局正反颜色共 8 局，Stockfish 1000 nodes，最多 120 秒。使用仓库 `configs/monitor-openings.fen` 的固定常见开局，**这是开发监控集，不是独立 Elo 测试集**。

评测复用已有模型，暂停训练计算，结束后继续，不额外占用另一份 GPU 模型。120 秒为协作式时限，单次推理、引擎启动／响应可能造成少量超出。达到时限／步数上限的局记截断，剩余未开始局单独计数；不算成和棋。引擎故障写入 `arena.jsonl`，训练继续，页面显示失败。

结果是固定条件下的配对得分率与置信区间，**不是绝对 Elo**。少量开局区间不稳定；若中途换了引擎／预算／开局／模式，图上分开画线。

定制频率、开局和预算，把以下参数加进 `NC_JOB`（一旦显式提供 `--arena-every`，自动配置不再覆盖你）：

```text
--arena-every 2000 --arena-stockfish /usr/games/stockfish
--arena-openings configs/monitor-openings.fen --arena-pairs 4
--arena-nodes 1000 --arena-max-seconds 120 --arena-max-plies 200
```

关闭自动评测：`--env NC_ARENA=0`。使用已有自定义引擎：`--env NC_STOCKFISH=/path/to/stockfish`。直接运行 `python -m neurochess.train` 时不自动安装引擎，周期评测仍需显式指定参数。

## 继续训练

确认 pilot 完成后，启动同样的命令，**去掉** `"--stop-after","100"`，并加入：

```text
"--resume","/content/drive/MyDrive/neurochess/human-recent/last.pt"
```

保留总步数、batch、accum、workers、seed、lr 和 warmup 不变。可以把保存频率从 100 改为 500。避免在当前流程尚未完成时重复启动；同一输出目录有进程锁。

旧脚本的自定义单 PGN 路线仍可用 `NC_PGN_URL`、`NC_MIN_ELO`、`NC_MAX_POSITIONS` 覆盖。使用自行生成的非 profile 数据（例如 teacher 蒸馏分片）时，显式加 `NC_REQUIRE_PROFILE=0`，否则默认会拒绝悄悄复用不符的旧数据。

结束后确认 Drive 里的 checkpoint，再 `colab stop -s neurochess-train`。后台训练、监控服务不会阻止 Colab 回收实例，跨实例恢复仍依赖持久数据和 checkpoint。

## 本次训练的断线恢复

`scripts/remote_resume.py` 专用于 `human-recent` 和 `elite-recent-2026q3` 缓存，不能直接当成任意实验的通用启动器。先上传包含最新源码的 `/content/neurochess-source.tar.gz` 并挂载 Drive，再执行：

```bash
colab exec -s neurochess-train -f scripts/remote_resume.py --timeout 86400
```

恢复器将原训练目录和数据缓存复制到 VM 本地，从 `last.pt` 保留优化器状态续训，保持原总步数和学习率计划。训练持续占用内核执行任务；监控服务独立读取 `/content/neurochess-runs`。日志和 checkpoint 先写本地，每 30 秒尝试向原 Drive 目录同步，文件先复制到临时文件再替换。Drive 失效时同步报错并重试，不因单次日志写入失败中止训练，也不自动创建消失的 Drive 根目录。训练结束后等待最终同步成功才返回。

这不保证 Colab 不回收实例；若实例在 Drive 不可用期间被回收，只能恢复最后成功同步的 checkpoint。不要同时启动两个恢复器，也不要在同一实例已有本地恢复目录时盲目重复执行。

## 本地配对评测与一步搜索

本轮本地评测固定使用已校验的 5100 步和 8400 步检查点、128 个不同 ECO 开局，逐开局交换颜色。每个模型分别对 Stockfish 18 的 Skill Level 20、0 各下 256 局，再进行 256 局模型间对战，共 1280 局。Stockfish 使用每步 1000 nodes、单线程、16 MiB Hash；达到 800 个半回合仍未结束的局会单独标记，不算和棋。结果是本协议下的得分，不能换算绝对 Elo。

沿用本机已有 CUDA 环境，不安装新环境或依赖：

```bash
cd /path/to/NeuroChess
python scripts/local_arena_existing_env.py \
  --old output/local-arena-20261030/checkpoints/step-5100.pt \
  --new output/local-arena-20261030/checkpoints/step-8400.pt \
  --openings output/local-arena-20261030/openings.fen \
  --stockfish local/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2 \
  --out output/local-arena-20261030/oneply-results \
  --device cuda --mode oneply --batch-games 16 --nodes 1000 --max-plies 800
```

`local_arena_existing_env.py` 使用当前解释器已有的 PyTorch/NumPy；缺少 `python-chess` 时读取项目 `.venv` 中已有的纯 Python 包。`--mode policy` 直接选择策略头最大概率的合法走法；`--mode oneply` 枚举所有合法走法，精确判定子局面的终局结果，否则取价值头 `P(win)-P(loss)` 的相反数作为当前方分数，再选最大者。不展开对手回复，不混合策略先验，不使用 Stockfish 给本方评估。子局面价值分批计算，保持棋局历史以处理重复与五十步规则。

输出包含输入哈希和协议 `plan.json`、逐局记录 `games.jsonl`、棋谱 `games.pgn`、进度 `status.json` 和结果 `result.json`。同一协议重复启动时跳过已完整保存的 GameId；启动前应确认原进程已停止。统计以完整的换色开局对为单位做 bootstrap，并额外保留保守的 Hoeffding 区间，避免全胜/全负时 bootstrap 区间退化被误解为零误差。Skill Level 0 自带随机性，重复运行不保证逐局相同。

### MCTS 与剪枝

`--mode mcts --simulations 256 --c-puct 1.5 --root-keep 12` 启用策略先验引导的 PUCT。叶节点使用价值头，边的 Q 是从父节点行棋方视角回传的均值，每跨一层取反；最终按访问次数选点，精确终局证明优先。搜索不使用未训练的网络 Q 头。

所有节点按访问次数渐进扩展，保留将军、吃子和升变候选；根节点用完 75% 预算后，仅继续搜索主要访问分支、前四策略先验分支、战术分支与已证明的必胜分支。根节点剪枝是启发式，不代表被剪分支已经证明较差；只有所有合法子分支都被解出，才可证明某节点必败或必和。检查分支会显式探测一步将杀。树保留完整棋局历史，不合并忽略历史的转置，不添加根噪声。可在不同棋局之间批量评估叶节点，每棵树每轮只选择一个叶节点，因此不需要并发虚拟损失。

本次小样本用 `mcts-smoke/openings.fen` 中的两个开局，指定 `--conditions strong weak --batch-games 4`，共 16 局。最终单文件结果为 `output/local-arena-20261030/mcts-smoke/result.json`，含协议、输入哈希、搜索与剪枝计数、逐局 UCI 走法、同开局基线和复核结果。该样本仅用于功能验证，不能用于可靠棋力或绝对 Elo 估计。

MCTS 的原生 C++ 后端、编译方法、子树复用规则和测速命令见
[NATIVE_MCTS.md](NATIVE_MCTS.md)。使用 `--mode mcts` 前先运行
`python scripts/build_native_mcts.py`；默认使用 C++ 和 16 叶批量推理，
`--mcts-backend python` 可切回参考实现。
