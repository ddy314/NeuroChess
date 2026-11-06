# PUCT 推理优化证据

本轮候选没有晋升；原最终优化版继续作为基线。摘要在 `summary.json`，完整位置和比赛数据为 UTF-8 JSON 的 gzip 压缩文件。主机路径已规范化，实验数值未修改。模型、训练数据和 TensorRT 二进制未收录。

`games/` 包含全部 8 局棋谱及其逐步耗时、推理输入和搜索次数。为缩小体积，单局 JSON 省略 root 的领导着法等非核验字段；完整字段保留在 `arena.json.gz`。可在项目环境运行 `python scripts/audit_fast_duel.py --arena reports/evidence/puct-10x-20261101/games --out /tmp/puct-audit.json` 独立回放及核对。

`critical-grade.json` 展示同一劣势局面在不同 Stockfish 节点预算下的标注变化；不能将单次 CP 标签当成精确真值。原始实验目录为 `output/puct-10x-20261031/`。

`reinvest-*` 为后续把时间收益投入 0.5/1/2 秒搜索的独立位置测试：32 个新棋局局面与 1 个已知回归局面分开统计。这些长预算版本尚无直接对弈结果，不能使用 0.2 秒版本的对弈成绩代替。
