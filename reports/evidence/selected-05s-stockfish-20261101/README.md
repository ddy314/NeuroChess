# 最新选中模型，0.5 秒 PUCT 对固定节点 Stockfish

1500、2000 节点各 10 局；每档同样 5 个开局换色，全部正常终局。摘要与核验见 `summary.json`，完整逐步输出为 `full-result.json.gz`。单局 JSON 省略 root 领导着法等非核验字段，所有数值保留，主机路径已规范化。模型与 TensorRT 引擎未收录。

可运行 `python scripts/audit_fast_fixed_node_arena.py --arena reports/evidence/selected-05s-stockfish-20261101 --out /tmp/selected-05s-audit.json` 独立回放全部棋谱并核对胜负、颜色、模型指纹、逐步计时和搜索/推理计数。
