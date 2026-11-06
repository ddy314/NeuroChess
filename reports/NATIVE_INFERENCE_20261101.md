# C++/CUDA 原生推理与搜索运行时（2026-11-01）

已实现可独立落子的 C++/CUDA 程序。训练、模型结构、监督和检查点保持不变；实际运行不加载 Python、NumPy、PyTorch 或 LibTorch。相较此前已经使用 TensorRT/CUDA Graph 的优化后端，本轮观测为小幅提速，没有达到新的数量级提升，也未替换冻结基线。

## 实现范围

- `neurochess/native/trt_inference.cu/.hpp`：C++ TensorRT runtime，固定设备缓冲区和页锁定 CPU 缓冲区，异步搬运，每个尺寸独立 context 和 CUDA Graph。
- 手写 CUDA 输出头：合法着法遮罩、稳定 top-5、policy 支持的 Q 校正、WDL 置信度及价值处理。矩阵乘和 Transformer 仍由 TensorRT CUDA 内核执行，没有从零重写 GEMM。
- `native_player.cpp`：C++ 输入缓存、唯一未命中项聚合、命中立即回传、批处理、时钟预算和异步固定预算 UCI；原 C++ `mcts_pipeline.cpp` 原样复用。停止或错误时撤回虚拟预留，非法历史出错后重建状态。
- CMake 使用 C++17、CUDA、TensorRT SDK。离线导出和测试控制器仍可使用 Python，Python 不进入原生棋手的落子计算路径。

依赖检查显示原生程序链接 `libnc_inference`、TensorRT、CUDA runtime 和系统 C++ 库，无 Python/PyTorch 动态库。程序仍需要匹配的 NVIDIA 原生运行库，不是无依赖单文件。

[构建、运行命令与协议范围](../docs/NATIVE_INFERENCE.md)。本机可执行文件位于 `output/native-runtime-20261101/build/neurochess_native`，默认固定步时 0.5 秒，可用 `--time -1 --simulations 512` 做固定模拟次数对照。

## 模型与语义检查

使用此前 20 局 Stockfish 测试的同一份累计 15,500 步选中模型引擎。训练检查点 SHA256 为 `e4ed8978db91eeef8360c4dea87faf7a4f826325bce45aa7efc024692d585ef7`，引擎 SHA256 为 `1b1213289aa79d30b116a6b08130ad8138aec0fc7fb9ce453e0df6a400d66ac8`。本轮未重新训练或重新量化模型。

- 原始/融合输出在 1–8 的不同批次尺寸交替调用中核对，最大误差约 `2.38e-7`。支持遮罩一致，未宣称浮点逐位等价。
- 合成并列分数、少于 5 个合法着法等输出头检查通过。
- 16 个局面，关闭缓存/32768 缓存两种设置，固定 512 次搜索：原生程序与“Python 调度＋同一个 C++ 推理核”返回的可见根结果全部一致，包括着法、总模拟数、节点数、保留访问以及前 5 候选的访问/Q/先验信息。这不是对未导出的整棵树的逐字节比较。
- 同样 32 个固定预算对照中，原生程序与原 Python/Triton 输出头方案的最终着法均一致。
- 7 项真实原生进程检查通过：固定预算、完整历史复用与历史身份、王车易位、吃过路兵、升变、提前停止后继续搜索、8192 次搜索下的数组扩容、非法历史失败后恢复。部分场景合并在同一测试中。

这些核验支持本轮移植的局部一致性，不能替代大样本棋力验证。没有为原生版本重新跑完整 Stockfish 对弈，不能把之前 Python 后端的 20 局成绩记到原生后端名下。原冻结两秒基线的源文件、规则库和二进制指纹均未改变。

## 同预算计时

比较的是同一 TensorRT plan，而不是拿新的原生程序与未优化的 PyTorch eager 做不等条件对照。初始化/图捕获在计时前完成；一次独立 CLI 启动的加载与预热约 1.38 秒，单独记录，不计入每步搜索时间。

最终一轮 16 局面固定 512 次搜索：

| 调度/推理 | 关闭缓存，中位耗时 | 缓存 32768，中位耗时 |
|---|---:|---:|
| 原 Python＋TensorRT＋Triton 头 | 0.28031 s | 0.25787 s |
| Python 调度＋同一个 C++/CUDA 推理核 | 0.27553 s | 0.25335 s |
| 全原生 C++/CUDA | 0.27298 s | 0.25161 s |

开启缓存时，中位耗时比约 1.025，耗时降低约 **2.4%**。配对几何平均耗时比为 1.036，局面 bootstrap 95% 区间为 [1.012, 1.058]。关闭缓存的配对区间覆盖 1。此前一轮观察约 5% 的改善，说明目前只能报告“小幅改善”，不能挑更快的一轮作为稳定收益。

同样 0.5 秒预算，最终一轮原生/原 Python 搜索次数中位数为 1048/979，原生 P95/最大耗时为 0.49653/0.49702 秒。搜索量增加不直接构成棋力提升的证据。

本次不是空闲 GPU 测量：实验以外存在图形负载，结束后单独采样仍约 30% 利用率。测试内部采样为 44%–55%，其中也包含本实验，不能全部归为外部负载。双方按局面交替顺序比较，但负载变化仍限制了百分比结论。因此不与之前空闲时的绝对耗时混合，也不声称这些区间覆盖系统负载的所有不确定性。

此前剖析已显示大头是 GPU GEMM，原优化后端早已在执行原生 CUDA 算子；本轮移除的是剩余调度与搬运开销。这解释了为何移植可行且部署依赖减少，但没有带来新的 10 倍提速。

## 证据与复现

- 原始结果：`output/native-runtime-20261101/final-comparison/result.json`。
- 原生进程检查：`native-tests.txt`，7 passed；测试控制器只需 CPU 项目环境和 python-chess，不需要 GPU PyTorch。
- 归档：[摘要](evidence/native-runtime-20261101/summary.json)、[完整对照](evidence/native-runtime-20261101/README.md)。包含源码/二进制/引擎哈希与动态依赖记录，不包含模型、NVIDIA SDK 或二进制。
- 构建使用官方 TensorRT v11.4 头文件、现有 11.4.0.106 Linux 运行库和 CUDA Toolkit 13.4。每个 graph 使用独立 context 并在形状设定后预热，遵循 NVIDIA 的 [C++ runtime 接口](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/getting-started/quick-start-runtime-tutorial.html)与 [CUDA Graph 约束](https://docs.nvidia.com/deeplearning/tensorrt/11.4.1/performance/optimization.html)。
