# C++/CUDA 原生推理运行时

训练保持 PyTorch，不改结构、监督或检查点。离线导出仍可用 Python；实际落子进程没有 Python、NumPy、PyTorch 或 LibTorch 依赖。当前实现为实验后端，未替换冻结的两秒 PUCT 基线。

模型计算复用 TensorRT 的 CUDA 内核；输入缓冲区、批处理、缓存、PUCT 调度和计时由 C++ 执行，合法着法遮罩、稳定 top-5、Q/WDL 融合使用手写 CUDA。并非从零重写所有矩阵乘算子。

## 构建

需要 C++17、CUDA 编译器和匹配引擎版本的 TensorRT C++ 头文件/运行库。本次验证使用 TensorRT 11.4.0.106、CUDA Toolkit 13.4、RTX 4060 Laptop（SM89）。训练及原 Python 环境无需修改。

TensorRT 头文件来自官方仓库的 `v11.4`，提交 `16ed0636df41e4a27d46f82d9028fc50fadc7200`，保存在实验目录，未复制进项目源码。已下载的 Linux 运行库位于 `output/puct-10x-20261031/trt-deps/tensorrt_libs/`；这里只加载原生 `.so`，不加载 Python 模块。

```sh
cmake -S neurochess/native -B output/native-runtime-20261101/build \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DTENSORRT_ROOT="$PWD/output/native-runtime-20261101/tensorrt-sdk" \
  -DTENSORRT_LIB_DIR="$PWD/output/puct-10x-20261031/trt-deps/tensorrt_libs"
cmake --build output/native-runtime-20261101/build -j 2
```

如果需要重新取得头文件：

```sh
git clone --depth 1 --filter=blob:none --sparse --branch v11.4 \
  https://github.com/NVIDIA/TensorRT.git output/native-runtime-20261101/tensorrt-sdk
git -C output/native-runtime-20261101/tensorrt-sdk sparse-checkout set include
```

换设备或运行时版本时需要重新核对/生成引擎，不能把模型 `.pt` 文件直接作为 TensorRT plan 传入。

## 运行

最新选中模型的现有引擎来自累计 15,500 步的 `selected.pt`；与此前 0.5 秒对 Stockfish 测试使用的是同一份 plan。

```sh
output/native-runtime-20261101/build/neurochess_native \
  --engine output/selected-05s-sf-20261101/engine/model.plan \
  --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' \
  --time .5 --history-free
```

输出一行 JSON，包含着法、实际搜索时间、真实网络输入数量、缓存和原生搜索统计。初始化/模型预热时间单独写入 stderr。固定模拟次数的对照可使用 `--time -1 --simulations 512`；`--cache 0` 关闭缓存，默认容量 32768，默认叶批次 8。

加 `--uci` 运行持久进程。实验接口支持 `uci`、`isready`、`ucinewgame`、`position startpos/fen ... moves ...`、`go movetime`、`go nodes`、`stop`、`quit`。`go nodes` 指 MCTS 模拟数，不等同于 Stockfish 搜索节点。本版用于固定步时或固定模拟次数，未实现常规比赛时钟分配、ponder 和 MultiPV。Python 可作为外部比赛控制器，但不进入落子计算路径。

`--history-free` 只应对确知忽略历史特征的模型使用：它在神经输入缓存键中屏蔽相应特征，不合并规则节点、重复局面证明或搜索树。未传此开关时缓存保留完整输入，默认更保守。

## 运行时结构

- 原 C++ `mcts_pipeline.cpp` 原样复用，保留规则、PUCT、价值回传、强制终局证明、部分结果回传和虚拟损失取消。
- C++ LRU 只缓存完整有效神经输入及合法着法顺序。命中立即回传，GPU 批次只装唯一未命中项。
- 每个批次尺寸使用独立 TensorRT context、固定 GPU 缓冲区、固定页锁定 CPU 缓冲区以及 CUDA Graph。
- Graph 包含 H2D、模型执行、CUDA 输出头融合和 D2H；每种可能的部分批次均在计时前预热。
- 采用非阻塞 stream，停止或异常时撤销所有未完成的虚拟预留；非法历史失败后重建状态。

TensorRT 官方运行时接口支持直接从 C++ 加载 plan 并调用 `enqueueV3()`；图捕获需在形状更新后预热，并保持 context 和缓冲区地址稳定。[运行时接口](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/getting-started/quick-start-runtime-tutorial.html)、[CUDA Graph 约束](https://docs.nvidia.com/deeplearning/tensorrt/11.4.1/performance/optimization.html)。

## 检查与对照

```sh
NEUROCHESS_NATIVE_BINARY="$PWD/output/native-runtime-20261101/build/neurochess_native" \
NEUROCHESS_NATIVE_PLAN="$PWD/output/selected-05s-sf-20261101/engine/model.plan" \
  .venv/bin/python -m pytest tests/test_native_runtime.py -q
```

这些测试使用外部原生进程，不要求测试控制器安装 GPU PyTorch。未设置路径时默认跳过。

`scripts/benchmark_native_runtime.py` 是独立 QA/计时控制器，比较同一 TensorRT plan 下的输出、固定搜索次数与固定时间预算。运行器本身可以用 Python，不表示原生棋手内部使用 Python。详见 [本轮结果](../reports/NATIVE_INFERENCE_20261101.md)。
