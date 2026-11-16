# Model and runtime

The network represents a position as 64 square tokens and metadata. A 16-layer Transformer with 896 channels and 14 attention heads produces a board representation. The model has 155,691,271 parameters.

- **Policy:** scores each legal move.
- **Action value:** predicts a distribution over the value of each legal move using 128 bins.
- **Position value:** predicts win, draw and loss logits for the side to move.

PUCT combines the network outputs with tree search. The native runtime batches leaf evaluations, caches network results and supports UCI. Its action-value prior uses the `--q-prior-beta` setting. See the [model implementation](../neurochess/model.py) and [search implementation](../engine/src/mcts.hpp) for details.

The [model release](https://github.com/ddy314/NeuroChess/releases/tag/model-v1) contains PyTorch inference weights, 622,819,723 bytes. SHA-256: `910e517774770754500ff3ad6b66380d0bf3d36f5e1f85f7b2451407b95c18a7`. Load checkpoints from sources you trust.

The export command in the [README](../README.md#get-started) calibrates on 1,024 positions, exports ONNX and builds a TensorRT plan with INT8 Transformer matrix multiplications. The exporter requires a CUDA-capable GPU, ONNX, ml_dtypes and TensorRT Python bindings. The native engine also requires the TensorRT SDK and CUDA Toolkit. GPU support and library versions depend on the local installation. A plan is specific to its TensorRT and GPU environment; build it where it will run.

`python -m neurochess.predict` performs a single network evaluation. The UCI executable performs search. Its default history-free configuration and fixed simulation budget are shown in the README. The historical locked two-second baseline uses a different model and configuration; its [record](SEARCH_BASELINE.md) governs future search optimization comparisons.
