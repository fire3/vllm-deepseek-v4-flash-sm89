**[English](README-en.md)**

# vLLM DeepSeek V4 (Flash) — SM89 (Ada) 移植

[vLLM DeepSeek V4 (Flash) 稀疏 MLA](https://github.com/vllm-project/vllm) 在 **NVIDIA Ada (SM89, 如 L40S)** 平台的移植总览仓库。

## 关联仓库

| 组件 | 仓库 | 分支 |
| --- | --- | --- |
| 推理框架 | [fire3/vllm](https://github.com/fire3/vllm) | `v0.28.0-sm89` |

SM89 上 DeepSeek V4 Flash 的稀疏 MLA 全链路均由 **Triton** 实现。

## 背景

上游依赖 DeepGEMM、FlashMLA、CuTe-DSL 仅支持 SM90/SM100/SM120，Ada 全链路不可用。本移植从 SGLang 的 SM120 Triton 稀疏 MLA 算子移植出 vLLM 侧的 `TRITON_MLA_SPARSE_DSV4` 后端，在 SM89 上跑通 DeepSeek V4 Flash：稀疏 MLA 的 prefill + decode、FP8 KV cache（`fp8_ds_mla`）、CUDA graph，并已启用 DSpark 投机解码。

## 使用

### 环境

Python 3.12、CUDA 13.0、PyTorch 2.13.0+cu130、L40S。
关键依赖：`tokenspeed-mla`、`humming-kernels`、`tilelang`。

### 安装

按上游标准流程从源码构建（v0.28 起默认经 `uv` + `--torch-backend=auto` 引导编译）：

```bash
git clone git@github.com:fire3/vllm.git -b v0.28.0-sm89
cd vllm
uv pip install -e . --torch-backend=auto
```

要点：

- 编译器需 GCC/G++ ≥ 11.3（PyTorch C++20 头文件不兼容更旧的 GCC）。
- 默认 `CUDA_SUPPORTED_ARCHS` 已含 8.9，L40S（SM89）标准构建即可；只想编目标架构时可用 `TORCH_CUDA_ARCH_LIST=8.9` 缩短构建时间。
- 复用已有 PyTorch（如 conda env 里的 cu130）时，按上游「使用已有 PyTorch」流程：

  ```bash
  python use_existing_torch.py
  uv pip install -r requirements/build/cuda.txt
  uv pip install --no-build-isolation -e .
  ```

- 编译内存受限时用 `MAX_JOBS` 限制并行度（WSL 下建议 `export MAX_JOBS=1`）。
- 稀疏 MLA 后端为纯 Triton 实现。

#### 编译 wheel

构建可分发/可离线安装的 wheel（产物在 `dist/`，`--no-build-isolation` 复用当前环境里的 cu130 torch）：

```bash
uv build --wheel --no-build-isolation
uv pip install dist/vllm-*.whl          # 本机安装，或在其他机子上拷贝后安装
```

要点：

- 目标架构从 PyTorch 环境的 `CMAKE_CUDA_FLAGS` 继承；只想编 SM89 可用 `TORCH_CUDA_ARCH_LIST=8.9 uv build --wheel --no-build-isolation` 缩短构建与缩小 wheel。
- wheel 版本经 setuptools-scm 派生：无 tag 时形如 `vllm-0.28.0.devXXX+gHASH.dYYYYMMDD`，本地构建与上游 nightly 区分开（如需要可自行打 tag）。
- C/C++/Triton 内核改动后需重新 `uv build`（Python 改动可用上文的 `-e .` 可编辑安装，无需重编）。

### 启动服务

环境变量：

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX:-<env>}/lib:$LD_LIBRARY_PATH"   # 需 CXXABI_1.3.15，供 ICU 78
export NCCL_P2P_DISABLE=1
export PATH="/usr/local/cuda/bin:$PATH"
```

启动命令（TP=8 参考，按实际部署调整）：

```bash
vllm serve <model_dir> \
  --served-model-name deepseek-v4-flash \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8_ds_mla \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --kernel-config '{"moe_backend":"humming"}' \
  --attention-backend TRITON_MLA_SPARSE_DSV4 \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --tokenizer-mode deepseek_v4 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --host 0.0.0.0 --port 8091
```

要点：

- `--attention-backend` 必须显式指定 `TRITON_MLA_SPARSE_DSV4`：SM89 上 FlashMLA 后端（仅 SM90/SM100）不可用，只能用 Triton 后端。
- SM89 上 MoE 后端 MXFP4 可选 `humming`，比 Marlin 快约 5%，必须显式指定，`"auto"` 不会选它。
- 262k 上下文比 1M 快约 11%；并发建议 ≤16；`--gpu-memory-utilization` 建议 ≤0.9，过高可能导致 CUDA graph 预留空间不足而启动失败。
- **DSpark 投机解码可用**：DeepSeek-V4 的 DSpark 权重随目标 checkpoint 一并加载，启用只需加 `--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'`（如需完整块验证，把 `draft_sample_method` 换成 `"greedy"`）。

## 移植思路

由 vLLM 侧 **Triton** 实现稀疏 MLA：

1. **decode + prefill**：共用同一个分块双源融合 kernel（query-major、head-blocked，SWA 与压缩源共享 online-softmax 状态）；decode（每行一 token）通过 `triton_sparse_mla_decode_vllm` 薄封装路由到同一 kernel（`triton_sparse_mla_prefill.py`）。
2. **K-split**：小 decode batch（T≤64）按固定默认做 (B, S) CTA 切分 + partial-LSE 合并，更大的 T 退回单 CTA 融合 kernel，无需开关。
3. **后端/路由**：`triton_sparse.py` 提供 `DeepseekV4TritonMLASparseBackend` / `DeepseekV4TritonMLAAttention`（SM89 + SM120），复用同一套 indexer、`fp8_ds_mla` KV page 布局与稀疏索引元数据。
4. **SM89 fallbacks**：DeepGEMM 不可用时 Triton grouped-bf16 `o_proj`/`tf32_hc_prenorm_gemm` fallback、SM89 indexer top-k 的 `fp8_mqa_logits_triton`（fp16 MMA + fp32 accumulate）等。

内核始终运行默认配置（固定 config，确定性、无运行时 benchmark），不做 autotune，也无 A/B 切换开关。

## 开发

```bash
git submodule update --init
```

- 开发阶段推荐可编辑安装；切换分支后按 vllm 上游流程重构建。
- 所有 SM89 Triton 内核/JIT 改动都需同时验证 **decode 与 prefill**。

### 验证

```bash
pytest tests/kernels/attention/test_triton_sparse_mla_prefill.py \
       tests/v1/attention/test_triton_sparse_mla_backend.py
```

联合数值验证 decode + prefill 通过（NH=8/16/32、topk、nt、sink 开关、变长 topk_length）；decode 覆盖在 `test_triton_sparse_mla_prefill.py` 内的 `test_triton_decode_vllm_tiled_matches_reference`。

## 已知限制

- Triton 实现以正确性为先，性能未逐项对齐上游 SGLang SM120 算子（固定 config，不做 autotune，TMA 退化可能带来性能差距）。
- 源码/日志仍可能保留部分架构术语（共享 indexer/KV page 布局与能力探测框架）。

## 维护

- 基线 vLLM `v0.28.0`，上游升级从新 release tag 重新移植。
- 原始开发历史已保留，详细调试文档可从 git 历史恢复。
- 任何 SM89 内核改动都需同时验证 **decode 与 prefill**。
