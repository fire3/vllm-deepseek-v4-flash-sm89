# vLLM DeepSeek V4 (Flash) — SM89 (Ada) 移植

[vLLM DeepSeek V4 (Flash) 稀疏 MLA](https://github.com/vllm-project/vllm) 在 **NVIDIA Ada (SM89, 如 L40S)** 平台的移植总览仓库。

## 关联仓库

| 组件 | 仓库 | 分支 |
| --- | --- | --- |
| 推理框架 | [fire3/vllm](https://github.com/fire3/vllm) | `v0.28.0-sm89` |

SM89 上 DeepSeek V4 Flash 的稀疏 MLA 全链路均由 **Triton** 实现。

## 背景

上游依赖 DeepGEMM、FlashMLA、CuTe-DSL 仅支持 SM90/SM100/SM120，Ada 全链路不可用。本移植从 SGLang 的 SM120 Triton 稀疏 MLA 算子移植出 vLLM 侧的 `TRITON_MLA_SPARSE_DSV4` 后端，在 SM89 上跑通 DeepSeek V4 Flash：稀疏 MLA 的 prefill + decode、FP8 KV cache（`fp8_ds_mla`）、CUDA graph。DSpark 投机解码当前 SM89 尚未启用。

## 使用

### 环境

Python 3.12、CUDA 13.0、PyTorch 2.13.0+cu130、L40S。
关键依赖：`tokenspeed-mla`、`humming-kernels`、`tilelang`。

### 安装

在 vllm 仓库内按上游常规流程构建 wheel（`TORCH_CUDA_ARCH_LIST=8.9+PTX` 等，产物带 `+sm89` local 标记），或按该分支 README / 构建文档执行。

要点：

- vLLM 构建用 `--no-build-isolation` 复用 conda env 里的 cu130 torch，且默认 `FETCHCONTENT_FULLY_DISCONNECTED=ON` 直接使用 `.deps`，缺失会报错；需要联网首次下载时可设 `FETCHCONTENT_FULLY_DISCONNECTED=0`。
- 稀疏 MLA 后端为纯 Triton 实现。

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
- **DSpark 投机解码不可用**：勿加 `--speculative-config '{"method":"dspark",...}'`——SM89 上 CUDA graph 重放仍有未根因的异步越界问题。

## 移植思路

由 vLLM 侧 **Triton** 实现稀疏 MLA：

1. **decode**：移植自 SGLang SM120 的 `flash_mla_sparse_decode_triton` 分块稀疏 MLA decode 内核（`triton_sparse_mla_decode.py`）。
2. **prefill**：phase-2A 分块双源融合 prefill 内核（可选 K-split），prefill token 按 decode 风格逐行展开，`triton_sparse_mla_prefill.py`。
3. **后端/路由**：`triton_sparse.py` 提供 `DeepseekV4TritonMLASparseBackend` / `DeepseekV4TritonMLAAttention`（SM89 + SM120），复用同一套 indexer、`fp8_ds_mla` KV page 布局与稀疏索引元数据。
4. **SM89 fallbacks**：DeepGEMM 不可用时 Triton grouped-bf16 `o_proj`/`tf32_hc_prenorm_gemm` fallback、SM89 indexer top-k 的 `fp8_mqa_logits_triton`（fp16 MMA + fp32 accumulate）等。

Triton decode kernel 默认用固定 config（确定性、无运行时 benchmark）；也可通过环境变量开启 autotune 或切换旧流程做 A/B 对比：

- `VLLM_TRITON_SPARSE_MLA_DECODE_AUTOTUNE=1` — decode kernel 镜像上游 SGLang 算子的 autotune。
- `VLLM_TRITON_SPARSE_MLA_PREFILL_AUTOTUNE=1` — prefill kernel autotune。
- `VLLM_TRITON_SPARSE_MLA_KSPLIT=0` — 关闭 (B, S) CTA 切分，退回单 CTA 融合 kernel（小 decode batch 时提高并行度，默认开）。
- `VLLM_TRITON_SPARSE_MLA_PREFILL_DECODE_WRAPPER=1` — prefill 改用 decode-wrapper 启动器做 A/B 对比。
- `VLLM_TRITON_SPARSE_MLA_DECODE_LEGACY=1` — decode 退回 phase-1 elementwise 两遍 kernel。

## 开发

```bash
git submodule update --init
```

- 开发阶段推荐可编辑安装；切换分支后按 vllm 上游流程重构建。
- 所有 SM89 Triton 内核/JIT 改动都需同时验证 **decode 与 prefill**。

### 验证

```bash
pytest tests/kernels/attention/test_triton_sparse_mla_decode.py \
       tests/kernels/attention/test_triton_sparse_mla_prefill.py \
       tests/v1/attention/test_triton_sparse_mla_backend.py
```

联合数值验证 decode + prefill 通过（NH=8/16/32、topk、nt、sink 开关、变长 topk_length）。

## 已知限制

- Triton 实现以正确性为先，性能未逐项对齐上游 SGLang SM120 算子（如未开启 autotune、TMA 退化可能带来性能差距）。
- DSpark 投机解码未启用（CUDA graph 重放异步越界问题未根因）。
- 源码/日志仍可能保留部分架构术语（共享 indexer/KV page 布局与能力探测框架）。

## 维护

- 基线 vLLM `v0.28.0`，上游升级从新 release tag 重新移植。
- 原始开发历史已保留，详细调试文档可从 git 历史恢复。
- 任何 SM89 内核改动都需同时验证 **decode 与 prefill**。
