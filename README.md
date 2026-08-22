# vLLM DeepSeek V4 (Flash) — SM89 (Ada) 移植

[vLLM DeepSeek V4 (Flash) 稀疏 MLA](https://github.com/vllm-project/vllm) 在 **NVIDIA Ada (SM89, 如 L40S)** 平台的移植总览仓库。

## 关联仓库

| 组件 | 仓库 | 分支 |
| --- | --- | --- |
| 推理框架 | [fire3/vllm](https://github.com/fire3/vllm) | `v0.27.0-sm89` |
| 内核/底层库 | [fire3/flashinfer](https://github.com/fire3/flashinfer) | `v0.6.17-sm89` |

vLLM 侧负责调用策略与能力探测，FlashInfer 侧负责稀疏 MLA 内核的 SM89 适配，两端必须锁步升级。

## 背景

上游依赖 DeepGEMM、FlashMLA、CuTe-DSL 仅支持 SM90/SM100/SM120，Ada 全链路不可用。本移植目标是在 SM89 上跑通 DeepSeek V4 Flash：`FLASHINFER_MLA_SPARSE_DSV4` 注意力后端、FP8 KV cache、稀疏 MLA 的 prefill + decode、CUDA graph。DSpark 投机解码当前 SM89 尚未启用。

## 使用

### 环境

Python 3.12、CUDA 13.0、PyTorch 2.13.0+cu130、L40S。
关键依赖：`tokenspeed-mla`、`humming-kernels`、`tilelang`、`flashinfer-python`（各仓库的 `scripts/build/sm89_*.sh` 脚本可自动安装依赖并构建 wheel）。

### 安装

两端均有独立的 `scripts/build/sm89_*.sh` 脚本集，构建目标统一为 `FLASHINFER_CUDA_ARCH_LIST=8.9`（vLLM 侧等价 `TORCH_CUDA_ARCH_LIST=8.9+PTX`），wheel 均带 `+sm89` local 标记。**顺序：先 FlashInfer，后 vLLM**。

#### FlashInfer（内核/底层库）

```bash
cd flashinfer
conda activate <env>          # 或 ENV_NAME=flashinfer-sm89 由脚本建环境

# 1. 一次性：装 build backend 依赖 + cu130 torch + requirements
bash scripts/build/sm89_install_deps.sh

# 2. 每次切换分支后：初始化三方依赖 submodule（cutlass/spdlog/cccl）
bash scripts/build/sm89_prepare_deps.sh
bash scripts/build/sm89_prepare_deps.sh --check    # 只校验不修改

# 3a. 开发模式（editable，Python 改动即时生效；C++/CUDA 改动需重跑）
bash scripts/build/sm89_build_wheel.sh --editable

# 3b. 或构建分发 wheel
bash scripts/build/sm89_build_wheel.sh           # AOT: dist/flashinfer_python-0.6.17+sm89-*.whl
bash scripts/build/sm89_build_jit_cache_wheel.sh # JIT-cache 平台 wheel（运行时免 nvcc/ninja）
```

#### vLLM（推理框架）

```bash
cd vllm
conda activate <env>          # 脚本要求已激活的 conda 环境

# 1. 一次性：装 torch 2.13.0+cu130 + build 工具链（--runtime 另同步运行时依赖）
bash scripts/build/sm89_install_deps.sh
#    --runtime 会按 requirements 把 flashinfer/tilelang 等对齐到锁定版本

# 2. 每次切换分支后：按当前 checkout 的 CMake pins 同步 .deps
#    （cutlass/deepgemm/flashmla/vllm-flash-attn 等约 10 个仓库）
bash scripts/build/sm89_prepare_deps.sh
bash scripts/build/sm89_prepare_deps.sh --gpu --list      # 查看清单
bash scripts/build/sm89_prepare_deps.sh --gpu --check     # 校验现有 .deps

# 3. 构建 wheel（产物 dist/vllm-0.27.1+sm89-*.whl；--editable 额外做 editable 安装）
bash scripts/build/sm89_build_wheel.sh
# 或开发模式：
bash scripts/build/sm89_build_wheel.sh --editable
```

要点：

- vLLM 构建用 `--no-build-isolation` 复用 conda env 里的 cu130 torch，且默认 `FETCHCONTENT_FULLY_DISCONNECTED=ON` 直接使用 `.deps`，缺失会报错；需要联网首次下载时可设 `FETCHCONTENT_FULLY_DISCONNECTED=0`。
- FlashInfer 的 editable 安装已将 `flashinfer/data/cccl` 映射到 `3rdparty/cccl`，JIT include 路径可用；wheel 安装则需另备 cccl 头（见开发一节）。
- 版本配对 vLLM `0.27.1+sm89` + FlashInfer `0.6.17+sm89`（PEP 440 `+sm89` local 标记，specifier 匹配忽略 local 版本）。切勿用上游 flashinfer——其缺 SM89 稀疏 MLA，启动校验 `has_flashinfer_sparse_mla_sm89()` 会失败。

### 启动服务

环境变量：

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX:-<env>}/lib:$LD_LIBRARY_PATH"   # 需 CXXABI_1.3.15，供 ICU 78
export NCCL_P2P_DISABLE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
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
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --tokenizer-mode deepseek_v4 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --host 0.0.0.0 --port 8091
```

要点：

- `--attention-backend` 必须显式指定 `FLASHINFER_MLA_SPARSE_DSV4`（内部用 `has_flashinfer_sparse_mla_sm89()` 探测确认后才放行）。
- SM89 上 MoE 后端 MXFP4 可选 `humming`，比 Marlin 快约 5%，必须显式指定，`"auto"` 不会选它。
- 262k 上下文比 1M 快约 11%；并发建议 ≤16；`--gpu-memory-utilization` 建议 ≤0.9，过高可能导致 CUDA graph 预留空间不足而启动失败。
- **DSpark 投机解码不可用**：勿加 `--speculative-config '{"method":"dspark",...}'`——SM89 上 CUDA graph 重放仍有未根因的异步越界问题。

## 移植思路

不重写整套内核，而是**共享 FlashInfer `sparse_mla_sm120` 内核主体，通过架构 shim（`SPARSE_MLA_USE_SM89_PRIMS` 宏）把 SM120 专属原语收敛到 SM89 兼容实现**，三重"身份"显式区分 SM89/SM120：

1. **JIT module 身份**：产物按架构命名（如 `sparse_mla_sm89`），避免跨架构 `.so` 混名。
2. **Autotune / cache 身份**：decode 缓存文件名与签名按架构、page block size 区分，避免跨架构复用次优策略。
3. **运行时路由**：`backend="auto"` 在 SM89 + `sparse_mla_top_k > 0` 时正确落到 sparse 实现。

适配要点：SM120 专属指令换成 SM89 兼容原语（FP8 block-scale MMA → 普通 FP8 MMA + UE8M0 scale 软件重建，TMA/`cp.async.bulk` 退化为逐段 copy，mbarrier 用 SM89 合法写法），补齐 NH=8（TP=8 时每 rank 8 头，padded 16-head 方案）与 SM89 合法的 kernel 启动方式。

## 开发

```bash
git submodule update --init --recursive
```

- 开发阶段推荐两端 **editable 安装**；切换分支后重跑 `sm89_prepare_deps.sh`。
- 运行时 JIT 需要 `FLASHINFER_DISABLE_VERSION_CHECK=1`，且 `flashinfer/data/cccl/` 非空（来自 `3rdparty/cccl` submodule；从 wheel 安装未初始化 submodule 时，可从任一已装 flashinfer 的环境拷贝该目录）。

### 验证

```bash
pytest tests/attention/test_sparse_mla_sm89_gate.py tests/attention/test_sparse_mla_sm89_decode.py
```

联合数值验证（vLLM 侧 `scripts/build/sm89_verify.sh` / `sm89_precision_verify.py`）：decode + prefill **37/37 PASS**（NH=8/16/32、topk=128/512、nt≤2048、sink 开/关、变长 topk_length），误差 ≤2.1e-3、LSE ≤1e-4。

## 已知限制

- "共用 SM120 主体 + 架构 shim"以正确性为先，未与 SM120 逐项对齐（如 TMA 退化可能带来性能差距）。
- DSpark 投机解码未启用（CUDA graph 重放异步越界问题未根因）。
- 源码/日志仍保留 `sm120` 术语（JIT 与 cache 身份已区分）。

## 维护

- 基线 vLLM `v0.27.0`，上游升级从新 release tag 重新移植。
- vLLM 与 FlashInfer 两端必须锁步提交/升级。
- 原始开发历史已保留，详细调试文档可从 git 历史恢复。
- 任何 SM89 内核/JIT 改动都需同时验证 **decode 与 prefill**。
