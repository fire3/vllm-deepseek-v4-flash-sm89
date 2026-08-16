# vLLM DeepSeek V4 (Flash) — SM89 (Ada) 移植版

本仓库是 vLLM DeepSeek V4 (Flash) 稀疏 MLA 在 **NVIDIA Ada (SM89, 如 L40S)** 平台移植工作的**总览仓库**，面向两位读者：**使用者**（如何部署运行）与**开发者**（移植思路、开发调试、如何继续迭代）。技术细节见下方关联的两个分支文档。

## 关联仓库与分支

| 组件 | 仓库 | 分支 | 详细文档 |
| --- | --- | --- | --- |
| 推理框架 | `github.com/fire3/vllm` | `v0.27.0-dsv4-sm89` | [README_SM89.md](https://github.com/fire3/vllm/blob/v0.27.0-dsv4-sm89/README_SM89.md) |
| 内核/底层库 | `github.com/fire3/flashinfer` | `v0.6.16.post3-dev-sm89-dsv4` | [README_SM89.md](https://github.com/fire3/flashinfer/blob/v0.6.16.post3-dev-sm89-dsv4/README_SM89.md) |

vLLM 侧负责调用策略与能力探测，FlashInfer 侧负责稀疏 MLA 内核的 SM89 适配，两端**必须锁步升级**。

## 背景

上游 vLLM 的 DeepSeek V4 稀疏 MLA 依赖 DeepGEMM、FlashMLA、CuTe-DSL，这些仅支持 SM90/SM100/SM120，Ada 全链路不可用。本移植的目标是在 SM89 上跑通 DeepSeek V4 Flash：`FLASHINFER_MLA_SPARSE_DSV4` 注意力后端、FP8 KV cache、稀疏 MLA 的 prefill + decode、CUDA graph、DSpark 投机解码（当前 SM89 尚未启用）。

## 使用方法

### 环境（推荐基线）

Python 3.12、CUDA 13.0、PyTorch 2.13.0+cu130、L40S。

关键依赖：`tokenspeed-mla`、`humming-kernels`、`tilelang`、`flashinfer-python`（各仓库的 `scripts/build/sm89_*.sh` 脚本可自动完成依赖安装与 wheel 构建）。

### 安装

```bash
# 1. conda 环境中安装依赖（一次性）
bash scripts/build/sm89_install_deps.sh

# 2. 拉取三方依赖（每次切换分支后重跑）
bash scripts/build/sm89_prepare_deps.sh

# 3. 构建 wheel（AOT 方式，产物带 +sm89 标记）
bash scripts/build/sm89_build_wheel.sh
# 或 JIT-cache wheel，运行时免 nvcc/ninja：
bash scripts/build/sm89_build_jit_cache_wheel.sh
# 或开发模式（editable + 运行时 JIT，Python 改动即时生效）：
bash scripts/build/sm89_build_wheel.sh --editable
```

**顺序**：先装 FlashInfer，后装 vLLM。版本配对 vLLM `0.27.1+sm89` + FlashInfer `0.6.16.post3+sm89`（两者均为 PEP 440 `+sm89` local 标记，specifier 匹配会忽略 local 版本）。切勿用上游 flashinfer 替代——其缺少 SM89 稀疏 MLA，vLLM 启动校验会失败。

### 启动服务

示例（TP=8）：

```bash
vllm serve <model> \
  --kv-cache-dtype fp8_ds_mla \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --kernel-config '{"moe_backend":"humming"}' \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --enable-auto-tool-choice
```

要点：
- `--attention-backend` 必须显式指定 `FLASHINFER_MLA_SPARSE_DSV4`（内部通过 `has_flashinfer_sparse_mla_sm89()` 探测确认 SM89 sparse 可用后才放行）。
- MoE 后端在 SM89 上 MXFP4 可选 `humming`；实测 8×L40S、1M ctx、8 并发时，humming-indexed 比 Marlin 快约 5%，必须显式指定，`"auto"` 不会选它。
- 262k 上下文比 1M 快约 11%；并发上限建议 ≤16（24 时 TTFT 会升到秒级）；`--gpu-memory-utilization` 建议 ≤0.91（CUDA graph 预留）。
- **DSpark 投机解码当前不可用**：不要添加 `--speculative-config`——CUDA graph replay 在长运行下仍有未根因的异步越界问题，仅覆盖了已复现场景。

## 移植思路

不重写整套内核，而是**共享 FlashInfer `sparse_mla_sm120` 内核主体，通过架构 shim（`SPARSE_MLA_USE_SM89_PRIMS` 宏）把 SM120 专属原语收敛到小范围兼容实现**，同时在三重"身份"上显式区分 SM89 / SM120：

1. **JIT module 身份**：JIT 产物按架构命名（如 `sparse_mla_sm89`），避免跨架构 `.so` 混名。
2. **Autotune / cache 身份**：decode 缓存文件名与签名按架构、page block size 区分，避免跨架构复用次优策略。
3. **运行时路由**：`backend="auto"` 在 SM89 + `sparse_mla_top_k > 0` 时正确落到 sparse 实现。

主要适配点：

- **FP8 block-scale MMA** `mma_fp8_block_scaled_m16n8k32`（SM120 专属）→ 普通 FP8 MMA + 软件重建 UE8M0 scale；
- **TMA / cp.async.bulk** → 16B 循环 copy，`cp.async` 的 L2 hint 退化为普通 `cp.async.cg`；
- **mbarrier / LDGSTSBAR** 等待语义改为 SM89 兼容版本（decode 用 `mbarrier_arrive`，prefill IO gather 用 `cp_async_wait_all` + bar_sync + lane0 单次 `mbarrier_arrive`）；
- **NH=8**（TP=8 时每 rank 8 头）→ 移植 padded 16-head MG 方案，8 头时上 8 行零填充，不触碰调用方全局内存；
- **cudaLaunchKernelExC + __grid_constant__** 在 SM89 被驱动拒绝 → 改经典 `kernel<<<grid, block, smem, stream>>>` 启动；
- **fastdiv** 恢复 `cuda::fast_mod_div<uint32_t>` 包装（依赖 `<cuda/cmath>`/cccl，缺失回退普通除法会导致注意力/MLA 热路径性能回退约 1.76~2.1x）。

## 开发与调试

### 环境

- CUDA_HOME / `TORCH_CUDA_ARCH_LIST=8.9+PTX` / `FLASHINFER_DISABLE_VERSION_CHECK=1`，conda env 的 lib 加入 `LD_LIBRARY_PATH`（ICU 78 需要 CXXABI_1.3.15），`/usr/local/cuda/bin` 加入 `PATH`。
- JIT include 路径需要 `flashinfer/data/cccl/` 非空（来自 `3rdparty/cccl` submodule）；缺失会导致 `<cuda/cmath>` 编译失败（`fastdiv.cuh` 依赖）。从 wheel 安装、未初始化 submodule 时，可从任一已装 flashinfer 的环境拷贝该目录。

### 开发工作流

- 推荐两端均以 **editable 安装**：Python 改动即时生效，C++/CUDA 改动需重新构建对应扩展，接口改动需两个仓库分支同步提交。
- 分支版本由 git tag 派生并加 `+sm89` local 后缀；切换分支后需重跑 `sm89_prepare_deps.sh` 把三方依赖拉到当前锁定版本。

### 验证

```bash
# FlashInfer 侧单元/路由测试
pytest tests/attention/test_sparse_mla_sm89_gate.py tests/attention/test_sparse_mla_sm89_decode.py
```

联合数值验证（vLLM 侧 `scripts/build/sm89_precision_verify.py` 等）：decode + prefill **37/37 PASS**（NH=8/16/32、topk=128/512、nt≤2048、sink 开/关），输出误差 ≤2.1e-3、LSE ≤1e-4；fastdiv 在 90,037 个除数 × 2048 个 n 共 1.84 亿次 divmod 上 **0 错误**。

## 已知限制

- 走"共用 SM120 主体 + 架构 shim"路线，性能以正确性为先，未与 SM120 逐项对齐（如 TMA 退化可能带来性能差距）。
- DSpark 投机解码未启用（CUDA graph replay 仍存在未根因的异步越界问题）。
- 源码/日志仍保留 `sm120` 术语（JIT 与 cache 身份已区分），长期建议命名收口。

## 维护

- 基线为 vLLM `v0.27.0`，上游升级时应从新的 release tag 重新移植。
- vLLM 与 FlashInfer 两个分支必须锁步提交/升级。
- 原始开发历史（含调试往返提交）已保留，详细的调试文档可从 git 历史恢复。
- 任何 SM89 内核/JIT 改动都需同时验证 **decode 与 prefill** 两条路径，避免 mbarrier / `cp.async.bulk` 兼容性回退。

---

更多细节：见 [vLLM 侧 README_SM89](https://github.com/fire3/vllm/blob/v0.27.0-dsv4-sm89/README_SM89.md)（编译与服务方法）与 [FlashInfer 侧 README_SM89](https://github.com/fire3/flashinfer/blob/v0.6.16.post3-dev-sm89-dsv4/README_SM89.md)（内核移植与数值验证）。
