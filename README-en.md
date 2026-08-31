**[简体中文](README.md)**

# vLLM DeepSeek V4 (Flash) — SM89 (Ada) Port

Overview repository for porting [vLLM DeepSeek V4 (Flash) Sparse MLA](https://github.com/vllm-project/vllm) to **NVIDIA Ada (SM89, e.g. L40S)**.

## Related Repositories

| Component | Repository | Branch |
| --- | --- | --- |
| Inference framework | [fire3/vllm](https://github.com/fire3/vllm) | `v0.28.0-sm89` |

The full Sparse MLA pipeline of DeepSeek V4 Flash on SM89 is implemented with **Triton**.

## Background

Upstream dependencies DeepGEMM, FlashMLA, and CuTe-DSL only support SM90/SM100/SM120, so the full Ada pipeline is unavailable. This port adapts SGLang's SM120 Triton Sparse MLA operator into a vLLM-side `TRITON_MLA_SPARSE_DSV4` backend that runs DeepSeek V4 Flash on SM89: Sparse MLA prefill + decode, FP8 KV cache (`fp8_ds_mla`), CUDA graph, with DSpark speculative decoding enabled.

## Usage

### Environment

Python 3.12, CUDA 13.0, PyTorch 2.13.0+cu130, L40S.
Key dependencies: `tokenspeed-mla`, `humming-kernels`, `tilelang`.

### Installation

Build from source following the standard upstream flow (from v0.28 on, the build bootstraps via `uv` + `--torch-backend=auto`):

```bash
git clone git@github.com:fire3/vllm.git -b v0.28.0-sm89
cd vllm
uv pip install -e . --torch-backend=auto
```

Key points:

- The compiler needs GCC/G++ >= 11.3 (PyTorch C++20 headers are incompatible with older GCC).
- The default `CUDA_SUPPORTED_ARCHS` already includes 8.9, so a standard L40S (SM89) build works; to build only the target arch, use `TORCH_CUDA_ARCH_LIST=8.9` to shorten build time.
- When reusing an existing PyTorch (e.g. cu130 in a conda env), follow the upstream "use existing PyTorch" flow:

  ```bash
  python use_existing_torch.py
  uv pip install -r requirements/build/cuda.txt
  uv pip install --no-build-isolation -e .
  ```

- Limit parallelism with `MAX_JOBS` when build memory is constrained (recommended `export MAX_JOBS=1` under WSL).
- The sparse-MLA backend is a pure Triton implementation.

#### Building a wheel

Built for a distributable/offline-installable wheel (artifacts in `dist/`; `--no-build-isolation` reuses the cu130 torch in the current environment):

```bash
uv build --wheel --no-build-isolation
uv pip install dist/vllm-*.whl          # install locally, or copy to another machine and install
```

Key points:

- Target archs are inherited from the PyTorch environment's `CMAKE_CUDA_FLAGS`; to build only SM89, use `TORCH_CUDA_ARCH_LIST=8.9 uv build --wheel --no-build-isolation` to shorten the build and shrink the wheel.
- The wheel version is derived by setuptools-scm: with no tag it looks like `vllm-0.28.0.devXXX+gHASH.dYYYYMMDD`, distinguishing local builds from upstream nightlies (you can tag yourself if needed).
- Re-run `uv build` after C/C++/Triton kernel changes (Python-only changes can use the `-e .` editable install above without a rebuild).

### Starting the service

Environment variables:

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX:-<env>}/lib:$LD_LIBRARY_PATH"   # needs CXXABI_1.3.15, for ICU 78
export NCCL_P2P_DISABLE=1
export PATH="/usr/local/cuda/bin:$PATH"
```

Start command (TP=8 for reference; adjust for your deployment):

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

Key points:

- `--attention-backend` must be explicitly set to `TRITON_MLA_SPARSE_DSV4`: the FlashMLA backend (SM90/SM100 only) is unavailable on SM89, so only the Triton backend can be used.
- On SM89 the MXFP4 MoE backend can be set to `humming`, which is about 5% faster than Marlin; it must be set explicitly, `"auto"` will not select it.
- 262k context is about 11% faster than 1M; recommended concurrency <= 16; `--gpu-memory-utilization` recommended <= 0.9, higher values may leave too little headroom for CUDA graph reservation and fail startup.
- **DSpark speculative decoding works**: DeepSeek-V4's DSpark weights are loaded together with the target checkpoint; enable it by adding `--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'` (swap `draft_sample_method` to `"greedy"` if full-block verification is desired).

## Porting approach

Sparse MLA implemented in Triton on the vLLM side:

1. **decode + prefill**: share a single chunked dual-source fused kernel (query-major, head-blocked; SWA and compressed sources share the online-softmax state); decode (one token per row) is routed to the same kernel via the thin `triton_sparse_mla_decode_vllm` wrapper (`triton_sparse_mla_prefill.py`).
2. **K-split**: small decode batches (T<=64) use a fixed default (B, S) CTA split with partial-LSE merging; larger T falls back to a single-CTA fused kernel, no toggle needed.
3. **Backend/routing**: `triton_sparse.py` provides `DeepseekV4TritonMLASparseBackend` / `DeepseekV4TritonMLAAttention` (SM89 + SM120), reusing the same indexer, `fp8_ds_mla` KV page layout, and sparse index metadata.
4. **SM89 fallbacks**: Triton grouped-bf16 `o_proj` / `tf32_hc_prenorm_gemm` fallbacks when DeepGEMM is unavailable, and `fp8_mqa_logits_triton` (fp16 MMA + fp32 accumulate) for the SM89 indexer top-k, among others.

The kernels always run with the default configuration (fixed config; deterministic, no runtime benchmarking), with no autotune and no A/B switches.

## Development

```bash
git submodule update --init
```

- Editable install is recommended during development; rebuild following the upstream vllm flow after switching branches.
- All SM89 Triton kernel/JIT changes must be verified for both **decode and prefill**.

### Verification

```bash
pytest tests/kernels/attention/test_triton_sparse_mla_prefill.py \
       tests/v1/attention/test_triton_sparse_mla_backend.py
```

Combined numerical verification of decode + prefill passes (NH=8/16/32, topk, nt, sink toggles, variable-length topk_length); decode coverage is in `test_triton_decode_vllm_tiled_matches_reference` inside `test_triton_sparse_mla_prefill.py`.

## Known limitations

- The Triton implementation prioritizes correctness; performance is not aligned kernel-by-kernel with the upstream SGLang SM120 operator (fixed config, no autotune; some performance gap may come from TMA fallback).
- Source code/logs may still retain some architecture terms (shared indexer/KV page layout and capability-probing framework).

## Maintenance

- The baseline is vLLM `v0.28.0`; upstream upgrades are re-ported from new release tags.
- The original development history is preserved; detailed debugging docs can be recovered from git history.
- Any SM89 kernel change must be verified for both **decode and prefill**.
