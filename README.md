# LLM Hardware Calculator

MoonBit-native GGUF inspection and tensor-level memory placement planning, with a lightweight parameter-only calculator as a fallback.

**Status: engineering preview, 17 October 2026.** The artifact pipeline reads real GGUF directories, validates supported model structures, and plans actual tensor payload placement for a pinned llama.cpp policy. Two real models have been compared with the official GGUF reader and loaded on a CPU backend. A first seven-case CUDA run on a WSL2 RTX 5060 Ti passed the recorded placement and rounded model/KV buffer checks, using five official-model cases and two derived tied-output cases. No versioned release or general hardware accuracy guarantee is claimed.

## Inspect a real model

Install the matching [MoonBit toolchain](docs/toolchain.md), then run:

```sh
moon run src/cmd/llmcalc --target native -- inspect /path/to/model.gguf --json
```

Inspection returns metadata summaries, tensor names, dimensions, storage types, exact byte counts, offsets and I/O statistics. It reads the header and directory, not the multi-GB weight payload. Unknown tensor types remain visible with incomplete sizes; malformed counts, overflow, truncated or overlapping known spans are errors. Raw 64-bit integers are JSON decimal strings, preserving offsets above `2^53`.

The reusable pure MoonBit `gguf` package accepts a bounded `ReaderAt`; the native file adapter uses a small POSIX C shim only for OS file access. See [format support, limits and API](docs/gguf.md).

## Plan actual tensors

```sh
moon run src/cmd/llmcalc --target native -- plan /path/to/model.gguf \
  --vram 16GiB --ram 64GiB --context 8K --json
```

The first supported profiles are dense, full-attention `llama` and `qwen2` with the documented tensor/shape subset. The planner derives architecture values from GGUF metadata and checks them against real tensor dimensions. It separately accounts for embeddings, output tensors, nonuniform transformer blocks and tied-output copies. It rejects incomplete sizes, unrecognized tensor roles and unsupported architecture features instead of falling back to an average layer size.

The pinned backend revision is `88aaf325a7026ac1127d0897394b0a3d6413dc2a`. Its offload slots are **output-first**: `n_gpu_layers=1` offloads output only, then subsequent slots offload the last transformer blocks. Input embeddings remain on the CPU; the highest supported result is therefore called `MAX_OFFLOAD`, not full model residency on the GPU. Tied output on the GPU requires an additional embedding copy.

By default, `plan` searches for the largest fitting placement. Add `--n-gpu-layers INTEGER` to evaluate exactly that request, from `0` through the file's `block_count + 1`. A fixed request that exceeds a GPU or RAM steady/loading budget returns `INSUFFICIENT` with a null plan; it does not fall back to another placement. JSON preserves the request in `input.n_gpu_layers`, or null when search was requested. `inspect` does not accept this flag.

Reports include every tensor placement, exact payload conservation, requested and allocated KV, separate GPU/RAM ledgers, steady and loading budgets, source revision and assumptions. Context allocation is padded to 256 tokens for this pinned single-sequence backend. Runtime buffers, device fallback, tensor alignment and startup copies retain explicit planning limitations. See [artifact planning](docs/artifact-planning.md).

`--kv FP16` is the default. Q8_0/Q4_0 are separately disclosed capacity scenarios with alignment restrictions and unverified runtime compatibility. Memory and reserve flags accept GiB, decimal GB, or plain GiB numbers. Budget controls are `--gpu-runtime`, `--gpu-margin`, `--host-runtime`, `--os-reserve`, `--loading-workspace`, `--gpu-staging`; use `--help` for all options.

## Evidence

| Verification | Recorded result |
| --- | --- |
| MoonBit native tests, warnings denied | 100/100 passed |
| Existing manual-calculator CLI verification | 42/42 cases, 44 real processes |
| GGUF CLI verification | 29/29 cases, 56 real processes |
| Validation-tool guards and allocation/log checks, no GPU used | 25/25 passed |
| Real GGUF versus pinned official reader | 2 models, all 105 tensor descriptors matched |
| Actual sparse-file inspection | 4,294,967,460 logical bytes; 109 bytes read; no tensor payload read |
| CPU backend loading comparison | 4/4 runs; model and allocated KV predictions within the backend log rounding intervals |
| CUDA backend comparison, WSL2 RTX 5060 Ti | 7/7 cases; per-layer model/KV placement matched and allocation oracle fell within printed buffer intervals |

The sparse-file fixture allocated 4,096 disk bytes on the local filesystem. A separate virtual `ReaderAt` test covers offsets above `2^53`; this is not presented as a physical petabyte-file test. CPU peak process RSS is measured independently via `wait4` and is kept separate from backend buffer totals and OS planning reserves. Periodic `ps` sampling was unavailable in the local sandbox, so no steady-state RSS claim is made.

[Validation results and boundaries](docs/validation.md) · [Recorded reader comparison](examples/validation/gguf-comparison.json) · [CPU runtime report](examples/validation/cpu-runtime.json) · [Reproduce the backend comparison](docs/backend-validation.md)

[CUDA results and reproduction](docs/gpu-validation.md) describe the 17 October run on driver 581.57 with CUDA 13.0 (NVCC 13.0.48): output-only, partial and maximum offload, context 257 allocated as 512, and retention of CPU input embeddings alongside a tied GPU output copy. The [CUDA report and its 14 logs](examples/validation/gpu/gpu-runtime-run1.json) record exact inputs and checks. Model/KV logs have two decimal MiB precision; agreement is limited to their rounding intervals. Each case has two or three valid in-run device-memory samples; process VRAM peak remains unavailable. The `ngl=0` weight/KV baseline still logged a 17.69 MiB CUDA0 compute buffer, so it does not validate a zero-total-GPU budget.

These small-model measurements do not establish general GPU accuracy, large-model capacity, calibrated runtime reserves, acceptable generation speed, model quality, all-backend compatibility or hackathon acceptance. Qwen2 hardware loading and quantized KV compatibility remain unverified. No model weights or upstream inference engine are embedded in this repository.

The four CPU runs remain a separate historical evidence set with their recorded calculator binary and capture time. The CUDA report identifies the calculator and backend binaries used for its seven cases. Tokenizer fallback and context-window warnings are retained; output quality is not assessed.

## Manual estimate mode

If you do not have a GGUF file, the original five-input calculator remains available:

```sh
moon run src/cmd/llmcalc --target native -- \
  --params 27B --quant Q4_K_M --vram 16GiB --ram 64GiB --context 8K --json
```

This mode uses versioned BPW allowances and 7/14/32/72B structural proxies. It does not infer a model identity. Optional `--layers`, `--kv-heads`, `--head-dim` and `--query-heads` replace the proxy. Weight presets are FP16, BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS and IQ4_NL.

Manual mode retains its uniform-layer approximation; its layer count is not the artifact planner's output-first `n_gpu_layers`. Its `FULL_GPU` result is a policy scenario. Do not substitute manual estimates for artifact-specific evidence. See [manual methodology](docs/methodology.md), [memory policy](docs/memory-policy.md) and the historical [27B computed example](examples/27b-comparison.md).

## Build and verify

```sh
moon check --target native --deny-warn
moon test --target native --deny-warn
moon fmt --check
moon build --target native --release --deny-warn
python3 scripts/verify_cli.py _build/native/release/build/cmd/llmcalc/llmcalc.exe
python3 scripts/verify_gguf.py _build/native/release/build/cmd/llmcalc/llmcalc.exe
python3 scripts/verify_gpu_tools.py
```

The generated `.exe` name is also used on macOS and Linux. The calculator has no external Mooncakes dependencies; native file access requires a POSIX platform with 64-bit offsets. Basic integration tests use only Python's standard library. Optional official-reader verification additionally needs the pinned Python dependencies listed in `scripts/validation-requirements.txt`.

[Native CI](https://github.com/Nicksxs/llm-hardware-calculator/actions/workflows/ci.yml) runs compilation, warnings-as-errors tests, formatting, release build and both CLI verifiers on Linux. [CI setup](docs/ci.md) pins toolchain archives and hashes. JS, Wasm, web UI, split GGUF, MoE, MLA, sliding/hybrid attention, multi-GPU and unified-memory planning remain outside this delivery.

Package layout: `gguf` for pure binary parsing, `nativeio` for file access, `model` for checked tensor profiles and placement, `artifact_cli` for real-file commands, `core` and `cli` for manual estimation, and `cmd/llmcalc` for process entry.

## 中文简述

项目现已实现“读取真实 GGUF → 校验并统计张量 → 提取模型结构 → 按固定后端规则规划卸载”的完整路径，保留原五项输入估算入口。MoonBit 承担二进制解析、整数范围校验、结构提取、张量分组与规划；C 仅桥接随机文件读取，Python 用于独立验收与测量。两个真实模型的目录对照和四次 CPU 加载已有记录；WSL2 RTX 5060 Ti 上首轮七项 CUDA 后端实测通过，包括五项官方小模型和两项明确标注的衍生 tied fixture。结果限定于逐层放置和日志舍入区间内的模型/KV 分配对照，尚未校准运行时预留或验证大模型容量。

[早期设计记录](docs/project-proposal.md)仅供参考，当前功能与验证结果以本文及相关文档为准。

## License

[MIT](LICENSE). Referenced model artifacts and upstream tooling retain their own licenses; model weights are not redistributed here.
