# Verification evidence

Calculator checks recorded on 16–17 October 2026 on macOS arm64 and WSL2 x86_64, with MoonBit v0.10.14. The CPU and CUDA runtime reports preserve their respective calculator/backend hashes and capture times. Calculator checks and backend measurements remain separate evidence sets.

## Calculator checks

| Check | Result |
| --- | --- |
| `moon check --target native --deny-warn` | Passed |
| Native unit and behavior tests, warnings denied | 100/100 passed |
| Native release build and formatting | Passed |
| Manual CLI verifier | 42/42 cases; 44 real processes |
| GGUF CLI verifier | 29/29 cases; 56 real processes |
| Validation-tool guards and allocation/log checks, no GPU used | 25/25 passed |

Core and manual tests cover quantization/KV oracles, input units, non-finite values, placement boundaries, loading peaks and the model/context matrix. GGUF tests add checked UInt64 arithmetic, offsets above 2^53, v2/v3, block alignment, empty directories, typed alignment, short reads and incomplete unknown types. Model tests check real tensor roles/shapes, mutable Profile inconsistencies, output-first slots, CPU embeddings, tied output copies, nonuniform layers and context padding at 1/255/256/257 tokens.

The process-level GGUF verifier independently writes fixtures using Python `struct`; it checks every scalar/array metadata type, mixed layouts, malformed files, strict JSON, expected error exits, retention policies and both supported profile families. These synthetic Qwen2 fixtures establish code behavior, not a real Qwen2 loading measurement.

The fixed-offload addition tests `--n-gpu-layers` at `0`, `1`, `2` and `L+1` under ample budgets and verifies that the output equals the requested count. Unit tests cover GPU steady, GPU loading and host loading failures at exact byte boundaries. The CLI verifier confirms that a valid request that does not fit returns `INSUFFICIENT` with `fit.plan = null`, preserves `input.n_gpu_layers`, and does not fall back to CPU-only. It also checks malformed, missing, duplicate, negative and overflowing values, rejection by `inspect` (exit 2), and a file-specific count above `L+1` (exit 1). Omitted requests retain the descending search and JSON null input. These are self-contained behavior checks, not GPU measurements.

## Physical large file

The native verifier created a real sparse file, not only a mock:

- Logical file length: **4,294,967,460 bytes**.
- Local allocated disk space: **4,096 bytes**, eight `st_blocks` units.
- Native adapter bytes read: **109**.
- Maximum read end: **109**, before data offset **128**.

The descriptor points beyond 4 GiB, and both its offset and file size survive JSON as exact decimal strings. The temporary file was deleted after validation. Filesystem allocation counts are environment-specific; the verifier prints them rather than assuming an identical count everywhere. Separate virtual-reader tests cover >2^53 offsets and UInt64 overflow. No physical petabyte-scale file was created.

## Independent real-file comparison

The two models in [the fixed source manifest](../examples/validation/model-manifest.json) were compared with the official GGUFReader at llama.cpp `88aaf325a7026ac1127d0897394b0a3d6413dc2a`:

| Model | File bytes | Tensors | Exact tensor payload bytes |
| --- | ---: | ---: | ---: |
| stories260K.gguf | 1,185,376 | 48 | 1,171,200 |
| stories15M-q4_0.gguf | 19,077,344 | 57 | 18,350,208 |

All **105** descriptors matched name, original GGML dimensions, type ID, elements, payload bytes and relative/absolute offset. Selected structural metadata and totals matched. A deliberately altered shape caused the comparator to fail with a field-specific difference, checking that it can detect disagreement. [The report](../examples/validation/gguf-comparison.json) records input and reference-source hashes and dependency versions, without local file paths. It is a comparison of directories, not tensor-value dequantization or inference output.

## Actual CPU loading

A CPU-only `llama-completion` was built from the same pinned source. Metal, CUDA, Vulkan, BLAS and Accelerate were disabled; runtime flags disable repack, lazy reading, automatic fit and mmap. The exact recorded flags and source archive hash are in [backend build](../examples/validation/backend-build.json).

Both models were actually loaded and evaluated at requested contexts **128 and 512**. The backend allocates **256 and 512** cells respectively. All four runs exited successfully. Exact predicted model payload and allocated KV fell within the rounding intervals of the backend's two-decimal MiB logs. [The runtime report](../examples/validation/cpu-runtime.json) retains the component comparisons and stderr logs.

Those four runs remain the CPU evidence for the calculator binary and capture time recorded in that report. The later CUDA measurements below are a separate evidence set.

The OS independently recorded each child process's peak RSS using `wait4`. Periodic `ps` sampling was denied by the local sandbox; its fields remain unavailable, with the reason recorded. No sampled steady-state RSS is claimed. Process RSS includes executable code, allocator and compute buffers and is not equated to tensor payload or the calculator's host/OS budget reserves. Some allocated contexts exceed the tiny fixtures' training window; warnings are preserved, and generated-text quality is not assessed.

## Actual CUDA run

On **17 October 2026**, the first **7/7** cases passed on WSL2 x86_64, an NVIDIA GeForce RTX 5060 Ti, Windows driver **581.57** and CUDA **13.0** (NVCC **13.0.48**), using the pinned llama.cpp revision `88aaf325a7026ac1127d0897394b0a3d6413dc2a`. The [report](../examples/validation/gpu/gpu-runtime-run1.json) references 14 retained stdout/stderr logs; [build provenance](../examples/validation/cuda-build.json) records the backend binary, libraries and configuration.

Five cases use the official `stories15M-q4_0.gguf`: context 256 at `ngl=0/1/2/7`, and context 257 at `ngl=7`. Two cases use the explicitly derived `stories260K-tied.gguf` at context 256 and `ngl=1/6`; it is an allocation fixture, not a trained tied-weight model. Actual per-layer CPU/CUDA0 model and KV assignments matched the independent inventory and fixed-request predictions. The full-offload 15M KV log increased from **1.69 MiB** at context 256 to **3.38 MiB** at requested context 257, which allocated 512 cells; the latter oracle is **3,538,944 bytes**. The tied cases retained CPU input embeddings and allocated an additional GPU output copy.

Model/KV allocation comparisons passed within the logs' two-decimal MiB rounding intervals. This does not resolve padding byte-for-byte: some raw and padded totals fit the same printed interval. Each case evaluated five prompt tokens and reported 31 evaluation runs. Tokenizer-default and training-context warnings remain in the logs; generation quality was not assessed.

Each case has two or three valid device-memory samples taken while the backend was alive. These are total-device `nvidia-smi` observations, including other processes and CUDA/driver state; `process_vram_peak_bytes` remains null. The `ngl=0` case placed model weights and KV on CPU but still logged a **17.69 MiB CUDA0 compute buffer**. It establishes a placement baseline in a CUDA-enabled backend, not fully CPU execution or a zero-total-GPU budget. [The CUDA results and procedure](gpu-validation.md) give the complete case table and reproduction steps.

## Evidence boundary

- Recorded runtime evidence covers four CPU runs on two official Llama fixtures and seven CUDA cases on one official Llama fixture plus one explicitly derived tied-output fixture, all with FP16 KV and one pinned backend.
- The CUDA run verifies its listed per-layer model/KV assignments and rounded component-buffer comparisons. Process VRAM peaks, GPU startup peaks and default runtime-reserve calibration remain unverified.
- Large-model capacity, Qwen2 hardware loading and quantized KV compatibility remain unverified.
- Raw payload agreement and rounded log agreement do not establish a universal memory-error percentage, successful loading of arbitrary models, throughput or model quality.
- Inspection does not read tensor payload contents or verify their hashes. The external validation tools hash model artifacts separately.
- JS, Wasm and a web frontend were not validated.

[Reproduction instructions](backend-validation.md) explain the optional reference-reader and CPU checks. [Linux CI](https://github.com/Nicksxs/llm-hardware-calculator/actions/workflows/ci.yml) repeats self-contained unit and CLI checks; the separately recorded CPU and CUDA runs do not establish compatibility across arbitrary platforms or GPUs.
