# Verification evidence

Recorded on 16 October 2026, macOS arm64, with MoonBit v0.10.14. The calculator and backend results are separate evidence sets.

## Calculator checks

| Check | Result |
| --- | --- |
| `moon check --target native --deny-warn` | Passed |
| Native unit and behavior tests, warnings denied | 95/95 passed |
| Native release build and formatting | Passed |
| Manual CLI verifier | 42/42 cases; 44 real processes |
| GGUF CLI verifier | 28/28 cases; 39 real processes |

Core and manual tests cover quantization/KV oracles, input units, non-finite values, placement boundaries, loading peaks and the model/context matrix. GGUF tests add checked UInt64 arithmetic, offsets above 2^53, v2/v3, block alignment, empty directories, typed alignment, short reads and incomplete unknown types. Model tests check real tensor roles/shapes, mutable Profile inconsistencies, output-first slots, CPU embeddings, tied output copies, nonuniform layers and context padding at 1/255/256/257 tokens.

The process-level GGUF verifier independently writes fixtures using Python `struct`; it checks every scalar/array metadata type, mixed layouts, malformed files, strict JSON, expected error exits, retention policies and both supported profile families. These synthetic Qwen2 fixtures establish code behavior, not a real Qwen2 loading measurement.

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

The OS independently recorded each child process's peak RSS using `wait4`. Periodic `ps` sampling was denied by the local sandbox; its fields remain unavailable, with the reason recorded. No sampled steady-state RSS is claimed. Process RSS includes executable code, allocator and compute buffers and is not equated to tensor payload or the calculator's host/OS budget reserves. Some allocated contexts exceed the tiny fixtures' training window; warnings are preserved, and generated-text quality is not assessed.

## Evidence boundary

- Actual loading evidence is **CPU-only, two Llama fixtures, FP16 KV**, one pinned backend and one local platform.
- Single-GPU placements, preferred-device kernel behavior, GPU startup peaks and Qwen2 hardware loading remain unmeasured.
- Raw payload agreement and rounded log agreement do not establish a universal memory-error percentage, successful loading of arbitrary models, throughput or model quality.
- Inspection does not read tensor payload contents or verify their hashes. The external validation tools hash model artifacts separately.
- JS, Wasm and a web frontend were not validated.

[Reproduction instructions](backend-validation.md) explain the optional reference-reader and backend checks. [Linux CI](https://github.com/Nicksxs/llm-hardware-calculator/actions/workflows/ci.yml) repeats self-contained unit and CLI checks; the recorded local backend runs are not presented as GPU or cross-platform runtime measurements.
