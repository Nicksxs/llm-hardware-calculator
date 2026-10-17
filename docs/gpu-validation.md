# CUDA results and reproduction

The first hardware run on **17 October 2026 passed all 7/7 recorded cases** on WSL2 x86_64 with an NVIDIA GeForce RTX 5060 Ti, Windows driver **581.57** and CUDA **13.0** (NVCC **13.0.48**). The [runtime report](../examples/validation/gpu/gpu-runtime-run1.json) and its 14 stdout/stderr logs preserve inputs, binary/model hashes, per-layer device checks, allocation comparisons and telemetry. This validates the listed small-fixture cases within the stated measurement limits.

## Fixed scope

`scripts/validate_gpu_backend.py` targets one CUDA GPU, the pinned llama.cpp revision `88aaf325a7026ac1127d0897394b0a3d6413dc2a`, plain CPU/CUDA buffers, full attention, one sequence and FP16 KV. It uses fixed `--n-gpu-layers`, independently reads the GGUF directory and verifies every tensor placement against the calculator.

The run used five cases from the official `stories15M-q4_0.gguf` in the [model manifest](../examples/validation/model-manifest.json), and two from the explicitly derived tied-output fixture. All exited successfully. Actual per-layer model/KV devices matched the independent oracle and calculator; model/KV byte expectations fell within the printed buffer intervals:

| Fixture | Requested → allocated context | `n_gpu_layers` | GPU blocks | CUDA0 model / KV logs, MiB | Valid in-run samples |
| --- | ---: | ---: | ---: | --- | ---: |
| Official stories15M-q4_0 | 256 → 256 | 0 | 0 | absent / absent | 2 |
| Official stories15M-q4_0 | 256 → 256 | 1 | 0 | 9.34 / absent | 2 |
| Official stories15M-q4_0 | 256 → 256 | 2 | 1 | 9.88 / 0.28 | 3 |
| Official stories15M-q4_0 | 256 → 256 | 7 | 6 | 12.56 / 1.69 | 2 |
| Official stories15M-q4_0 | 257 → 512 | 7 | 6 | 12.56 / 3.38 | 3 |
| Derived stories260K-tied | 256 → 256 | 1 | 0 | 0.13 / absent | 2 |
| Derived stories260K-tied | 256 → 256 | 6 | 5 | 0.99 / 0.16 | 3 |

An absent CUDA0 model or KV line is expected when that component has no CUDA allocation; it does not mean the process uses no GPU memory. In particular, `ngl=0` is a CPU weight/KV placement baseline, but its CUDA-enabled execution still logged **17.69 MiB CUDA0 compute**. It does not validate the calculator's CPU-only zero GPU allowance as a total-runtime budget.

The context boundary case changed allocated cells from 256 to 512 and the GPU FP16 KV oracle from **1,769,472 to 3,538,944 bytes**, matching the **1.69 to 3.38 MiB** log intervals. The tied cases retained CPU input embeddings and added a **131,072-byte** GPU output copy in the oracle; model-buffer logs were consistent with the resulting totals. The collector checks exact hashes of both inputs; an arbitrary five-layer model is not accepted as the derived fixture.

Each case evaluated five prompt tokens and reported 31 evaluation runs. Logs retain four warnings per case: two tokenizer-default fallbacks and two warnings that the allocated 256/512 context exceeds the fixture's 128-token training context. No generated-text quality or context-extrapolation quality was assessed.

## Derive the tied fixture

Use the pinned official GGUF Python source and Python validation dependencies from [the CPU/reference setup](backend-validation.md). Choose new output and manifest paths; the generator refuses existing destinations:

```sh
"$LLMCALC_VALIDATION_DIR/venv/bin/python" scripts/make_tied_fixture.py \
  --source "$LLMCALC_VALIDATION_DIR/stories260K.gguf" \
  --gguf-python "$LLMCALC_REFERENCE_DIR/gguf-py" \
  --output "$LLMCALC_VALIDATION_DIR/stories260K-tied.gguf" \
  --manifest "$LLMCALC_VALIDATION_DIR/tied-fixture.json"
```

The script pins both source-model and official writer hashes. It removes only `output.weight`, labels the model as a derived structural fixture, and verifies that all 47 retained tensor descriptors and payloads are unchanged. The derived SHA-256 is `67af9dce730ec642b9648dfa18488597b049811292cbff7317f9fe8d6090f196`; file length is 1,054,464 bytes and retained tensor payload is 1,040,128 bytes.

The fixture is not a trained tied-weight model. Loading it checks structure and allocation; generation quality is not assessed. No model weights are redistributed in this repository.

## Build the reference CUDA backend

The measured WSL2 host has an RTX 5060 Ti and Windows driver 581.57. Development tools are isolated in a user workspace. The CUDA 13.0.0 distribution manifest supplies NVCC/runtime **13.0.48**, alongside the recorded `cuda_cccl`, `cuda_crt`, `cuda_cudart`, `cuda_nvcc`, `libnvvm` and `libcublas` packages; archive hashes come from the [official fixed manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.0.json). CMake 4.1.3 is used. [Toolchain hashes](../examples/validation/cuda-toolchain.json) and the [completed build's configuration and binary/library hashes](../examples/validation/cuda-build.json) are recorded. Their preparation-time GPU-run fields describe the build stage; the later runtime report records the seven actual inference cases.

For a manually merged SDK, the NVCC profile expects `lib64`; the prepared SDK uses `lib64 -> lib`. Keep the original WSL driver bridge at `/usr/lib/wsl/lib`. NVIDIA requires using the Windows-provided driver in WSL; installing a Linux NVIDIA driver would overwrite that bridge. See [NVIDIA's WSL guide](https://docs.nvidia.com/cuda/archive/13.0.0/wsl-user-guide/index.html).

Set `LLMCALC_CUDA` to the SDK prefix, and use the same fixed source downloaded in the reference setup:

```sh
export PATH="$LLMCALC_CUDA/bin:$PATH"
export LD_LIBRARY_PATH="$LLMCALC_CUDA/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S "$LLMCALC_REFERENCE_DIR" -B "$LLMCALC_VALIDATION_DIR/backend-cuda" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_STATIC=OFF \
  -DGGML_CUDA_GRAPHS=OFF -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DCUDAToolkit_ROOT="$LLMCALC_CUDA" \
  -DCMAKE_CUDA_COMPILER="$LLMCALC_CUDA/bin/nvcc" \
  -DCUDA_cuda_driver_LIBRARY=/usr/lib/wsl/lib/libcuda.so \
  -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF -DGGML_ACCELERATE=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_NUMBER=0 \
  -DLLAMA_BUILD_COMMIT=88aaf325a7026ac1127d0897394b0a3d6413dc2a
cmake --build "$LLMCALC_VALIDATION_DIR/backend-cuda" -j 2 --target llama-completion
```

`120a-real` is specific to this Blackwell target. Other GPUs need a suitable explicitly recorded architecture. Stub-library directories are for linking and must not be placed in the runtime `LD_LIBRARY_PATH`.

## Run only when GPU resources are available

The collector runs inference. It refuses unknown free memory or less than 2 GiB free, and checks again immediately before each launch. The 2 GiB threshold is a guard for these tiny fixtures, not a general capacity guarantee. Use a fresh report basename in a results directory, because existing report/log destinations are rejected to protect inputs and prior evidence:

```sh
moon build --target native --release --deny-warn
python3 scripts/validate_gpu_backend.py \
  --calculator _build/native/release/build/cmd/llmcalc/llmcalc.exe \
  --backend "$LLMCALC_VALIDATION_DIR/backend-cuda/bin/llama-completion" \
  --model "$LLMCALC_VALIDATION_DIR/stories15M-q4_0.gguf" \
  --tied-model "$LLMCALC_VALIDATION_DIR/stories260K-tied.gguf" \
  --nvidia-smi /usr/lib/wsl/lib/nvidia-smi \
  --output "$LLMCALC_VALIDATION_DIR/results/gpu-runtime.json"
```

Inference disables automatic fit, repacking, host-buffer preference, mmap, lazy loading and Flash Attention; log verbosity 5 exposes actual per-layer device assignments. The collector rejects preferred-buffer fallback, absent/duplicate required logs and mismatched model/KV allocations. The pinned backend's INFO repeating-layer count has an off-by-one at maximum offload, so acceptance uses DEBUG assignments rather than that summary.

## Interpret the results

- **Raw tensor payload** remains the calculator's exact file accounting. The independent allocation oracle applies 128-byte CUDA tensor alignment and one quantized tail-row padding allowance to a 512-element boundary; plain CPU tensors use 32-byte alignment. The log comparison establishes consistency with two-decimal MiB rounding intervals, not byte-exact physical measurement of those rules. Some raw and padded totals fit the same interval in this run, so small padding differences are not independently resolved by the logs.
- **KV** follows actual repeating-block devices and the padded context. The output slot has no KV. FP16 K/V tensors are accounted separately; compute/output buffers are retained as observations rather than silently treated as weights.
- **Device telemetry** contains before/during/after `nvidia-smi` samples. Every case had two or three valid in-run samples, with the backend alive at both ends of each query. Device totals include other processes, driver and CUDA context; their sampled maximum is not the process's instantaneous VRAM peak. `process_vram_peak_bytes` remains null in all seven cases.
- **Unavailable telemetry** stays `null` with a reason. WSL/WDDM does not provide every NVML metric; see [NVIDIA's documented limitations](https://docs.nvidia.com/cuda/archive/13.0.0/wsl-user-guide/index.html#features-not-yet-supported). A run with no valid in-run device-memory sample does not pass the full collector.
- **Provenance** includes binary/model hashes, exact command templates, platform and sanitized logs. GPU UUID is used internally to bind device selection and appears publicly only as a hash. Logs omit local user paths. The [post-run provenance record](../examples/validation/gpu/gpu-provenance.json) verifies the collector hash and all eight prepared backend executable/library artifacts after the run.

The seven passing cases establish the recorded per-layer model/KV placement and rounded component-buffer comparisons on this GPU. They do not calibrate default runtime reserves, validate a zero-total-GPU budget at `ngl=0`, measure process/GPU loading peaks, or establish large-model capacity, Qwen2 execution, quantized KV compatibility, throughput or output quality.
