# CUDA validation procedure

The GPU collector is prepared for hardware acceptance. **No CUDA inference results are recorded in this revision.** Local calculator checks and target WSL calculator tests do not establish that the CUDA backend or its memory predictions have passed hardware validation.

## Fixed scope

`scripts/validate_gpu_backend.py` targets one CUDA GPU, the pinned llama.cpp revision `88aaf325a7026ac1127d0897394b0a3d6413dc2a`, plain CPU/CUDA buffers, full attention, one sequence and FP16 KV. It uses fixed `--n-gpu-layers`, independently reads the GGUF directory and verifies every tensor placement against the calculator.

The five primary cases use the official `stories15M-q4_0.gguf` from the [model manifest](../examples/validation/model-manifest.json):

| Requested context | `n_gpu_layers` | Intended check |
| ---: | ---: | --- |
| 256 | 0 | CPU model/KV baseline in the CUDA-enabled backend |
| 256 | 1 | Output only on GPU; no GPU KV |
| 256 | 2 | Output and final repeating block on GPU |
| 256 | 7 | All six repeating blocks and output on GPU; input embedding remains CPU |
| 257 | 7 | Backend allocates 512 context cells |

An optional derived tied-output fixture adds context 256 with `n_gpu_layers=1` and `6` for its five repeating blocks. These cases check the additional GPU embedding copy. The collector checks the exact hashes of both inputs; an arbitrary five-layer model is not accepted as the derived fixture.

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

The prepared WSL host has an RTX 5060 Ti and Windows driver 581.57. Development tools are isolated in a user workspace. CUDA 13.0.0 redistributables provide `cuda_cccl`, `cuda_crt`, `cuda_cudart`, `cuda_nvcc`, `libnvvm` and `libcublas`; archive hashes come from the [official fixed manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.0.json). CMake 4.1.3 is used. [Toolchain hashes](../examples/validation/cuda-toolchain.json) and the [completed build's configuration and binary/library hashes](../examples/validation/cuda-build.json) are recorded. Compilation succeeded; CUDA inference acceptance remains pending.

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

- **Raw tensor payload** remains the calculator's exact file accounting. CUDA buffers include additional allocation: each tensor gets 128-byte alignment and quantized tensors get one tail-row padding allowance to a 512-element boundary. Plain CPU tensors use 32-byte alignment. The independent oracle applies these rules before comparing with printed buffer rounding intervals.
- **KV** follows actual repeating-block devices and the padded context. The output slot has no KV. FP16 K/V tensors are accounted separately; compute/output buffers are retained as observations rather than silently treated as weights.
- **Device telemetry** contains before/during/after `nvidia-smi` samples. Device totals include other processes, driver and CUDA context, and the sampled maximum is not the process's instantaneous VRAM peak.
- **Unavailable telemetry** stays `null` with a reason. WSL/WDDM does not provide every NVML metric; see [NVIDIA's documented limitations](https://docs.nvidia.com/cuda/archive/13.0.0/wsl-user-guide/index.html#features-not-yet-supported). A run with no valid in-run device-memory sample does not pass the full collector.
- **Provenance** includes binary/model hashes, exact command templates, platform and sanitized logs. GPU UUID is used internally to bind device selection and appears publicly only as a hash. Logs omit local user paths.

Passing these small-model checks would establish the recorded placement/allocation cases only. It would not calibrate default runtime reserves, GPU loading peaks, large-model capacity, Qwen2 execution, quantized KV, throughput or output quality.
