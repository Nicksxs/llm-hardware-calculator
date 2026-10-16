# Reproduce real-file and CPU backend checks

The calculator's tests are self-contained. The optional checks below additionally use public model files and a fixed upstream reference. They are technical verification steps, not a competition application document.

## Inputs and provenance

- llama.cpp source revision: `88aaf325a7026ac1127d0897394b0a3d6413dc2a`.
- GGUF artifacts: [ggml-org/tiny-llamas](https://huggingface.co/ggml-org/tiny-llamas/tree/99dd1a73db5a37100bd4ae633f4cfce6560e1567), revision `99dd1a73db5a37100bd4ae633f4cfce6560e1567`.
- Names, exact file lengths, SHA-256 and official download URLs: [model manifest](../examples/validation/model-manifest.json).
- Observed backend build flags and source archive hash: [backend-build.json](../examples/validation/backend-build.json).

The model files are not redistributed in this repository. Obtain them from the manifest URLs. A transport mirror may be used when necessary, but the SHA-256 and length must still match the official fixed artifacts. The recorded run used a mirror because the original host was unreachable, and both hashes were checked against the official artifact pages.

Use a scratch directory outside the repository to keep downloaded source, weights, virtual environments and build output out of Git. The following shell variables are examples, not required installation locations:

```sh
LLMCALC_VALIDATION_DIR="$(pwd)/../llmcalc-validation"
mkdir -p "$LLMCALC_VALIDATION_DIR"
curl -fL 'https://github.com/ggml-org/llama.cpp/archive/88aaf325a7026ac1127d0897394b0a3d6413dc2a.tar.gz' \
  -o "$LLMCALC_VALIDATION_DIR/llama.tar.gz"
tar -xzf "$LLMCALC_VALIDATION_DIR/llama.tar.gz" -C "$LLMCALC_VALIDATION_DIR"
LLMCALC_REFERENCE_DIR="$LLMCALC_VALIDATION_DIR/llama.cpp-88aaf325a7026ac1127d0897394b0a3d6413dc2a"
```

Check the downloaded source archive against the recorded SHA-256 before using it: `20c307e54cd299aab1b5dacd4fdd5d8ccd18c2120281836b97fd761b181730ad`. The committed reference comparison also records individual Python reference-source hashes.

## Independent GGUF directory comparison

Use Python 3.12 and an isolated environment. NumPy/PyYAML are validation dependencies, not dependencies of the MoonBit calculator:

```sh
python3.12 -m venv "$LLMCALC_VALIDATION_DIR/venv"
"$LLMCALC_VALIDATION_DIR/venv/bin/python" -m pip install -r scripts/validation-requirements.txt
moon build --target native --release --deny-warn
"$LLMCALC_VALIDATION_DIR/venv/bin/python" scripts/compare_gguf.py \
  --calculator _build/native/release/build/cmd/llmcalc/llmcalc.exe \
  --gguf-python "$LLMCALC_REFERENCE_DIR/gguf-py" \
  --model "$LLMCALC_VALIDATION_DIR/stories260K.gguf" \
  --model "$LLMCALC_VALIDATION_DIR/stories15M-q4_0.gguf" \
  --output "$LLMCALC_VALIDATION_DIR/gguf-comparison.json"
```

Download the two named files into the scratch directory first. The comparator independently reads each directory with the official `GGUFReader` and compares every tensor's original GGML dimensions, type ID, name, element count, byte size and data offset. It also compares totals and selected architecture metadata. Differences are recorded by field and cause nonzero exit. It does not compare dequantized tensor values or generated text. See the committed [comparison result](../examples/validation/gguf-comparison.json).

## Fixed CPU backend build

The recorded build uses plain CPU buffers, without device offload, Accelerate, BLAS or runtime repacking. Build only `llama-completion`; the newer interactive `llama-cli` target is not needed:

```sh
cmake -S "$LLMCALC_REFERENCE_DIR" -B "$LLMCALC_VALIDATION_DIR/backend-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
  -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_NUMBER=0 \
  -DLLAMA_BUILD_COMMIT=88aaf325a7026ac1127d0897394b0a3d6413dc2a
cmake --build "$LLMCALC_VALIDATION_DIR/backend-build" -j 2 --target llama-completion
```

`--version` must identify the fixed revision. Platform/compiler differences can change binary hashes and process RSS; source, build flags and runtime settings must be recorded alongside observations. The committed build was AppleClang 15.0.0 on macOS arm64, with all GPU backends disabled.

## Actual loading and memory reconciliation

The runner requires macOS or Linux and only Python's standard library:

```sh
python3 scripts/validate_backend.py \
  --calculator _build/native/release/build/cmd/llmcalc/llmcalc.exe \
  --backend "$LLMCALC_VALIDATION_DIR/backend-build/bin/llama-completion" \
  --model "$LLMCALC_VALIDATION_DIR/stories260K.gguf" \
  --model "$LLMCALC_VALIDATION_DIR/stories15M-q4_0.gguf" \
  --context 128 --context 512 \
  --manifest examples/validation/model-manifest.json \
  --output "$LLMCALC_VALIDATION_DIR/cpu-runtime.json"
```

For each case it executes both calculator and backend. The backend uses `--device none`, `-ngl 0`, `--fit off`, `--no-repack`, `--load-mode none`, `--lazy-mode off`, Flash Attention off, FP16 K/V, two CPU threads and 32 generated tokens. Its logs and an exact command template are kept with the report. Saved logs replace local artifact paths with basenames and remove trailing line whitespace.

The runner distinguishes:

1. **Exact file tensor payload**, with metadata and alignment padding excluded.
2. **Requested and allocated KV payload**, with the pinned 256-token context padding included in the allocation.
3. **Backend-reported model/KV/compute/output buffers**, whose two-decimal MiB values have explicit rounding intervals.
4. **OS process high-water RSS**, measured per child through `wait4`, separate from optional periodic `ps` samples.

It compares only matching model and KV components against their reported rounding intervals. A nonzero backend exit or component mismatch fails the run. It does not treat logs as an independent GPU peak measurement or sum them into a claimed steady RSS value.

In the recorded sandbox `ps` was denied. OS `wait4` peak RSS remained available; unavailable sample fields and their reason were retained. No steady-state phase was identified. The result is **CPU loading evidence**, not a universal empirical accuracy bound. The tiny fixtures declare a short training context; allocation beyond it produces warnings and is not a text-quality evaluation.

GPU/offload hardware, other inference revisions, device-kernel fallbacks, Qwen2 runtime loading and quantized-KV runtime combinations need their own future evidence. The project does not infer those outcomes from these CPU checks.
