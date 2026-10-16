# Planning from a GGUF tensor inventory

The artifact path uses the real tensor directory and architecture metadata. Weight placement comes from stored tensor sizes rather than a parameter-count BPW preset or a uniform weight-per-layer fraction. KV and runtime budgets remain explicitly scoped calculations.

```sh
moon run src/cmd/llmcalc --target native -- plan /path/to/model.gguf \
  --vram 16GiB --ram 64GiB --context 8K --json

moon run src/cmd/llmcalc --target native -- plan /path/to/model.gguf \
  --vram 0GiB --ram 64GiB --context 4K --kv FP16 --json

moon run src/cmd/llmcalc --target native -- plan /path/to/model.gguf \
  --vram 16GiB --ram 64GiB --context 8K --n-gpu-layers 2 --json
```

Use an existing supported local model file. There are no `--params` or `--quant` overrides in this path. `inspect` remains available for other architectures or incomplete type inventories. The Basic five-input calculator remains a separate proxy-based path, documented in [methodology](methodology.md).

## Supported metadata and tensors

`general.architecture` must be `llama` or `qwen2`. The first adapter supports dense decoder-only full attention, scalar structure values shared across blocks, separate Q/K/V projections, equal K/V head dimensions and one sequence. It does not infer support from a model's display name.

The following keys use the selected architecture prefix, such as `qwen2.block_count`:

| Architecture metadata | Rule |
| --- | --- |
| `block_count` | Required scalar integer; 1–512. |
| `embedding_length` | Required scalar integer; 1–1,048,576. |
| `feed_forward_length` | Required scalar integer; 1–1,048,576. |
| `context_length` | Required positive scalar integer representable by `Int`; retained as model context metadata. |
| `attention.head_count` | Required scalar integer; at most 512 and positive. |
| `attention.head_count_kv` | Optional; defaults to query heads, matching the pinned loader's MHA default. Must be positive and divide query heads. |
| `attention.key_length`, `attention.value_length` | Optional; each defaults to embedding length / query heads when that division is exact. Final K/V dimensions must be equal and in 1–1,024. |
| `rope.dimension_count` | Optional; defaults to the final key dimension and must equal it for this adapter. |
| `attention.causal` | Optional; only Boolean true or absence is supported. |

Architecture structure arrays are rejected. Nonzero or complex expert, sliding-window, recurrent/SSM and next-token-predictor declarations are rejected. These are adapter scope checks, not a statement that those features are generally invalid GGUF. Other metadata, including tokenizer metadata, remains available through inspection. When `tokenizer.ggml.tokens` is present, it must be a string array whose declared count agrees with the embedding vocabulary dimension; full tokenizer semantics are not validated.

Let `E` be embedding length, `H` query heads, `K` KV heads, `D` the common head dimension, `F` feed-forward length and `V` the vocabulary dimension. Array dimensions below use GGUF order, with the contiguous dimension first.

| Tensor name | Required shape |
| --- | --- |
| `token_embd.weight` | `[E, V]`, required. |
| `output_norm.weight` | `[E]`, required. |
| `output.weight` | `[E, V]`, optional; absence selects tied output. |
| `output.bias` | `[V]`, optional for qwen2 only. |
| `blk.N.attn_norm.weight`, `blk.N.ffn_norm.weight` | `[E]`, required in every block. |
| `blk.N.attn_q.weight` | `[E, D × H]`, required. |
| `blk.N.attn_k.weight`, `blk.N.attn_v.weight` | `[E, D × K]`, required. |
| `blk.N.attn_output.weight` | `[D × H, E]`, required. |
| `blk.N.ffn_gate.weight`, `blk.N.ffn_up.weight` | `[E, F]`, required. |
| `blk.N.ffn_down.weight` | `[F, E]`, required. |

The qwen2 adapter additionally requires `D × H = E`; llama can use an explicitly declared `D` that differs from `E/H`. Block indices must be canonical decimal indices from zero through `block_count - 1`; every required tensor must be present. Q/K/V biases are optional for both architectures, with sizes `D × H`, `D × K` and `D × K`. Only llama accepts the optional `attn_output.bias[E]`, `ffn_gate.bias[F]`, `ffn_up.bias[F]` and `ffn_down.bias[E]`.

The allowlist deliberately excludes fused QKV, RoPE factor tensors, `.scale`/`.input_scale` extensions, expert tensors, embedded adapters and other unknown names. Some are supported by upstream llama.cpp, but they need additional allocation and alias handling before this planner can account for them. Extra names are errors rather than ignored bytes. Unknown byte sizes and storage types outside the planner subset are also errors. The accepted weight storage types are F32/F16/BF16, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0, Q2_K through Q6_K, IQ1_S/IQ1_M, IQ2_XXS/XS/S, IQ3_XXS/S and IQ4_NL/XS. Inspection can recognize additional storage layouts without admitting them for planning.

The shape and tensor checks follow the fixed [llama model loader](https://github.com/ggml-org/llama.cpp/blob/88aaf325a7026ac1127d0897394b0a3d6413dc2a/src/models/llama.cpp) and [qwen2 model loader](https://github.com/ggml-org/llama.cpp/blob/88aaf325a7026ac1127d0897394b0a3d6413dc2a/src/models/qwen2.cpp). Support here does not certify successful inference.

## Exact payload grouping and public profile checks

`extract_profile` aggregates `blk.N.*` into separate UInt64 layer totals. Input embeddings and stored output tensors remain separate groups. It retains each accepted tensor's original dimensions, type ID, element count, byte size, assigned role and block index. It recomputes storage size from the layout, validates dimensions against metadata and checks required tensor presence.

`tensor_payload_bytes` sums stored payload once, excluding file padding and metadata. `parameter_count` is the sum of stored tensor elements, rather than a certified trainable parameter count. `output_bytes` includes only actually stored output-group tensors; a missing tied output matrix is handled during placement instead of being added to file bytes.

Because the public `Profile` can be constructed or modified by library callers, `plan` re-derives the supported profile from retained raw tensor descriptors and verifies its structure, roles, block indices and summary totals. Changing a head dimension, moving an output tensor into a block, or changing a block assignment without consistent source descriptors is rejected. This is consistency validation, not a cryptographic binding to the original file or payload.

Payload and element totals must not exceed `2^53 - 1` before they enter the existing Double-based capacity calculation. Integer grouping and duplicate-copy accounting use checked UInt64 addition. This planner restriction does not reduce the GGUF parser's UInt64 file-offset domain. Raw artifact and placement integers remain decimal strings in JSON; the derived KV/runtime ledger uses Double values.

## Fixed single-GPU backend policy

Placement is pinned to llama.cpp `88aaf325a7026ac1127d0897394b0a3d6413dc2a`. Its [device-selection code](https://github.com/ggml-org/llama.cpp/blob/88aaf325a7026ac1127d0897394b0a3d6413dc2a/src/llama-model.cpp#L1656) treats the output as the first offload slot. This is revision-specific behavior.

For `L` repeating blocks and a requested `n_gpu_layers = n`:

| `n` | Intended GPU payload placement |
| --- | --- |
| 0 | No output or repeating block offload. |
| 1 | Output group only; no repeating blocks or KV on GPU. |
| 2 through L | Output group plus the final `n - 1` blocks. |
| L + 1 | Output group and every repeating block. |

The first GPU block is `L + 1 - n`; the output is a separate slot at index `L`. The input embedding remains on CPU. `output_norm.weight` and qwen2's optional output bias follow the output group. The library's `Plan.ledger.gpu_layers` counts only repeating blocks; `Plan.n_gpu_layers` includes the output slot.

The artifact CLI reports `MAX_OFFLOAD` for all blocks plus output, since CPU embeddings still remain. `GPU+CPU OFFLOAD` includes output-only offload. `CPU ONLY` and `INSUFFICIENT` retain their capacity-policy meanings.

The optional `--n-gpu-layers INTEGER` flag fixes this output/block slot count. It accepts nonnegative ASCII decimal integers, including zero, in either `--n-gpu-layers 2` or `--n-gpu-layers=2` form. Without the flag, the CLI searches for the largest fitting placement. With the flag, it evaluates exactly the requested `n`; a budget failure returns `INSUFFICIENT` and `fit.plan = null` without searching for another placement. JSON `input.n_gpu_layers` contains the requested integer even when it does not fit, or null when the flag was omitted. Text output identifies fixed selection versus search.

Malformed, negative, overflowing, duplicate or missing values are CLI errors with exit code 2, and `inspect` rejects the flag. A decimal integer outside the file-specific `0..L+1` range is a model planning error with exit code 1. A valid fixed request that cannot satisfy the budgets remains a successful report with exit code 0 and `INSUFFICIENT` status.

When `output.weight` is absent, the pinned backend creates its output from `token_embd.weight`. The [loader reclassifies this duplicate as an output operation](https://github.com/ggml-org/llama.cpp/blob/88aaf325a7026ac1127d0897394b0a3d6413dc2a/src/llama-model-loader.cpp#L1163), then [reuses it only within the same buffer context](https://github.com/ggml-org/llama.cpp/blob/88aaf325a7026ac1127d0897394b0a3d6413dc2a/src/llama-model-loader.cpp#L1371). Under this planner's standard CPU-buffer policy, CPU-only tied output reuses the embedding. GPU output requires a second embedding-sized GPU payload while preserving the CPU input copy:

```text
CPU weight bytes + GPU weight bytes = stored payload bytes + duplicated bytes
```

This policy assumes one discrete GPU, plain CPU buffers, no ACCEL device selection, no lazy weight loading, no repacking, no host-buffer or tensor-buffer overrides, and supported kernels on the intended GPU. The upstream [CPU buffer candidate list](https://github.com/ggml-org/llama.cpp/blob/88aaf325a7026ac1127d0897394b0a3d6413dc2a/src/llama-model.cpp#L1126) can select ACCEL, host or extra buffers before ordinary CPU buffers. `--no-repack --no-host` alone does not prove that every assumption holds. Kernel fallback and actual buffer selection must be checked against backend logs; the planner does not simulate all such choices.

These are tensor payload assignments. Backend buffer alignment, work graphs, driver memory and transient copies remain outside exact payload accounting. No GPU runtime validation is claimed by this policy description.

## KV, budgets and search

The adapter reuses the core's raw single-sequence KV formula and FP16/Q8_0/Q4_0 scale-inclusive storage costs. `Plan.kv_bytes` preserves raw payload for the requested context. The pinned backend's [context initialization](https://github.com/ggml-org/llama.cpp/blob/88aaf325a7026ac1127d0897394b0a3d6413dc2a/src/llama-context.cpp#L377) rounds context up to a multiple of 256 tokens. For the supported single-sequence case, `n_ctx_seq` has the same allocated capacity:

```text
allocated_context = ceil(requested_context / 256) × 256
allocated_kv_bytes = core KV formula evaluated at allocated_context
```

The ledger splits `allocated_kv_bytes` proportionally across the model's `L` repeating blocks: GPU KV is `allocated_kv_bytes × gpu_block_count / L`, with the remainder on the CPU. Output offload contributes no KV. It assumes KV follows its block's device. Requested context is limited to 1–1,048,576 tokens; the upper limit is already divisible by 256. A warning compares the allocated capacity, including padding, with the model's context metadata. It does not establish that context extrapolation works. Quantized KV requires head dimension divisible by 32 and still needs separately verified backend/Flash Attention compatibility. Tensor/buffer alignment and other backend allocations remain outside this token-padding calculation.

The search visits `n = L + 1` down to zero and accepts the first placement whose GPU and RAM steady and modeled loading requirements fit independently. Output-only offload still incurs GPU runtime, margin and staging allowances even though it has zero GPU blocks. CPU-only planning incurs none of those GPU allowances.

Library callers can choose between three entry points:

| API | Behavior |
| --- | --- |
| `model.plan(profile, context, precision, n_gpu_layers, policy)` | Validates the profile and constructs one payload placement and ledger without accepting capacity budgets. |
| `model.evaluate_plan(profile, context, precision, n_gpu_layers, vram_gib, ram_gib, policy)` | Evaluates exactly one requested placement against both GPU/RAM steady and loading budgets; returns `PlanSearch` with no plan when insufficient. |
| `model.search_plan(profile, context, precision, vram_gib, ram_gib, policy)` | Searches descending output/block slot counts and returns the largest fitting placement. |

The fixed evaluator and search use the same budget predicate. `evaluate_plan` does not clamp the requested count or call the search after a failure; its reason identifies the fixed request. Both capacity inputs and all existing profile/context/policy checks still apply.

The [capacity policy](memory-policy.md) supplies runtime, margin, OS and workspace allowances. Host loading retains one full stored-payload source copy plus CPU KV, host runtime, OS reserve and workspace. It replaces steady host weights rather than adding an unexplained second source copy. Tied output's additional GPU allocation is accounted on the GPU side. Mmap residency and actual loading peaks can differ; the full-host-copy policy is an explicit conservative scenario, not an observed universal loader requirement.

## Nonuniform example and evidence boundary

The model tests include a synthetic two-block directory with 304-byte and 608-byte blocks, 128 bytes of CPU input embeddings and 144 bytes of output tensors. At 16 requested tokens, raw FP16 KV is 256 bytes, while the backend's 256 allocated tokens require 4,096 bytes. With output plus one block offloaded:

| Budget component | Uniform-block approximation | Actual descriptor grouping |
| --- | ---: | ---: |
| Output payload | 144 B | 144 B |
| Final block payload | 456 B | 608 B |
| One block of allocated KV | 2,048 B | 2,048 B |
| GPU sum before allowances | 2,648 B | 2,800 B |

With sufficient host memory, a 2,700-byte GPU budget admits the uniform projection but rejects the actual final block. The tensor-based search selects output-only offload. This is an arithmetic/placement regression fixture, not a measured language model or hardware benchmark.

The `src/model` tests cover this distinction, output-first slot boundaries, the maximum-offload CPU embedding, tied-output duplication, independent host loading requirements, one-byte budget boundaries, 1/255/256/257-token padding boundaries, quantized KV and inconsistent public profiles. Fixed-request tests additionally verify exact `n=0/1/2/L+1` results under ample budgets and rejection of a requested placement whose GPU steady, GPU loading or host loading budget is too small, without fallback. CLI process tests retain the request in JSON and distinguish usage errors from model bounds. Measured backend evidence, hardware details and any observed errors are maintained separately in [validation](validation.md). Parser success, exact tensor bytes and a fitting plan do not establish GPU loading success, calibrated total-memory accuracy, throughput or model quality. The newly added fixed-request path has not been exercised on GPU hardware.
