# Estimation methodology

The calculator estimates a memory budget under displayed assumptions. Its current scope is one discrete GPU, one sequence, dense decoder-only models, uniform full-attention layers, and MHA or GQA with the same key/value head dimension. It does not model MoE, MLA, sliding-window/hybrid attention, recurrent state, multiple GPUs or shared CPU/GPU memory.

The formulas and capacity rules are implemented and tested with the MoonBit native target. Tests check arithmetic, validation and policy decisions; no hardware measurements have calibrated the estimates. Memory fit does not establish successful execution, acceptable throughput or model quality. Source revisions are listed in [sources](sources.md).

## Units and validation

Parameter count `P` uses decimal billions: `P × 10^9` parameters. Capacities and output values use binary GiB: `1 GiB = 2^30 bytes`. A GB input means `10^9 bytes`; it is converted before comparing budgets. Context means allocated token capacity, not the number of tokens currently occupied; a CLI `K` suffix means 1,024 tokens.

Core inputs reject non-finite numbers, nonpositive parameter counts, negative capacities and contexts outside `1..1,048,576`. Zero VRAM remains valid for CPU-only evaluation. Advanced structures validate `layers=1..512`, `kv_heads=1..512`, `head_dim=1..1024`, and `kv_heads ≤ query_heads ≤ 512`, with query heads divisible by KV heads for complete MHA/GQA groups. These checks bound arithmetic; they do not certify a model architecture or its supported context window.

## Weight payload and planning policy

For a chosen BPW value `b`:

`weight bytes = P × 10^9 × b / 8`

The baseline column uses the dominant tensor type's block storage cost. The planning, low and high columns are separate **uncalibrated policy values**, not measured whole-model tensor inventories.

| Weight recipe | Block baseline BPW | Planning BPW | Low BPW | High BPW |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 16 | 16 | 16 | 16 |
| BF16 | 16 | 16 | 16 | 16 |
| Q8_0 | 8.5 | 8.6 | 8.5 | 8.8 |
| Q6_K | 6.5625 | 6.65 | 6.5625 | 6.9 |
| Q5_K_M | 5.5 | 5.75 | 5.5 | 6.1 |
| Q4_K_M | 4.5 | 4.85 | 4.5 | 5.2 |
| IQ4_XS | 4.25 | 4.6 | 4.25 | 5.1 |
| IQ4_NL | 4.5 | 4.7 | 4.5 | 5.2 |

The quantized baselines correspond to Q8_0 `34/32`, Q6_K `210/256`, Q5_K `176/256`, Q4_K `144/256`, IQ4_XS `136/256`, and IQ4_NL `18/32` bytes per parameter, including their block scales. Q4_K_M and Q5_K_M are mixed tensor recipes: their dominant Q4_K/Q5_K costs do not describe every tensor. IQ recipes can also select other tensor types. A real artifact can differ because of tensor mix, non-quantized tensors, alignment and metadata.

`planning-policy-v1` supplies a small disclosed allowance for whole-model tensor mix. Its low/high values describe alternative weight-cost scenarios, **not confidence intervals or guaranteed error bounds**. Weight estimates exclude GGUF metadata, runtime buffers and driver allocations. The current calculator accepts parameter count and does not inspect a GGUF tensor inventory.

## KV cache

The general full-attention payload relation is:

`KV bytes = T × Σ_layers [H_kv × (D_k × B_k + D_v × B_v)]`

The implemented uniform-layer, equal-dimension form is:

`KV bytes = T × L × H_kv × D × 2 × B`

`T` is token capacity, `L` layer count, `H_kv` KV-head count and `D` head dimension. The factor of two counts K and V, not two sequences. Arithmetic converts dimensions to Double before multiplication to avoid Int overflow.

| KV precision | Bytes per element, including scales |
| --- | ---: |
| FP16 | 2 |
| Q8_0 | `34/32 = 1.0625` |
| Q4_0 | `18/32 = 0.5625` |

Q8_0/Q4_0 require `head_dim` divisible by 32. The calculator rejects unsupported partial head blocks instead of silently rounding them. This simplified row-alignment rule does not reproduce every backend layout. Backend padding, graph buffers, fragmentation, cache streams and temporary copies are outside the raw payload formula.

KV precision is independent of weight precision; the default is FP16 KV. Quantized KV inputs are capacity scenarios. Support depends on the inference backend, device, attention kernel and model, and has not been verified by this project.

For `L=64`, `H_kv=8`, `D=128`, and `T=8,192`, raw KV is exactly `2 GiB` for FP16, `1.0625 GiB` for Q8_0 and `0.5625 GiB` for Q4_0. These are formula fixtures, not observed allocations.

## Structural proxies and sensitivity

Basic selects the smallest published size bucket whose upper bound covers the user's parameter count: `(0,7]`, `(7,14]`, `(14,32]`, `(32,72]` B. It displays the selected Qwen2.5 source and shape. This is an illustrative architecture policy; it does not identify a model and is not a conservative upper bound for every architecture. For parameter counts above 72B, supply an Advanced architecture rather than extrapolate.

Advanced replaces the proxy with explicit layers, KV heads and head dimension. Query heads can be supplied for the MHA sensitivity scenario. The calculator does not force `head_dim = hidden_size/query_heads` for arbitrary models.

Sensitivity reports compare the declared GQA shape with an MHA alternative using query heads as KV heads, and compare low/planning/high weight costs. They demonstrate how changed assumptions can change memory and placement conclusions. They are not statistical confidence intervals, a survey of all architectures, or calibrated probability statements.

## Separate GPU and host ledgers

At `n` GPU layers out of `L`, the equal-layer approximation assigns the fraction `n/L` of both weight and KV payloads to the GPU and the remainder to the host. Non-layer tensors, unequal layer sizes and actual runtime placement can violate this approximation. It is not a backend launch instruction.

Default `capacity-policy-v1` allowances are 0.5 GiB GPU runtime, 1 GiB GPU margin, 1 GiB host runtime, 4 GiB OS reserve, 1 GiB host loading workspace and 0.25 GiB additional GPU loading staging. All GPU-only allowances are zero when `n=0`. Capacity inputs describe budgets before these allowances; adjust the OS reserve if supplying RAM that is already free of OS use.

Steady-state GPU and host budgets add their resident weights, resident KV and applicable allowances. GPU loading adds staging to its steady budget. Host loading retains **one full weight copy**, replacing steady host-resident weights, then adds host KV, host runtime, OS reserve and loading workspace. It does not add an unexplained second full weight copy. This policy can differ from actual mmap/page residency and loading behavior; see the complete equations in [memory policy](memory-policy.md).

The fit search starts at all layers on the GPU and descends through integer splits to CPU-only. A candidate must fit **both steady-state and loading budgets**, independently in VRAM and RAM. The first fitting candidate is classified as `FULL_GPU`, `GPU+CPU OFFLOAD` or `CPU ONLY`. If none fits, `INSUFFICIENT` means no placement satisfies this policy; it does not prove that every backend configuration is unable to run.

## Verification boundary

Native tests include known block-cost and KV fixtures, invalid inputs, unit conversion, context growth, architecture buckets, aligned quantized KV, separate capacity limits and placement boundaries. They establish implementation consistency with the documented formulas and policies.

Future empirical validation needs a named model artifact and hash, backend revision, launch options, hardware and driver information, predicted component budgets, observed loading peaks and observed steady-state usage. Until that evidence exists, the project provides no measured accuracy percentage, fixed error tolerance, loading guarantee or throughput prediction. JS, Wasm and a web interface are not part of the current validated delivery.
