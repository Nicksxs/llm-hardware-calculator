# LLM Hardware Calculator

A proposed MoonBit tool for explaining the memory budget of local LLM inference.

**Status: proposal stage, 7 October 2026.** This repository currently contains project documentation. The MoonBit toolchain has not been installed in the development environment. The calculation engine and CLI have not been implemented, and no project tests or hardware measurements have been run. There is no runnable release yet.

The project is being prepared as a proposal for the October 2026 MoonBit Hackathon season. This statement does not indicate that registration or submission has been completed.

## The problem

Parameter count and a quantization label alone do not describe an inference memory budget. Weights, KV cache, runtime buffers, placement, and available host memory all matter. Longer contexts can change the result even when the model file stays the same size.

LLM Hardware Calculator will let a user start with five familiar inputs and inspect the assumptions behind the result. It will estimate memory requirements and explain feasible placement scenarios. It will not infer generation speed, model quality, or successful execution from a memory estimate.

## Planned MVP

- A standalone **MoonBit core** for validation, memory arithmetic, structural proxies, placement scenarios, and explanations.
- A thin **native CLI** that produces human-readable reports and structured **JSON**.
- **Automated tests** for units, formulas, invalid inputs, placement boundaries, context growth, and report consistency.
- **Transparent methodology**: formulas, preset sources, proxy assumptions, reserves, and sensitivity scenarios included in each result.

The initial scope is **one discrete GPU, dense decoder-only models, full attention, and MHA or GQA**, with one sequence in the baseline. Both partial GPU/CPU offload and CPU-only evaluation are part of the first MVP. MoE, MLA, sliding-window or hybrid attention, recurrent/state-space models, multimodal state, and multi-GPU placement are outside the initial scope.

## Basic and Advanced inputs

**Basic requires exactly these five inputs:**

| Input | Meaning |
| --- | --- |
| `params` | Model parameter count |
| `quant` | Weight precision or quantization recipe |
| `vram` | Available GPU memory budget |
| `ram` | Available system memory budget |
| `context` | Allocated token capacity; K = 1,024 |

Basic will use an explicit, documented structural proxy for layer count, KV heads, and head dimension, with **FP16 KV cache by default**. A proxy is a modeling assumption, not automatic identification of a real model's architecture. The report will display the proxy values, source and revision, selection rule, and alternative structural scenarios showing how KV memory and fit conclusions change. A concrete model profile is helpful but is not required to use Basic.

**Advanced optionally replaces the proxy** with `layers`, `kv_heads`, and `head_dim`, and selects `kv_precision` from **FP16, Q8_0, or Q4_0**. Quantized KV options describe memory scenarios; they do not guarantee that a particular inference backend, device, attention kernel, and model supports that combination. The default simplified structure uses a common K/V head dimension and uniform layers, and the report will state those assumptions.

The first MVP will include weight presets for **FP16, BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS, and IQ4_NL**. Weight precision and KV-cache precision are independent inputs.

## Placement and output

GPU and host memory will be budgeted separately. System RAM is not added to VRAM as a larger GPU budget.

| Planned result | Meaning under the displayed assumptions |
| --- | --- |
| Full GPU fit | GPU-resident weights, KV, and runtime reserves fit VRAM, and required host allocations fit RAM. |
| `GPU+CPU OFFLOAD` | An estimated layer split fits the GPU and host budgets independently. |
| `CPU ONLY` | A complete host-resident scenario fits RAM when the GPU scenarios do not. |
| `INSUFFICIENT` | No supported placement satisfies the applicable memory budgets. |

The MVP will search approximate layer splits rather than postpone offload support. It will use a disclosed **uniform-per-layer approximation** for splittable weight and KV allocation. This cannot reproduce every tensor placement, non-layer tensor, backend copy, or layer-specific size. The report will expose the estimated GPU layer count, CPU remainder, KV placement policy, and overhead assumptions instead of presenting the split as a backend launch prescription.

Each scenario will distinguish **steady-state residency** from **loading peak**. Staging buffers, simultaneous host/device copies, mapping assumptions, and initialization workspace can make startup require more memory than steady-state inference. The report will show separate GPU/RAM component budgets and loading assumptions; fitting the steady-state budget alone will not be presented as proof that the model can be loaded.

## Planned demonstration

The first comparison will use **27B parameters, 16 GiB GPU VRAM, and 64 GiB system RAM**, with **Q4_K_M, IQ4_XS, Q5_K_M, and Q8_0** at **8K, 16K, 32K, and 64K** context capacities. Basic will display its structural proxy and sensitivity scenarios. Advanced will demonstrate the effect of known structural values and different KV precisions.

These are planned scenarios, not measured results or a claim that every 27B model has the same architecture or fits. A named model's structure will be checked before it is presented as an exact profile. Outputs will include weights, KV, reserves, separate VRAM/RAM headroom, placement classification, and loading-peak assumptions.

## Method and evidence

The following describes the intended method, not an implemented API.

For weights, Basic will estimate bytes from parameter count and a versioned effective bits-per-weight preset, exposing the allowance for scales, mixed tensor types, and non-quantized tensors. Advanced or later artifact-specific profiles can use a tensor inventory or documented tensor-data size. All quantities will state their units; 1 GiB = 2^30 bytes.

GGML block layouts include scale metadata. Q4_K, IQ4_XS, Q5_K, Q6_K, Q8_0, and IQ4_NL have block storage costs of 4.5, 4.25, 5.5, 6.5625, 8.5, and 4.5 bits per weight respectively. These describe tensor block types, not whole-model recipes: [GGML block definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h).

Q4_K_M and Q5_K_M use tensor-dependent type selection; IQ4_XS can also contain other types. Their effective model storage depends on the tensor mix. Preset approximations and sensitivity ranges will disclose that limitation: [llama.cpp quantization selection](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp). GGUF metadata and alignment also mean file bytes and resident GPU bytes are different quantities: [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

For the full-attention, single-sequence baseline, the raw FP16 KV payload is:

`KV bytes = T × Σ_layers [H_kv × (D_k × B_k + D_v × B_v)]`

Here `T` is allocated token capacity; `H_kv` is each layer's KV-head count; `D_k` and `D_v` are key/value head dimensions; and `B_k = B_v = 2` for FP16. MHA uses the attention-head count for `H_kv`; GQA uses its KV-head count. Basic substitutes its displayed proxy; Advanced substitutes user values. This follows the dimensions in [llama.cpp hyperparameters](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-hparams.cpp) and the allocations in [llama.cpp KV cache](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp).

Q8_0 and Q4_0 KV scenarios will use block-size-aware storage costs, including scales and required rounding, rather than dividing FP16 bytes by an integer bit ratio. They remain hypothetical capacity scenarios unless the selected runtime combination is independently verified. Their block layouts are in the linked GGML definitions.

Raw payload calculations exclude allocation padding, graph/work buffers, backend-specific copies, driver reservations, fragmentation, and memory used by other applications. Explicit reserves and sensitivity scenarios will account for these uncertain components. Host file mapping, host-resident layer weights, staging copies, and loading workspace will be identified separately to avoid unexplained double counting.

There is currently **no validated accuracy figure or fixed error bound**. Structural-proxy scenarios are sensitivity analyses, not measured confidence intervals. Future validation will record the model artifact, backend version, launch settings, hardware, predicted components, and observed loading/steady-state allocations. Source links currently point to upstream branches; implementation will record exact revisions for presets and fixtures.

## Development plan

| Stage | Status and intended evidence |
| --- | --- |
| Scope and source research | Completed initial MoonBit and upstream quantization/KV documentation review; this is research only. |
| Toolchain and setup | Planned: install and record a MoonBit version, establish packages and repeatable checks. |
| Core and presets | Planned: Basic proxies, Advanced values, all listed precisions, and sensitivity reports. |
| Placement scenarios | Planned: full GPU, GPU+CPU offload, CPU only, insufficient budgets, and loading versus steady-state accounting. |
| CLI and JSON | Planned: native entry point and documented input/output after implementation. |
| Tests and validation | Planned: formula fixtures, edge cases, JSON checks, and reproducible backend comparisons when hardware is available. |
| Public release | Planned: reviewed limitations, examples, documentation, and a versioned release. |

A later **static web demo using the MoonBit JavaScript backend** may reuse the same core through a small JSON boundary. It is optional and is not required for the initial MVP.

## 中文简述

本项目拟用 MoonBit 实现本地大模型内存预算计算器。Basic 仅需参数量、权重量化、显存、内存和上下文五项，通过显示来源的结构代理与敏感性情景完成估算；Advanced 可指定层数、KV 头数、头维度及 KV 精度。首期同时评估 GPU 驻留、GPU+CPU offload、CPU only 和资源不足，分开报告 GPU/RAM 预算、稳态与加载峰值。计划以 27B、16 GiB VRAM、64 GiB RAM 为核心案例，交付独立 MoonBit core、Native CLI、JSON、测试及公开方法；Web 为后期可选。目前仅完成立项与调研，尚未安装工具链、实现 CLI 或运行测试。

See the [Chinese planning reference](docs/project-proposal.md) for an AI-assisted summary of the proposed scope. It is not the official application document. The registration form requires an applicant-written Markdown proposal of at most one page and at least 10 meaningful repository commits; registration has not been submitted.

## License

[MIT](LICENSE).
