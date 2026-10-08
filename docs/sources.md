# Primary sources

Checked on 8 October 2026. These links pin the source revision used to check the calculator's formulas and presets. The calculator implements its own MoonBit arithmetic; it does not embed upstream inference or quantization code.

## Quantization and KV tensor layout

llama.cpp revision: [`06cad0b9e77315bd2930bd4f70a6d7b37b2a01c1`](https://github.com/ggml-org/llama.cpp/commit/06cad0b9e77315bd2930bd4f70a6d7b37b2a01c1).

| Source | What was checked |
| --- | --- |
| [GGML block definitions](https://github.com/ggml-org/llama.cpp/blob/06cad0b9e77315bd2930bd4f70a6d7b37b2a01c1/ggml/src/ggml-common.h) | Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, IQ4_XS and IQ4_NL block lengths and storage, including scales. |
| [llama.cpp quantization recipe selection](https://github.com/ggml-org/llama.cpp/blob/06cad0b9e77315bd2930bd4f70a6d7b37b2a01c1/src/llama-quant.cpp) | Recipes can select different types by tensor category, layer and architecture; Q4_K_M/Q5_K_M and IQ4 recipes are not a single block type for all tensors. |
| [llama.cpp hyperparameters](https://github.com/ggml-org/llama.cpp/blob/06cad0b9e77315bd2930bd4f70a6d7b37b2a01c1/src/llama-hparams.cpp) | Key/value embedding widths use the corresponding head dimension times KV heads, with possible differences across layers. |
| [llama.cpp KV cache allocation](https://github.com/ggml-org/llama.cpp/blob/06cad0b9e77315bd2930bd4f70a6d7b37b2a01c1/src/llama-kv-cache.cpp) | Per-layer K/V tensors use embedding width, cache capacity and stream count; layout, padding and backend allocation remain runtime dependent. |
| [llama.cpp license](https://github.com/ggml-org/llama.cpp/blob/06cad0b9e77315bd2930bd4f70a6d7b37b2a01c1/LICENSE) | MIT at the pinned revision. |

The [GGUF specification](https://github.com/ggml-org/ggml/blob/ffa4e8b80930029a35991f94e7c8a93cd67730ab/docs/gguf.md) is pinned separately to ggml revision `ffa4e8b80930029a35991f94e7c8a93cd67730ab`. Metadata and alignment are distinct from tensor payload. Its [repository license](https://github.com/ggml-org/ggml/blob/ffa4e8b80930029a35991f94e7c8a93cd67730ab/LICENSE) is MIT at that revision.

## Basic structural proxies

These are official Qwen model configuration files, not a claim that a user's model belongs to the Qwen family. `head_dim = hidden_size / query_heads` was checked for these four configurations only; the calculator does not impose that identity on arbitrary Advanced inputs.

| Basic parameter bucket, decimal B | Proxy and pinned configuration | Layers | KV heads | Query heads | Head dimension |
| --- | --- | ---: | ---: | ---: | ---: |
| `0 < P ≤ 7` | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/config.json) | 28 | 4 | 28 | 128 |
| `7 < P ≤ 14` | [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/config.json) | 48 | 8 | 40 | 128 |
| `14 < P ≤ 32` | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd/config.json) | 64 | 8 | 40 | 128 |
| `32 < P ≤ 72` | [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/495f39366efef23836d0cfae4fbe635880d2be31/config.json) | 80 | 8 | 64 | 128 |

The parameter buckets are calculator policy, rather than model-identification metadata. For example, 27B uses the 32B structural proxy. The proxy's size label is not substituted for the user's parameter count in weight arithmetic. Values above 72B require an explicit Advanced architecture. These configurations have full attention (`use_sliding_window: false`); the calculator does not infer or enforce their backend-specific supported context limits.

These links cite configuration facts. The project's MIT license does not relicense upstream model artifacts.

## Project policy and evidence

Upstream sources establish tensor shapes and block costs. They do not establish the calculator's `planning-policy-v1` BPW allowances, `capacity-policy-v1` reserves, equal-layer placement approximation or full-host-copy loading policy. Those are explicitly uncalibrated project choices documented in [methodology](methodology.md) and [memory policy](memory-policy.md).

Native MoonBit tests check arithmetic and policy behavior. No hardware measurements, empirical accuracy figure, performance benchmark or backend compatibility certification has been produced. Consult [toolchain](toolchain.md) for the verified environment; JS and Wasm execution are outside the current verification evidence.
