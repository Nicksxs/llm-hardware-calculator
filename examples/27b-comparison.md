# Computed 27B capacity comparison

Generated with the native CLI on 8 October 2026. These are formula/policy outputs, not hardware measurements. Inputs are 27 decimal billion parameters, 16 GiB GPU memory and 64 GiB host memory. The Basic Qwen2.5-32B proxy has 64 layers, 8 KV heads, 128 head dimension and 40 query heads. KV is FP16; reserves and loading copies use capacity-policy-v1.

| Quant | Context | Planning weights GiB | KV GiB | Result | GPU layers / 64 | GPU loading GiB | RAM loading GiB |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| Q4_K_M | 8K | 15.245 | 2.000 | GPU+CPU OFFLOAD | 52 | 15.761 | 21.620 |
| Q4_K_M | 16K | 15.245 | 4.000 | GPU+CPU OFFLOAD | 47 | 15.883 | 22.307 |
| Q4_K_M | 32K | 15.245 | 8.000 | GPU+CPU OFFLOAD | 39 | 15.915 | 24.370 |
| Q4_K_M | 64K | 15.245 | 16.000 | GPU+CPU OFFLOAD | 29 | 15.908 | 29.995 |
| IQ4_XS | 8K | 14.459 | 2.000 | GPU+CPU OFFLOAD | 55 | 15.894 | 20.740 |
| IQ4_XS | 16K | 14.459 | 4.000 | GPU+CPU OFFLOAD | 49 | 15.883 | 21.396 |
| IQ4_XS | 32K | 14.459 | 8.000 | GPU+CPU OFFLOAD | 40 | 15.787 | 23.459 |
| IQ4_XS | 64K | 14.459 | 16.000 | GPU+CPU OFFLOAD | 29 | 15.552 | 29.209 |
| Q5_K_M | 8K | 18.073 | 2.000 | GPU+CPU OFFLOAD | 45 | 15.864 | 24.667 |
| Q5_K_M | 16K | 18.073 | 4.000 | GPU+CPU OFFLOAD | 41 | 15.891 | 25.511 |
| Q5_K_M | 32K | 18.073 | 8.000 | GPU+CPU OFFLOAD | 34 | 15.602 | 27.823 |
| Q5_K_M | 64K | 18.073 | 16.000 | GPU+CPU OFFLOAD | 26 | 15.592 | 33.573 |
| Q8_0 | 8K | 27.032 | 2.000 | GPU+CPU OFFLOAD | 31 | 15.812 | 34.063 |
| Q8_0 | 16K | 27.032 | 4.000 | GPU+CPU OFFLOAD | 29 | 15.811 | 35.219 |
| Q8_0 | 32K | 27.032 | 8.000 | GPU+CPU OFFLOAD | 26 | 15.982 | 37.782 |
| Q8_0 | 64K | 27.032 | 16.000 | GPU+CPU OFFLOAD | 21 | 15.870 | 43.782 |

The GPU layer split is a uniform-layer capacity approximation. It is not a backend launch instruction. The full report includes block baseline, low/high weight policies, structural sensitivity and component ledgers; see [the 8K Q4 JSON report](27b-q4-8k.json).

After `moon build --target native --release`, reproduce a row with:

```sh
./_build/native/release/build/cmd/llmcalc/llmcalc.exe \
  --params 27B --quant Q4_K_M --vram 16GiB --ram 64GiB --context 8K --json
```

Replace quant and context to reproduce other rows. See [methodology](../docs/methodology.md), [sources](../docs/sources.md) and [memory policy](../docs/memory-policy.md). All results depend on the displayed uncalibrated weight and runtime policies; no hardware accuracy guarantee is implied.
