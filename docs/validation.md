# Verification evidence

Verified locally on macOS arm64, 8 October 2026, with the compiler recorded in [toolchain](toolchain.md).

| Check | Result |
| --- | --- |
| Native input/formula/placement/report/CLI tests, warnings denied | 54/54 passed |
| Native release build, warnings denied | Passed |
| Repository formatting check | Passed |
| Python native CLI integration verifier | 42/42 cases passed; 44 real process checks |

The core tests include independent scale-inclusive quantization and KV byte fixtures, invalid and non-finite values, zero-GPU behavior, architecture-bucket boundaries, GQA divisibility, separate GPU/RAM limits, one-byte placement boundaries, loading peaks, allocation conservation and increasing-capacity/context properties. A 7/14/27/32/70B model matrix with 4K/8K/32K/64K contexts checks report consistency. A synthetic maximum-domain MHA scenario verifies that an infeasible sensitivity does not cancel a valid baseline.

The real-process verifier parses JSON independently with Python, checks all four fit labels, text/JSON agreement, component budget conservation, a 27B × four weight recipes × four contexts matrix, Advanced Q8 KV, CPU-only/insufficient cases, decimal GB conversion and stderr/nonzero exits for invalid input.

These checks establish that the program follows its documented formulas and policy. They are not inference hardware tests, backend compatibility certification, accuracy measurements or performance benchmarks. No model was loaded on a GPU. JS, Wasm and the optional web interface were not validated.

Linux [CI](ci.md) is configured to repeat the same checks with a fixed compiler and source archive. Its latest actual outcome is visible in [GitHub Actions](https://github.com/Nicksxs/llm-hardware-calculator/actions/workflows/ci.yml); configuration alone does not establish a successful run.
