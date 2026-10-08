# Native continuous integration

The [workflow](../.github/workflows/ci.yml) runs on pushes to `main`, pull requests and manual dispatch. It uses `ubuntu-22.04`, read-only repository permissions and a 20-minute job timeout. CI currently targets native execution only. This document describes the configured checks; a successful remote run must be confirmed in GitHub Actions before claiming that Linux CI passes.

## Fixed inputs

The checkout action is pinned to [`11d5960a326750d5838078e36cf38b85af677262`](https://github.com/actions/checkout/commit/11d5960a326750d5838078e36cf38b85af677262), the verified `v4` commit on 8 October 2026. It does not persist checkout credentials. MoonBit installation uses official download URLs directly and does not depend on a setup action or its transitive tooling.

| Input | Fixed value |
| --- | --- |
| Toolchain download version | `0.10.14+7d59c7ec9` |
| [Linux x86_64 archive](https://cli.moonbitlang.com/binaries/0.10.14%2B7d59c7ec9/moonbit-linux-x86_64.tar.gz) SHA-256 | `9226694de9ff978db1ecf820b7710c4224e84ec7a76b19a222d96f0cd4e31b6a` |
| [Matching Core archive](https://cli.moonbitlang.com/cores/core-0.10.14%2B7d59c7ec9.tar.gz) SHA-256 | `6f18b8fdea18f85e628a75e4a1bd3977c5a5c9c6a836fd8824192b0e6bd91b14` |
| Expected `moon` | `0.1.20260920 (914d7da 2026-09-20)` |
| Expected `moonc` | `v0.10.14+7d59c7ec9 (2026-09-18)` |

The toolchain checksum was read from the [official archive checksum](https://cli.moonbitlang.com/binaries/0.10.14%2B7d59c7ec9/moonbit-linux-x86_64.tar.gz.sha256). The Core checksum was calculated from the complete 1,782,652-byte official archive fetched on 8 October 2026; its separate `.sha256` URL returned HTTP 403. It is a recorded content hash, rather than a published official checksum. Both hashes are checked before extraction, and mismatches fail the job.

The archive path uses the compiler version, not the `moon` build date. The `+` is URL-encoded as `%2B`. Neither archive URL uses `latest`, so a newer release will not silently replace the tested toolchain. Installation stays under `RUNNER_TEMP`; `MOON_HOME` and the toolchain path are exported through GitHub's environment/path files. The job checks both tool versions before bundling the native standard library.

## Checks

From the repository root, after bundling Core for native:

```sh
moon check --target native --deny-warn
moon test --target native --deny-warn
moon fmt --check
moon build --target native --release --deny-warn
python3 scripts/verify_cli.py _build/native/release/build/cmd/llmcalc/llmcalc.exe
```

Compilation and tests treat warnings as errors. Formatting is checked without changing sources. The integration script runs the release executable and checks CLI behavior and machine-readable output independently of package unit tests. Ubuntu's existing Python 3 runs the script; no Python packages are required.

A green job verifies these checks on that GitHub-hosted Linux runner. It does not establish hardware memory accuracy, inference-backend compatibility, performance, JS/Wasm support or successful model loading. Changing the fixed toolchain requires updating both artifact hashes and expected versions together, then confirming a new CI run.
