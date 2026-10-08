# Development toolchain

Verified on macOS arm64, 8 October 2026:

- `moon 0.1.20260920 (914d7da 2026-09-20)`
- `moonc v0.10.14+7d59c7ec9 (2026-09-18)`
- Native backend; `moon.mod` and `moon.pkg` configuration formats.
- No external Mooncakes dependencies.

Install the matching toolchain using the [official download instructions](https://www.moonbitlang.com/download/). For this development session the official archives were extracted into a workspace-local directory, leaving shell configuration untouched. The downloaded arm64 archive was verified against its official SHA-256: `20967f9389ac54508899ee2fc051a3470d051452bac5818a9696ed4c144f3ec3`.

From the repository root:

```sh
moon check --target native
moon test --target native
moon fmt --check
```
