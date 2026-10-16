# GGUF directory inspection

`inspect` reads a real model's metadata and tensor directory. The parser, storage arithmetic, validation and reporting are implemented in MoonBit. It does not load or dequantize tensor payloads.

```sh
moon run src/cmd/llmcalc --target native -- inspect /path/to/model.gguf --json
```

Replace the example path with an existing local file. Inspection does not require the model architecture to be supported by the [artifact planner](artifact-planning.md).

## Source and supported format

The format reference is the [GGUF specification at ggml `a05c9922613c6a2230dbb1d0de679519ecafb4f0`](https://github.com/ggml-org/ggml/blob/a05c9922613c6a2230dbb1d0de679519ecafb4f0/docs/gguf.md). Storage types were checked against [GGML block definitions](https://github.com/ggml-org/llama.cpp/blob/88aaf325a7026ac1127d0897394b0a3d6413dc2a/ggml/src/ggml-common.h) in llama.cpp `88aaf325a7026ac1127d0897394b0a3d6413dc2a`. The [upstream C++ reader](https://github.com/ggml-org/llama.cpp/blob/88aaf325a7026ac1127d0897394b0a3d6413dc2a/ggml/src/gguf.cpp) is a compatibility reference, rather than an inference engine embedded by this project.

The implemented subset is one little-endian GGUF v2 or v3 file. Metadata supports the standard scalar integer, floating-point, Boolean and string types, plus one-dimensional arrays of scalar values or strings. Nested arrays, big-endian files and automatic aggregation of split GGUF files are outside this version. Scalar metadata is supported; tensor descriptors require 1–4 dimensions with strictly positive sizes. This tensor restriction is a project boundary: the upstream reader can accept zero-dimensional or zero-element tensors.

The parser preserves metadata keys and tensor descriptors even when it cannot identify a tensor storage type. It does not interpret an unrecognized architecture as one of the planner's supported architectures.

## Random-access input and resource limits

The pure parser accepts `ReaderAt { size: UInt64, read: (UInt64, Int) -> Result[Bytes, String] }`. A read must return exactly the requested bytes or an error. The parser validates the requested range against the declared file size and rejects short reads. Offsets, lengths, element products and tensor byte spans use checked `UInt64` arithmetic; addition, multiplication and alignment must not wrap.

Default resource limits are deliberately smaller than the full integer ranges of the file format:

| Limit | Default |
| --- | ---: |
| Parsed header/directory byte budget | 64 MiB |
| Metadata entries | 100,000 |
| Tensor descriptors | 200,000 |
| Bytes in an individual string | 1 MiB |
| Elements in an individual metadata array | 2,000,000 |
| Array values retained in memory | 256 |

`Limits` can be supplied to the library API within its validated resource bounds. The CLI uses the defaults. These limits are allocation and work policies, not claims about maximum sizes permitted by GGUF. A parser/adapter read request is also bounded to 16 MiB.

Arrays up to the retention threshold are decoded and retained. Larger fixed-width numeric arrays are skipped after checked count × element-width and file-range validation. Larger Boolean arrays are scanned to validate each Boolean. For larger string arrays, every string length is read and bounded, and each string body is skipped. Those skipped string bodies are **not read or UTF-8 validated**. Retained strings are UTF-8 decoded; invalid UTF-8 in retained metadata keys, strings or tensor names is rejected. Array counts and retention status remain available even when their values are omitted.

Parser retention and CLI rendering are separate decisions. The CLI reports array summaries rather than array contents, including for arrays retained by the parser. It also omits long scalar string values from the report. These presentation choices do not remove the parser's retained-string validation.

The native adapter in `src/nativeio` contains a small C shim for read-only `open`, `fstat`, `pread` and `close`. It contains no GGUF decoding, model formulas or placement logic. It requires a regular file and a 64-bit `off_t`, retries interrupted reads, closes the descriptor on success and error paths, and checks that file size did not change during inspection. Native offsets must fit the operating system's signed 64-bit file-offset range. A same-size content change is not detected as a checksum failure.

`IoStats` records read calls, bytes read, maximum read end and largest request. Metadata inspection should access only the header/directory region; a large model file does not imply an equally large read or allocation. Sparse-file and process-level checks are described in [validation](validation.md).

## Tensor storage and range checks

For each descriptor, the parser records its name, dimensions, numeric type ID, relative and absolute offsets, element count, and known byte size. Names and metadata keys must be nonempty and unique within their respective collections. The implementation limits metadata keys to 1,024 UTF-8 bytes and tensor names to 64 UTF-8 bytes. A type ID greater than `Int::MAX` is rejected; smaller unknown IDs can be retained as unknown types.

Known tensor sizes use the block-element count and serialized block bytes in [the versioned registry](../src/gguf/layout.mbt). It covers the active scalar and quantized layouts in the pinned snapshot, including K-quants, IQ quants, TQ, MXFP4, NVFP4, Q1_0 and Q2_0. Removed/reserved IDs are not assigned guessed sizes. Knowing storage bytes does not establish that a type is usable as a model weight or supported by a particular device kernel; the planner accepts a smaller subset.

For a known layout:

```text
elements = product(dimensions)
payload_bytes = (elements / block_elements) × block_bytes
```

The first dimension must be divisible by the quantization block length. Block scales are included in `block_bytes`. For example, Q4_K is 144 bytes per 256 elements and Q8_0 is 34 bytes per 32 elements. File padding and metadata are excluded from tensor payload totals.

`general.alignment` defaults to 32 bytes. This implementation requires its declared metadata type to be UINT32 and its value to be a power of two from 1 to 65,536. Relative tensor offsets must meet that alignment. A tensor's absolute start is the aligned data-section start plus its relative offset, with checked arithmetic. Every known tensor end must fit the file size. Known spans must not overlap, and duplicate starts are rejected. A directory may contain aligned gaps or be ordered differently from its data spans; this differs from the pinned C++ reader's stricter contiguous-offset policy. Successful inspection is not a claim of identical upstream acceptance.

An unknown type has `byte_size: null`, generates a warning, and makes the full inventory `complete: false`. Its aligned start must still be inside the file. The parser cannot establish its full extent, payload completeness or overlap with a later tensor. It does not infer its bytes from the next offset. Reports retain a known-tensor subtotal and leave the complete total unknown. Planning rejects this incomplete inventory.

An empty directory does not require trailing padding bytes merely to align a nonexistent data section.

## Integer reporting and verification boundary

Raw file sizes, offsets, dimensions, tensor byte counts and element counts are emitted as decimal JSON strings. This preserves integer values above JavaScript's `2^53` precision boundary. Type IDs, descriptor counts and other bounded small values remain JSON numbers. GiB conversions and runtime/KV planning quantities are separate presentation or calculation values.

Inspection verifies directory structure and the file ranges it can determine. It does not read tensor contents, verify payload checksums, detect corrupt quantized values, establish tokenizer correctness or prove that a backend can load or run the model. `complete: true` means that all tensor sizes are known to this registry; it does not mean that payload contents have been validated.

These are implementation and format guarantees. Backend measurements and their hardware scope belong in [validation](validation.md); this document makes no GPU validation or empirical accuracy claim.
