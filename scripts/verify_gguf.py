#!/usr/bin/env python3
"""Verify the real Native GGUF CLI with independent stdlib wire fixtures.

Usage: python3 scripts/verify_gguf.py PATH_TO_LLMCALC_BINARY

Fixtures contain synthetic zero payloads, not usable model parameters. A real
>4 GiB file is created with ftruncate without writing its payload, inspected,
and removed by TemporaryDirectory. Failure to create it is a failed check,
not evidence of large-file support. Filesystem allocation is reported rather
than assumed. This script does not test actual offsets above 2^53.
"""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory


GIB = 1 << 30
HEADER_BOUND = 64 * (1 << 20)
SCALAR_FORMAT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
# Independent raw GGML block payload oracles used by these fixtures only.
LAYOUT = {0: (1, 4), 1: (1, 2), 8: (32, 34), 12: (256, 144), 14: (256, 210)}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def value(type_id, data):
    if type_id == 8:
        return string(data)
    if type_id == 9:
        element_type, elements = data
        return struct.pack("<IQ", element_type, len(elements)) + b"".join(value(element_type, item) for item in elements)
    return struct.pack("<" + SCALAR_FORMAT[type_id], data)


@dataclass
class RawMetadata:
    key: str
    type_id: int
    payload: bytes


@dataclass
class Tensor:
    name: str
    dimensions: tuple
    type_id: int
    offset: int = None
    reserve: int = None

    def byte_size(self):
        layout = LAYOUT.get(self.type_id)
        if layout is None:
            return None
        block, block_bytes = layout
        return math.prod(self.dimensions) // block * block_bytes


def align(value, alignment=32):
    return (value + alignment - 1) // alignment * alignment


def make_file(path, metadata=(), tensors=(), version=3, alignment=32):
    """Encode wire fields independently; leave payload bytes as file holes."""
    header = bytearray(b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(metadata)))
    for entry in metadata:
        if isinstance(entry, RawMetadata):
            key, type_id, payload = entry.key, entry.type_id, entry.payload
        else:
            key, type_id, data = entry
            payload = value(type_id, data)
        header += string(key) + struct.pack("<I", type_id) + payload
    next_offset = 0
    records = []
    for tensor in tensors:
        offset = next_offset if tensor.offset is None else tensor.offset
        size = tensor.byte_size()
        reserve = tensor.reserve if tensor.reserve is not None else (size if size is not None else 1)
        header += string(tensor.name) + struct.pack("<I", len(tensor.dimensions))
        header += b"".join(struct.pack("<Q", dimension) for dimension in tensor.dimensions)
        header += struct.pack("<IQ", tensor.type_id, offset)
        records.append((tensor, offset, size, reserve))
        next_offset = max(next_offset, align(offset + reserve, alignment))
    data_offset = align(len(header), alignment)
    file_size = max([data_offset] + [data_offset + offset + reserve for _, offset, _, reserve in records])
    with path.open("wb") as file:
        file.write(header)
        file.flush()
        os.ftruncate(file.fileno(), file_size)
    return {"path": path, "data_offset": data_offset, "file_size": file_size, "header_bytes": len(header), "records": records}


def scalar_metadata():
    return [
        ("test.u8", 0, 255), ("test.i8", 1, -128),
        ("test.u16", 2, 65535), ("test.i16", 3, -32768),
        ("test.u32", 4, 4294967295), ("test.i32", 5, -2147483648),
        ("test.f32", 6, 1.5), ("test.bool", 7, True),
        ("test.说明", 8, "MoonBit 月兔 🚀"),
        ("test.u64", 10, 18446744073709551615),
        ("test.i64", 11, -9223372036854775808), ("test.f64", 12, -0.125),
    ]


def model_fixture(path, architecture="llama", tied=False):
    metadata = [
        ("general.architecture", 8, architecture),
        (architecture + ".block_count", 4, 2),
        (architecture + ".embedding_length", 4, 32),
        (architecture + ".context_length", 4, 4096),
        (architecture + ".feed_forward_length", 4, 64),
        (architecture + ".attention.head_count", 4, 1),
        (architecture + ".attention.head_count_kv", 4, 1),
        ("tokenizer.ggml.tokens", 9, (8, ["a", "b", "c", "d"])),
    ]
    tensors = [Tensor("token_embd.weight", (32, 4), 0), Tensor("output_norm.weight", (32,), 0)]
    if not tied:
        tensors.append(Tensor("output.weight", (32, 4), 0))
    for block in range(2):
        for leaf, dimensions in (
            ("attn_norm.weight", (32,)), ("attn_q.weight", (32, 32)),
            ("attn_k.weight", (32, 32)), ("attn_v.weight", (32, 32)),
            ("attn_output.weight", (32, 32)), ("ffn_norm.weight", (32,)),
            ("ffn_gate.weight", (32, 64)), ("ffn_down.weight", (64, 32)),
            ("ffn_up.weight", (32, 64)),
        ):
            tensors.append(Tensor(f"blk.{block}.{leaf}", dimensions, 1 if block == 0 else 0))
    return make_file(path, metadata, tensors)


class Checks:
    def __init__(self, binary, directory):
        self.binary, self.directory = binary, directory
        self.cases = self.processes = 0
        self.failures = []
        self.sparse_evidence = None

    def path(self, name):
        return self.directory / (name + ".gguf")

    def run(self, args):
        self.processes += 1
        return subprocess.run([str(self.binary), *map(str, args)], capture_output=True, text=True, timeout=20)

    def case(self, name, action):
        self.cases += 1
        try:
            action()
        except (AssertionError, ValueError, KeyError, TypeError, OSError, subprocess.TimeoutExpired, struct.error) as error:
            self.failures.append(f"{name}: {error}")

    def error(self, args):
        result = self.run(args)
        require(result.returncode in (1, 2), f"expected nonzero parser/planner exit, got {result.returncode}")
        require(not result.stdout, "failed command must leave stdout empty")
        require(result.stderr.startswith("llmcalc:"), "failure must explain itself on stderr")
        return result.stderr

    def report(self, args):
        result = self.run([*args, "--json"])
        require(result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
        require(not result.stderr, f"unexpected stderr: {result.stderr.strip()}")
        def reject_constant(constant):
            raise ValueError("nonstandard JSON constant: " + constant)
        document = json.loads(result.stdout, parse_constant=reject_constant)
        require(document["schema_version"] == "1", "unexpected schema version")
        artifact = document["artifact"]
        for name in ("file_size", "data_offset", "known_tensor_bytes"):
            require(isinstance(artifact[name], str) and artifact[name].isdigit(), f"artifact.{name} must be an exact decimal string")
        for name in ("bytes_read", "max_read_end"):
            require(isinstance(artifact["io"][name], str) and artifact["io"][name].isdigit(), f"io.{name} must be an exact decimal string")
        require(int(artifact["io"]["max_read_end"]) <= int(artifact["data_offset"]), "inspection read tensor payload bytes")
        require(int(artifact["io"]["bytes_read"]) < HEADER_BOUND, "inspection read exceeds header resource bound")
        if document["command"] == "inspect":
            for tensor in document["tensors"]:
                for name in ("offset", "absolute_offset", "element_count"):
                    require(isinstance(tensor[name], str) and tensor[name].isdigit(), f"tensor.{name} must be an exact decimal string")
                require(all(isinstance(size, str) and size.isdigit() for size in tensor["dimensions"]), "dimensions must be decimal strings")
                require(tensor["byte_size"] is None or isinstance(tensor["byte_size"], str), "optional tensor bytes must be string/null")
        return document

    def inspect(self, fixture):
        report = self.report(["inspect", fixture["path"]])
        require(report["artifact"]["file_size"] == str(fixture["file_size"]), "file size differs from actual fixture")
        require(report["artifact"]["data_offset"] == str(fixture["data_offset"]), "data offset differs from independently encoded directory")
        return report

    def bad_fixture(self, name, metadata=(), tensors=(), raw=None):
        path = self.path(name)
        if raw is None:
            make_file(path, metadata, tensors)
        else:
            path.write_bytes(raw)
        self.error(["inspect", path, "--json"])

    def verify_plan(self, fixture, tied=False):
        args = ["plan", fixture["path"], "--vram=16GiB", "--ram=64GiB", "--context=1K"]
        report = self.report(args)
        require(report["fit"]["status"] == "MAX_OFFLOAD", "maximum offload must not be called FULL_GPU")
        plan = report["fit"]["plan"]
        payload = sum(size for _, _, size, _ in fixture["records"])
        require(report["profile"]["tensor_payload_bytes"] == str(payload), "profile payload differs from independent stored tensor sizes")
        require(report["profile"]["tied_output"] is tied, "tied output inference")
        require(int(plan["gpu_weight_bytes"]) + int(plan["cpu_weight_bytes"]) == payload + int(plan["duplicated_bytes"]), "placement payload conservation")
        require(plan["n_gpu_layers"] == 3 and plan["gpu_block_count"] == 2 and plan["output_on_gpu"], "output slot must be distinct from repeating blocks")
        require(all(item["device"] == "CPU" for item in plan["placements"] if item["name"] == "token_embd.weight" and not item["is_duplicate"]), "input embedding must remain on CPU")
        copies = [item for item in plan["placements"] if item["is_duplicate"]]
        require(len(copies) == (1 if tied else 0), "only tied output should create a duplicated placement")
        for device in ("gpu", "host"):
            for phase in ("steady", "loading_peak"):
                require(plan["ledger"][f"{device}_{phase}_bytes"] <= (16 if device == "gpu" else 64) * GIB, "placement exceeds independent device budget")
        result = self.run(args)
        require(result.returncode == 0 and not result.stderr, "plan text must succeed")
        require("Fit: MAX_OFFLOAD" in result.stdout, "text/JSON plan status mismatch")


def exercise(checks):
    def scalar_case(version):
        fixture = make_file(checks.path(f"元数据 scalars-v{version}"), scalar_metadata(), version=version)
        report = checks.inspect(fixture)
        require(report["artifact"]["gguf_version"] == version, "GGUF version mismatch")
        metadata = {entry["key"]: entry["value"] for entry in report["metadata"]}
        for key, type_id, expected in scalar_metadata():
            actual = metadata[key]["value"]
            if type_id in (0, 1, 2, 3, 4, 5, 10, 11):
                require(actual == str(expected), f"scalar integer mismatch: {key}")
            else:
                require(actual == expected, f"scalar/string mismatch: {key}")
    for version in (2, 3):
        checks.case(f"GGUF v{version} all scalar metadata and UTF-8 path", lambda version=version: scalar_case(version))

    def array_case():
        entries = [(key + ".array", 9, (type_id, [data, data])) for key, type_id, data in scalar_metadata()]
        report = checks.inspect(make_file(checks.path("all-1D-arrays"), entries))
        for entry, (_, _, (element_type, _)) in zip(report["metadata"], entries):
            summary = entry["value"]
            require(summary["kind"] == "array" and summary["element_type_id"] == element_type, "array element type lost")
            require(summary["total_count"] == "2" and summary["retention_complete"] and summary["retained_value_count"] == 2, "small array retention count mismatch")
            require(summary["omitted"] and summary["value"] is None, "report array omission must be explicit")
    checks.case("one-dimensional arrays of every scalar/string type", array_case)

    def nonfinite_case():
        fixture = make_file(checks.path("nonfinite-floats"), [
            ("test.nan_f32", 6, float("nan")),
            ("test.inf_f64", 12, float("inf")),
            ("test.neg_inf_f64", 12, float("-inf")),
        ])
        # report() rejects bare NaN/Infinity with json.loads(parse_constant=...).
        # GGUF float values are instead emitted as tagged JSON strings.
        report = checks.inspect(fixture)
        values = {entry["key"]: entry["value"] for entry in report["metadata"]}
        for key, expected in (("test.nan_f32", "NaN"), ("test.inf_f64", "Infinity"), ("test.neg_inf_f64", "-Infinity")):
            require(values[key]["kind"] == "float" and values[key]["value"] == expected, "nonfinite metadata must remain valid JSON with explicit float kind")
    checks.case("nonfinite float metadata remains strict valid JSON", nonfinite_case)

    def mixed_case():
        tensors = [Tensor("f32", (3,), 0), Tensor("f16", (5,), 1), Tensor("q4_k", (256,), 12), Tensor("q6_k", (256,), 14), Tensor("q8_0", (32,), 8)]
        fixture = make_file(checks.path("mixed-types"), tensors=tensors)
        report = checks.inspect(fixture)
        expected = {"f32": 12, "f16": 10, "q4_k": 144, "q6_k": 210, "q8_0": 34}
        require(report["artifact"]["complete"] and report["artifact"]["total_tensor_bytes"] == "410", "mixed payload total must exclude padding")
        for tensor, (_, offset, _, _) in zip(report["tensors"], fixture["records"]):
            require(tensor["byte_size"] == str(expected[tensor["name"]]), "mixed raw payload byte oracle")
            require(tensor["offset"] == str(offset) and tensor["absolute_offset"] == str(fixture["data_offset"] + offset), "tensor directory offsets mismatch")
    checks.case("mixed F32 F16 Q4_K Q6_K Q8_0 payload and directory oracles", mixed_case)

    def retention_case():
        fixture = make_file(checks.path("retention"), [("large.array", 9, (4, list(range(300))))])
        report = checks.inspect(fixture)
        summary = report["metadata"][0]["value"]
        require(summary["total_count"] == "300" and summary["retained_value_count"] == 0 and not summary["retention_complete"], "large array must not be materialized")
        require(any("retention" in warning for warning in report["artifact"]["warnings"]), "retention omission must be disclosed")
        require(int(report["artifact"]["io"]["bytes_read"]) < fixture["header_bytes"] - 1000, "large fixed-width array payload was read instead of skipped")
    checks.case("large array retention omission and bounded reads", retention_case)

    def unknown_case():
        fixture = make_file(checks.path("unknown-type"), tensors=[Tensor("known", (1,), 0), Tensor("unknown", (32,), 999)])
        report = checks.inspect(fixture)
        require(not report["artifact"]["complete"] and report["artifact"]["total_tensor_bytes"] is None, "unknown type cannot produce a fabricated total")
        require(report["artifact"]["known_tensor_bytes"] == "4" and report["tensors"][1]["byte_size"] is None, "known partial total and unknown span must remain distinct")
        checks.error(["plan", fixture["path"], "--vram=16GiB", "--ram=64GiB", "--context=8K", "--json"])
    checks.case("unknown type is inspectable incomplete and unplannable", unknown_case)

    for architecture, tied in (("llama", False), ("qwen2", True)):
        checks.case(f"{architecture} real-process plan tied={tied}", lambda architecture=architecture, tied=tied: checks.verify_plan(model_fixture(checks.path(f"{architecture}-plan"), architecture, tied), tied))

    def cpu_and_insufficient():
        fixture = model_fixture(checks.path("cpu-only"))
        for ram, status in (("64GiB", "CPU ONLY"), ("0GiB", "INSUFFICIENT")):
            report = checks.report(["plan", fixture["path"], "--vram=0GiB", "--ram=" + ram, "--context=1K"])
            require(report["fit"]["status"] == status, "zero-budget plan status mismatch")
            require((report["fit"]["plan"] is None) == (status == "INSUFFICIENT"), "insufficient plan must be null")
    checks.case("CPU-only and insufficient are valid plan results", cpu_and_insufficient)

    def sparse_case():
        offset = (1 << 32) + 32
        try:
            fixture = make_file(checks.path("actual-large-sparse"), [("test.exact_u64", 10, 9007199254740993)], [Tensor("large-offset.weight", (1,), 0, offset=offset)])
        except OSError as error:
            raise AssertionError(f">4 GiB sparse ftruncate NOT VERIFIED; filesystem/OS failure: {error}") from error
        stat = fixture["path"].stat()
        require(stat.st_size > 4 * GIB, "real file is not larger than 4 GiB")
        report = checks.inspect(fixture)
        require(report["tensors"][0]["offset"] == str(offset), "large relative offset precision lost")
        require(report["tensors"][0]["absolute_offset"] == str(fixture["data_offset"] + offset), "large absolute offset precision lost")
        require(report["metadata"][0]["value"]["value"] == "9007199254740993", "metadata above 2^53 lost precision")
        require(int(report["artifact"]["io"]["bytes_read"]) <= fixture["header_bytes"], "large file inspection read more than directory bytes")
        blocks = getattr(stat, "st_blocks", None)
        allocated = str(blocks * 512) if blocks is not None else "unavailable"
        checks.sparse_evidence = f"logical_bytes={stat.st_size}, st_blocks={blocks}, allocated_bytes={allocated}, bytes_read={report['artifact']['io']['bytes_read']}, max_read_end={report['artifact']['io']['max_read_end']}, data_offset={fixture['data_offset']}"
    checks.case("actual >4 GiB sparse file bounded Native reads", sparse_case)

    def malformed_headers():
        for name, raw in (
            ("bad-magic", b"NOPE" + struct.pack("<IQQ", 3, 0, 0)),
            ("future-version", b"GGUF" + struct.pack("<IQQ", 4, 0, 0)),
            ("truncated-header", b"GGUF\x03\x00\x00"),
        ):
            checks.bad_fixture(name, raw=raw)
    checks.case("wrong magic unsupported version and truncated header", malformed_headers)
    checks.case("truncated metadata string", lambda: checks.bad_fixture("truncated-string", raw=b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + string("key") + struct.pack("<IQ", 8, 20) + b"short"))

    def truncated_payload():
        fixture = make_file(checks.path("truncated-payload"), tensors=[Tensor("f32", (4,), 0)])
        os.truncate(fixture["path"], fixture["file_size"] - 1)
        checks.error(["inspect", fixture["path"], "--json"])
    checks.case("known tensor payload range truncated by one byte", truncated_payload)
    checks.case("overlapping tensor ranges", lambda: checks.bad_fixture("overlap", tensors=[Tensor("a", (16,), 0, offset=0), Tensor("b", (1,), 0, offset=32)]))
    checks.case("unaligned tensor offset", lambda: checks.bad_fixture("unaligned", tensors=[Tensor("a", (1,), 0, offset=1)]))
    checks.case("dimension product UInt64 overflow", lambda: checks.bad_fixture("dimension-overflow", tensors=[Tensor("a", (1 << 63, 3), 0, reserve=1)]))
    checks.case("tensor byte size UInt64 overflow", lambda: checks.bad_fixture("byte-overflow", tensors=[Tensor("a", (1 << 63,), 0, reserve=1)]))

    def offset_overflow():
        # Use raw directory bytes so no impossible operating-system size is requested.
        raw = b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + string("a") + struct.pack("<IQIQ", 1, 1, 0, (1 << 64) - 32)
        checks.bad_fixture("offset-overflow", raw=raw + b"\0" * (128 - len(raw)))
    checks.case("data offset addition UInt64 overflow", offset_overflow)
    checks.case("duplicate tensor names", lambda: checks.bad_fixture("duplicate-tensor", tensors=[Tensor("a", (1,), 0), Tensor("a", (1,), 0)]))
    checks.case("duplicate metadata keys", lambda: checks.bad_fixture("duplicate-key", metadata=[("key", 4, 1), ("key", 4, 2)]))
    checks.case("unknown metadata value type", lambda: checks.bad_fixture("unknown-metadata", metadata=[RawMetadata("key", 99, b"")]))

    def big_endian():
        for magic in (b"GGUF", b"FUGG"):
            checks.bad_fixture("big-endian-" + magic.decode(), raw=magic + struct.pack(">IQQ", 3, 0, 0))
    checks.case("big-endian version/magic rejected explicitly", big_endian)
    checks.case("nested metadata arrays unsupported", lambda: checks.bad_fixture("nested-array", metadata=[RawMetadata("key", 9, struct.pack("<IQ", 9, 1))]))
    checks.case("quantization row block alignment", lambda: checks.bad_fixture("bad-block", tensors=[Tensor("a", (257,), 12)]))

    def bad_alignment():
        for alignment in (0, 3):
            checks.bad_fixture(f"bad-alignment-{alignment}", metadata=[("general.alignment", 4, alignment)])
    checks.case("invalid general alignment metadata", bad_alignment)

    def resource_limits():
        for name, raw in (
            ("entry-limit", b"GGUF" + struct.pack("<IQQ", 3, 0, 100001)),
            ("string-limit", b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", (1 << 20) + 1)),
            ("array-limit", b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + string("key") + struct.pack("<IIQ", 9, 4, 2000001)),
        ):
            checks.bad_fixture(name, raw=raw)
    checks.case("configured entry string and array limits reject compact adversarial headers", resource_limits)

    def malformed_scalars():
        checks.bad_fixture("bad-boolean", metadata=[RawMetadata("key", 7, b"\x02")])
        checks.bad_fixture("bad-utf8", metadata=[RawMetadata("key", 8, struct.pack("<Q", 1) + b"\xff")])
    checks.case("invalid Boolean and retained UTF-8 scalar", malformed_scalars)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="path to an already-built Native llmcalc executable")
    binary = parser.parse_args().binary.resolve()
    if not binary.is_file():
        parser.error(f"executable does not exist: {binary}")
    with TemporaryDirectory(prefix="llmcalc-gguf-") as temporary:
        checks = Checks(binary, Path(temporary))
        exercise(checks)
        for failure in checks.failures:
            print("FAIL: " + failure, file=sys.stderr)
        if checks.sparse_evidence:
            print("Actual sparse-file evidence: " + checks.sparse_evidence)
        print(f"GGUF Native verification: {checks.cases - len(checks.failures)}/{checks.cases} cases passed; {checks.processes} real process checks")
        return 1 if checks.failures else 0


if __name__ == "__main__":
    sys.exit(main())
