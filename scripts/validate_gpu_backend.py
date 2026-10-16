#!/usr/bin/env python3
"""Validate fixed output-first offload against one pinned CUDA backend.

Standard library only. This script runs inference when invoked. It refuses to
start a backend run if nvidia-smi cannot establish at least 2 GiB free on GPU 0.
Device-wide samples are observations, not process VRAM or an allocation peak.

Example (run only after authorizing use of the otherwise idle GPU):
  python3 scripts/validate_gpu_backend.py --calculator PATH --backend PATH \
      --model stories15M-q4_0.gguf --output gpu-validation.json
An optional --tied-model adds two runs for a derived stories260K tied fixture.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time


REVISION = "88aaf325a7026ac1127d0897394b0a3d6413dc2a"
MODEL_SHA256 = "6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04"
MODEL_SIZE = 19077344
TIED_MODEL_SHA256 = "67af9dce730ec642b9648dfa18488597b049811292cbff7317f9fe8d6090f196"
TIED_MODEL_SIZE = 1054464
MIN_FREE_MIB = 2048
MIB = 1024 ** 2
GIB = 1024 ** 3
# GGML serialized block layouts at REVISION, not effective average bits/weight.
# Special transformed/split/ACCEL buffer types are deliberately unsupported.
LAYOUTS = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
    6: (32, 22), 7: (32, 24), 8: (32, 34),
    10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 16: (256, 66),
    17: (256, 74), 18: (256, 98), 19: (256, 50),
    20: (32, 18), 21: (256, 110), 22: (256, 82),
    23: (256, 136), 29: (256, 56), 30: (1, 2),
}
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
           6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
STRUCTURE_SUFFIXES = {
    "block_count", "embedding_length", "context_length", "attention.head_count",
    "attention.head_count_kv", "attention.key_length", "attention.value_length",
}
BUFFER_RE = re.compile(
    r"(?P<buffer>\S+)\s+(?P<kind>model|KV|compute|output) buffer size\s*=\s*"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s+(?P<unit>KiB|MiB|GiB)"
)
MODEL_LAYER_RE = re.compile(
    r"load_tensors: layer\s+(\d+)\s+assigned to device (\S+), is_swa = ([01])"
)
KV_LAYER_RE = re.compile(r"llama_kv_cache: layer\s+(\d+): dev = (\S+)")
FALLBACK_RE = re.compile(
    r"cannot be used with preferred buffer type|no usable GPU found|"
    r"gpu-layers option will be ignored|failed to allocate|out of memory|CUDA error",
    re.IGNORECASE,
)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(text):
    def reject(value):
        raise ValueError("nonstandard JSON number: " + value)
    return json.loads(text, parse_constant=reject)


def exact(value, label):
    require(isinstance(value, str) and value.isascii() and value.isdigit(),
            label + " must be an exact decimal JSON string")
    return int(value)


def align_up(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def acceptance_cases(model, tied):
    cases = [(model, 256, ngl) for ngl in (0, 1, 2, 7)] + [(model, 257, 7)]
    return cases + ([(tied, 256, ngl) for ngl in (1, 6)] if tied else [])


def same_file_or_path(left, right):
    if left.resolve() == right.resolve():
        return True
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return False


def validate_output_targets(output_argument, inputs, cases):
    """Read-only guard; callers must not catch its rejection and write a report.

    Require a fresh report and fresh log names. Resolve symlinks and compare
    existing inode identities as well, so hard links cannot overwrite inputs.
    """
    output = output_argument.resolve()
    logs = []
    for index, (model, context, ngl) in enumerate(cases, 1):
        base = output.stem + "-" + str(index) + "-" + model.stem + "-ctx" + str(context) + "-ngl" + str(ngl)
        logs.append((output.parent / (base + ".stderr.log"), output.parent / (base + ".stdout.log")))
    targets = [output] + [path for pair in logs for path in pair]
    require(len({path.resolve() for path in targets}) == len(targets), "output/log target names collide")
    for directory in [output.parent, *output.parent.parents]:
        require(not any(same_file_or_path(directory, source) for source in inputs),
                "output directory or ancestor aliases an input")
        require(not directory.exists() or directory.is_dir(), "output ancestor is not a directory")
    for target in targets:
        require(not any(same_file_or_path(target, source) for source in inputs),
                "output/log target aliases an input: " + target.name)
        require(not os.path.lexists(target), "output/log target already exists: " + target.name)
    # A dangling symlink is also an existing output entry, even though exists()
    # and resolving its missing target would otherwise hide that fact.
    require(not os.path.lexists(output_argument), "output entry already exists")
    return output, logs


def write_new_text(path, text):
    # Exclusive creation also refuses an input/symlink created after preflight.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


class HeaderReader:
    """Independent bounded GGUF directory reader; never reads tensor payload."""

    def __init__(self, stream, size):
        self.stream, self.size = stream, size

    def read(self, size):
        self.bound(size)
        value = self.stream.read(size)
        require(len(value) == size, "truncated GGUF directory")
        return value

    def bound(self, size):
        end = self.stream.tell() + size
        require(size >= 0 and end <= self.size and end <= 128 * MIB,
                "GGUF header exceeds the independent reader bound")

    def skip(self, size):
        self.bound(size)
        self.stream.seek(size, os.SEEK_CUR)

    def number(self, code):
        return struct.unpack("<" + code, self.read(struct.calcsize("<" + code)))[0]

    def string(self, retain=True):
        size = self.number("Q")
        if retain:
            require(size <= MIB, "retained GGUF string is too long")
            return self.read(size).decode("utf-8")
        self.skip(size)
        return None

    def value(self, type_id, retain=False):
        if type_id in SCALARS:
            value = self.number(SCALARS[type_id])
            return value if retain else None
        if type_id == 8:
            return self.string(retain)
        require(type_id == 9, "unknown GGUF metadata type")
        element_type, count = self.number("I"), self.number("Q")
        require(count <= 10000000 and element_type != 9,
                "nested or oversized GGUF array is outside the validation reader")
        if element_type in SCALARS:
            self.skip(count * struct.calcsize("<" + SCALARS[element_type]))
        else:
            require(element_type == 8, "unknown GGUF array element type")
            for _ in range(count):
                self.string(False)
        require(not retain, "required structure metadata is not scalar")
        return None


def read_directory(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        reader = HeaderReader(stream, size)
        require(reader.read(4) == b"GGUF", "expected little-endian GGUF magic")
        version, tensor_count, metadata_count = reader.number("I"), reader.number("Q"), reader.number("Q")
        require(version in (2, 3), "expected GGUF v2 or v3")
        require(0 < tensor_count <= 100000 and metadata_count <= 1000000,
                "directory count exceeds the validation reader bound")
        metadata, keys = {}, set()
        for _ in range(metadata_count):
            key, type_id = reader.string(), reader.number("I")
            require(key not in keys, "duplicate GGUF metadata key")
            keys.add(key)
            retain = key in {"general.architecture", "general.alignment"} or any(
                key.endswith("." + suffix) for suffix in STRUCTURE_SUFFIXES)
            value = reader.value(type_id, retain)
            if retain:
                metadata[key] = value
        alignment = metadata.get("general.alignment", 32)
        require(type(alignment) is int and 0 < alignment <= MIB and alignment & (alignment - 1) == 0,
                "invalid GGUF data alignment")
        tensors, names = [], set()
        for _ in range(tensor_count):
            name, ndim = reader.string(), reader.number("I")
            require(name not in names and 1 <= ndim <= 4, "duplicate name or invalid tensor rank")
            names.add(name)
            shape = [reader.number("Q") for _ in range(ndim)]
            type_id, offset = reader.number("I"), reader.number("Q")
            require(type_id in LAYOUTS and all(d > 0 for d in shape),
                    "tensor type or dimensions outside the allocation oracle")
            block, block_bytes = LAYOUTS[type_id]
            require(shape[0] % block == 0, "incomplete quantized tensor row")
            elements = math.prod(shape)
            payload = elements // block * block_bytes
            require(offset % alignment == 0, "unaligned GGUF tensor offset")
            tensors.append({"name": name, "dimensions": shape, "type_id": type_id,
                            "element_count": elements, "byte_size": payload, "offset": offset})
        data_offset = align_up(stream.tell(), alignment)
        ranges = []
        for tensor in tensors:
            tensor["absolute_offset"] = data_offset + tensor["offset"]
            end = tensor["absolute_offset"] + tensor["byte_size"]
            require(end <= size, "tensor payload extends beyond file")
            ranges.append((tensor["absolute_offset"], end))
        ranges.sort()
        require(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), "overlapping tensor payloads")
    architecture = metadata.get("general.architecture")
    require(architecture in ("llama", "qwen2"), "unsupported architecture for this CUDA acceptance matrix")
    prefix = architecture + "."
    positive = lambda key: type(metadata.get(prefix + key)) is int and metadata[prefix + key] > 0
    require(all(positive(k) for k in ("block_count", "embedding_length", "context_length", "attention.head_count")),
            "missing positive scalar structure metadata")
    layers, embedding, heads = (metadata[prefix + key] for key in ("block_count", "embedding_length", "attention.head_count"))
    require(embedding % heads == 0, "embedding/head count cannot determine default head dimension")
    head_dim = metadata.get(prefix + "attention.key_length", embedding // heads)
    value_dim = metadata.get(prefix + "attention.value_length", embedding // heads)
    kv_heads = metadata.get(prefix + "attention.head_count_kv", heads)
    require(type(kv_heads) is int and type(head_dim) is int and 0 < kv_heads <= heads
            and heads % kv_heads == 0 and head_dim > 0 and value_dim == head_dim,
            "invalid or unequal key/value structure")
    return {"version": version, "file_size": size, "data_offset": data_offset,
            "metadata_count": metadata_count, "tensors": tensors, "architecture": architecture,
            "layers": layers, "embedding": embedding, "query_heads": heads,
            "kv_heads": kv_heads, "head_dim": head_dim,
            "tied_output": "output.weight" not in names,
            "payload_bytes": sum(t["byte_size"] for t in tensors)}


def verify_tied_fixture(path):
    require(path.is_file() and path.stat().st_size == TIED_MODEL_SIZE and sha256(path) == TIED_MODEL_SHA256,
            "tied model differs from the fixed derived stories260K artifact")
    directory = read_directory(path)
    require(len(directory["tensors"]) == 47 and directory["layers"] == 5 and directory["tied_output"],
            "fixed tied fixture has an unexpected tensor directory")
    return directory


def oracle(directory, context, ngl):
    layers = directory["layers"]
    require(0 <= ngl <= layers + 1, "invalid fixed n_gpu_layers")
    allocations, placements = [], []
    raw = {"CPU": 0, "CUDA0": 0}
    model = {"CPU": 0, "CUDA0": 0}

    def add(tensor, device, role, duplicate=False):
        block, storage = LAYOUTS[tensor["type_id"]]
        # CUDA expands the LAST quantized row ONCE per tensor, not every row.
        tail_elements = (-tensor["dimensions"][0]) % 512
        tail = tail_elements // block * storage if device == "CUDA0" and block > 1 else 0
        allocated = align_up(tensor["byte_size"] + tail, 128 if device == "CUDA0" else 32)
        raw[device] += tensor["byte_size"]
        model[device] += allocated
        placements.append((tensor["name"], role, "GPU" if device == "CUDA0" else "CPU",
                           tensor["byte_size"], duplicate))
        allocations.append({"name": tensor["name"], "role": role, "device": device,
                            "is_duplicate": duplicate, "raw_payload_bytes": tensor["byte_size"],
                            "quantized_tail_padding_bytes": tail, "allocated_bytes": allocated})

    embeddings = None
    for tensor in directory["tensors"]:
        name = tensor["name"]
        if name == "token_embd.weight":
            embeddings = tensor
            add(tensor, "CPU", "input")
        elif name in ("output.weight", "output_norm.weight", "output.bias"):
            add(tensor, "CUDA0" if ngl else "CPU", "output")
        else:
            match = re.fullmatch(r"blk\.(\d+)\..+", name)
            require(match is not None and int(match[1]) < layers, "unsupported tensor role: " + name)
            add(tensor, "CUDA0" if int(match[1]) >= layers + 1 - ngl else "CPU", "block")
    require(embeddings is not None, "missing input embedding tensor")
    duplicated = embeddings["byte_size"] if directory["tied_output"] and ngl else 0
    if duplicated:
        add(embeddings, "CUDA0", "tied-output", True)
    require(sum(raw.values()) == directory["payload_bytes"] + duplicated,
            "independent raw placement conservation failure")
    allocated_context = align_up(context, 256)
    gpu_blocks = max(ngl - 1, 0)
    width = directory["kv_heads"] * directory["head_dim"]
    # Single-sequence FP16: K and V base tensors; view objects allocate nothing.
    one_cache = 2 * width * allocated_context
    kv = {"CPU": 2 * align_up(one_cache, 32) * (layers - gpu_blocks),
          "CUDA0": 2 * align_up(one_cache, 128) * gpu_blocks}
    layer_device = lambda il: "CUDA0" if ngl > 0 and il >= layers + 1 - ngl else "CPU"
    return {"raw": raw, "model": model, "kv": kv, "duplicated_bytes": duplicated,
            "allocated_context": allocated_context, "raw_kv_bytes": 4 * width * context * layers,
            "allocated_kv_bytes": 4 * width * allocated_context * layers,
            "gpu_blocks": gpu_blocks, "allocations": allocations, "placements": placements,
            "model_layer_devices": {i: layer_device(i) for i in range(layers + 1)},
            "kv_layer_devices": {i: layer_device(i) for i in range(layers)}}


def add_check(checks, name, actual, expected):
    checks.append({"check": name, "actual": actual, "expected": expected, "passed": actual == expected})


def calculator_json(command, env, timeout):
    result = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                            timeout=timeout, env=env)
    require(result.returncode == 0, "calculator exited " + str(result.returncode) + ": " + result.stderr.strip())
    require(not result.stderr.strip(), "calculator wrote unexpected stderr: " + result.stderr.strip())
    return strict_json(result.stdout)


def inspect_checks(calculator, path, directory, env):
    result = calculator_json([str(calculator), "inspect", str(path), "--json"], env, 30)
    checks = []
    add_check(checks, "inspect.command", result["command"], "inspect")
    add_check(checks, "inspect.complete", result["artifact"]["complete"], True)
    for field, expected in (("file_size", directory["file_size"]), ("data_offset", directory["data_offset"]),
                            ("total_tensor_bytes", directory["payload_bytes"])):
        add_check(checks, "inspect." + field, exact(result["artifact"][field], field), expected)
    records = {}
    for tensor in result["tensors"]:
        require(tensor["name"] not in records, "calculator returned duplicate tensor names")
        records[tensor["name"]] = {"name": tensor["name"], "type_id": tensor["type_id"],
                                  "dimensions": [exact(d, "dimension") for d in tensor["dimensions"]],
                                  **{key: exact(tensor[key], key) for key in
                                     ("byte_size", "element_count", "offset", "absolute_offset")}}
    add_check(checks, "inspect.independent_tensor_directory", records,
              {tensor["name"]: tensor for tensor in directory["tensors"]})
    # Keep report small on success, while retaining complete differences on failure.
    if checks[-1]["passed"]:
        checks[-1] = {"check": checks[-1]["check"], "tensor_count": len(records), "passed": True}
    return checks


def plan_checks(prediction, directory, expected, context, ngl):
    checks = []
    add_check(checks, "plan.command", prediction["command"], "plan")
    add_check(checks, "plan.requested_n_gpu_layers", prediction["input"]["n_gpu_layers"], ngl)
    add_check(checks, "plan.context", prediction["input"]["context_tokens"], context)
    add_check(checks, "plan.kv_precision", prediction["input"]["kv_precision"], "FP16")
    profile, plan = prediction["profile"], prediction["fit"]["plan"]
    require(plan is not None, "fixed calculator plan is infeasible under the selected capacity scenario")
    for field, wanted in (("n_gpu_layers", ngl), ("gpu_block_count", expected["gpu_blocks"]),
                          ("output_on_gpu", ngl > 0), ("allocated_context_tokens", expected["allocated_context"]),
                          ("kv_bytes", expected["raw_kv_bytes"]), ("allocated_kv_bytes", expected["allocated_kv_bytes"])):
        add_check(checks, "plan." + field, plan[field], wanted)
    add_check(checks, "profile.backend_revision", profile["backend_revision"], REVISION)
    add_check(checks, "profile.tied_output", profile["tied_output"], directory["tied_output"])
    add_check(checks, "profile.tensor_payload_bytes", exact(profile["tensor_payload_bytes"], "tensor_payload_bytes"), directory["payload_bytes"])
    for field, wanted in (("cpu_weight_bytes", expected["raw"]["CPU"]),
                          ("gpu_weight_bytes", expected["raw"]["CUDA0"]),
                          ("duplicated_bytes", expected["duplicated_bytes"])):
        add_check(checks, "plan." + field, exact(plan[field], field), wanted)
    for field, wanted in (("gpu_kv_bytes", expected["kv"]["CUDA0"]),
                          ("host_kv_bytes", expected["kv"]["CPU"]),
                          ("gpu_layers", expected["gpu_blocks"]),
                          ("cpu_layers", directory["layers"] - expected["gpu_blocks"])):
        add_check(checks, "ledger." + field, plan["ledger"][field], wanted)
    placements = [(p["name"], p["role"], p["device"], exact(p["byte_size"], "placement bytes"), p["is_duplicate"])
                  for p in plan["placements"]]
    add_check(checks, "plan.tensor_placements", sorted(placements), sorted(expected["placements"]))
    if checks[-1]["passed"]:
        checks[-1] = {"check": checks[-1]["check"], "placement_count": len(placements), "passed": True}
    wanted_status = "CPU ONLY" if ngl == 0 else "MAX_OFFLOAD" if ngl == directory["layers"] + 1 else "GPU+CPU OFFLOAD"
    add_check(checks, "plan.status", prediction["fit"]["status"], wanted_status)
    return checks


def parse_buffers(text):
    buffers = []
    factors = {"KiB": 1024, "MiB": MIB, "GiB": GIB}
    for line in text.splitlines():
        match = BUFFER_RE.search(line)
        if match:
            unit, number = match["unit"], Decimal(match["value"])
            places = len(match["value"].partition(".")[2])
            radius = Decimal(1).scaleb(-places) / 2 * factors[unit]
            buffers.append({"device": match["buffer"], "kind": match["kind"],
                            "reported_value": match["value"], "reported_unit": unit,
                            "rounded_bytes": float(number * factors[unit]),
                            "rounding_half_interval_bytes": float(radius)})
    return buffers


def backend_checks(raw, expected):
    raw = ANSI_RE.sub("", raw)
    checks = []
    model_matches, kv_matches = MODEL_LAYER_RE.findall(raw), KV_LAYER_RE.findall(raw)
    model_devices = {int(layer): device for layer, device, _ in model_matches}
    kv_devices = {int(layer): device for layer, device in kv_matches}
    add_check(checks, "backend.model_layer_devices", model_devices, expected["model_layer_devices"])
    add_check(checks, "backend.model_layer_log_count", len(model_matches), len(expected["model_layer_devices"]))
    add_check(checks, "backend.no_swa_layers", [swa for _, _, swa in model_matches], ["0"] * len(expected["model_layer_devices"]))
    add_check(checks, "backend.kv_layer_devices", kv_devices, expected["kv_layer_devices"])
    add_check(checks, "backend.kv_layer_log_count", len(kv_matches), len(expected["kv_layer_devices"]))
    fallbacks = [line for line in raw.splitlines() if FALLBACK_RE.search(line)]
    add_check(checks, "backend.no_fallback_or_allocation_error", fallbacks, [])
    buffers = parse_buffers(raw)
    unexpected = [b for b in buffers if b["kind"] in ("model", "KV") and b["device"] not in ("CPU", "CUDA0")]
    add_check(checks, "backend.only_plain_CPU_and_CUDA0_model_KV", unexpected, [])
    components = []
    for kind, values in (("model", expected["model"]), ("KV", expected["kv"])):
        for device, wanted in values.items():
            found = [b for b in buffers if b["kind"] == kind and b["device"] == device]
            if wanted == 0 and not found:
                passed, difference = True, None
                note = "no allocation/context expected; absent buffer line is the expected zero"
            elif len(found) == 1:
                difference = found[0]["rounded_bytes"] - wanted
                passed = abs(difference) <= found[0]["rounding_half_interval_bytes"] + 1
                note = "exact independent allocation is compared only with the printed rounding interval"
            else:
                passed, difference = False, None
                note = "expected one nonempty plain buffer; absent or duplicate logs are not a measurement"
            components.append({"component": kind, "device": device, "expected_allocated_bytes": wanted,
                               "matching_log_count": len(found), "difference_from_rounded_backend_bytes": difference,
                               "within_reported_rounding_interval": passed, "note": note})
    return checks, buffers, components


def smi_sample(executable, phase, started):
    fields = ["index", "uuid", "name", "driver_version", "memory.total", "memory.used", "memory.free", "utilization.gpu"]
    command = [executable, "-i", "0", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
    sample = {"phase": phase, "elapsed_seconds": round(time.monotonic() - started, 6)}
    try:
        result = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=5)
        omitted_utilization_reason = None
        if result.returncode:
            # Some WSL NVML implementations reject the utilization query rather
            # than returning N/A. Retry only the identity/memory fields needed
            # for the launch guard, retaining utilization as explicitly absent.
            omitted_utilization_reason = result.stderr.strip() or "combined utilization query failed"
            fields = fields[:-1]
            command[3] = "--query-gpu=" + ",".join(fields)
            result = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=5)
        sample["query_exit_code"] = result.returncode
        if result.returncode:
            sample.update({"values": None, "unavailable_reason": result.stderr.strip() or "nvidia-smi query failed"})
            return sample
        rows = list(csv.reader(io.StringIO(result.stdout.strip())))
        require(len(rows) == 1 and len(rows[0]) == len(fields), "unexpected nvidia-smi CSV shape")
        values, unavailable = {}, {}
        for field, value in zip(fields, rows[0]):
            value = value.strip()
            if value in ("", "N/A", "[N/A]", "[Not Supported]", "Not Supported", "Unknown Error"):
                values[field] = None
                unavailable[field] = "nvidia-smi returned " + (value or "an empty field")
            elif field in ("memory.total", "memory.used", "memory.free", "utilization.gpu", "index"):
                try:
                    number = float(value)
                    require(math.isfinite(number) and number >= 0, "invalid nvidia-smi numeric value")
                    values[field] = int(number) if number.is_integer() else number
                except ValueError:
                    values[field] = None
                    unavailable[field] = "unrecognized nvidia-smi numeric field: " + value
            else:
                values[field] = value
        if omitted_utilization_reason:
            values["utilization.gpu"] = None
            unavailable["utilization.gpu"] = omitted_utilization_reason
        sample.update({"values": values, "unavailable_fields": unavailable})
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        sample.update({"values": None, "unavailable_reason": str(error)})
    return sample


def available_gpu(sample):
    values = sample.get("values")
    require(values is not None and values.get("index") == 0, "cannot identify nvidia-smi GPU index 0")
    free, total, uuid = values.get("memory.free"), values.get("memory.total"), values.get("uuid")
    require(free is not None and total is not None and uuid and uuid.startswith("GPU-"),
            "GPU 0 free/total memory or UUID is unavailable; refusing inference")
    require(free >= MIN_FREE_MIB, "GPU 0 has less than 2 GiB free; refusing inference")
    return values


def run_backend(command, env, smi, timeout, interval, case_started):
    started = time.monotonic()
    samples = []
    timed_out = False
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   env=env, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                alive_start = process.poll() is None
                sample = smi_sample(smi, "during", case_started)
                sample["backend_alive_at_query_start"] = alive_start
                sample["backend_alive_at_query_end"] = process.poll() is None
                samples.append(sample)
                if process.poll() is None:
                    time.sleep(interval)
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        stdout.seek(0)
        stderr.seek(0)
        out, err = stdout.read().decode("utf-8", errors="replace"), stderr.read().decode("utf-8", errors="replace")
    return {"exit_code": process.returncode, "timed_out": timed_out,
            "elapsed_seconds": round(time.monotonic() - started, 6), "during_samples": samples}, out, err


def clean_environment():
    removed = sorted(key for key in os.environ if key.startswith(("LLAMA_ARG_", "GGML_"))
                     or key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"))
    return {key: value for key, value in os.environ.items() if key not in removed}, removed


def sanitize_text(text, paths):
    text = ANSI_RE.sub("", text)
    for path in sorted({str(p) for p in paths}, key=len, reverse=True):
        if path:
            label = "<home>" if path == str(Path.home()) else "<workspace>" if path == str(Path.cwd()) else Path(path).name
            text = text.replace(path, label)
    # Backend builds may log their source location, unrelated to input paths.
    text = re.sub(r"/(?:Users|home)/[^/\s\"'<>]+", "<home>", text)
    text = re.sub(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s\"'<>]+", "<home>", text)
    text = re.sub(r"\bGPU-[0-9a-fA-F-]{32,}\b",
                  lambda match: "GPU-UUID-SHA256:" + hashlib.sha256(match[0].encode()).hexdigest(), text)
    return text


def sanitize(value, paths):
    if isinstance(value, str):
        return sanitize_text(value, paths)
    if isinstance(value, dict):
        return {"uuid_sha256" if key == "uuid" and isinstance(item, str) else key:
                hashlib.sha256(item.encode()).hexdigest() if key == "uuid" and isinstance(item, str)
                else sanitize(item, paths) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item, paths) for item in value]
    return value


def backend_command(backend, path, context, ngl):
    return [str(backend), "-m", str(path), "-c", str(context), "-n", "32", "-t", "2",
            "-b", "64", "-ub", "64", "-np", "1", "-ngl", str(ngl), "--device", "CUDA0",
            "--split-mode", "none", "--fit", "off", "--no-repack", "--no-host",
            "--load-mode", "none", "--lazy-mode", "off", "--kv-offload", "-fa", "off",
            "-ctk", "f16", "-ctv", "f16", "-no-cnv", "-p", "Once upon a time",
            "-s", "1", "--temp", "0", "-lv", "5", "--log-colors", "off", "--no-display-prompt"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calculator", required=True, type=Path)
    parser.add_argument("--backend", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="official stories15M-q4_0.gguf at the fixed source SHA256")
    parser.add_argument("--tied-model", type=Path, help="optional derived stories260K fixture without output.weight")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=90, help="per inference process timeout in seconds")
    parser.add_argument("--sample-interval", type=float, default=0.1, help="pause between device queries in seconds")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    args = parser.parse_args()
    if args.timeout < 1 or not math.isfinite(args.sample_interval) or not 0.02 <= args.sample_interval <= 5:
        parser.error("positive timeout and sample interval within 0.02..5 seconds are required")
    # This guard is deliberately outside the report-writing exception handler.
    # A rejected target must never receive even an error report, and no child
    # process (including backend --version) may precede this check.
    try:
        calculator, backend, model = args.calculator.resolve(), args.backend.resolve(), args.model.resolve()
        tied = args.tied_model.resolve() if args.tied_model else None
        smi = shutil.which(args.nvidia_smi)
        inputs_to_protect = [calculator, backend, model, Path(args.nvidia_smi).resolve()]
        if tied:
            inputs_to_protect.append(tied)
        if smi:
            inputs_to_protect.append(Path(smi).resolve())
        cases = acceptance_cases(model, tied)
        output, case_logs = validate_output_targets(args.output, inputs_to_protect, cases)
    except (OSError, ValueError, RuntimeError) as error:
        print("CUDA validation output refused: " + sanitize_text(str(error), [Path.home(), Path.cwd()]), file=sys.stderr)
        return 1
    private_paths = [calculator, backend, model, output, Path.home(), Path.cwd()]
    if tied:
        private_paths.append(tied)
    env, removed = clean_environment()
    report = {
        "schema_version": "1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend_revision": REVISION, "status": "preflight", "all_passed": False,
        "expected_run_count": 7 if tied else 5, "runs": [],
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                     "python_version": sys.version.split()[0], "WSL_detected": "microsoft" in platform.release().lower()},
        "scope": "single CUDA0, fixed output-first ngl, plain CPU/CUDA buffers, no repack/host/mmap/lazy, full attention, FP16 KV, one sequence",
        "measurement_boundary": {
            "validated": "independent GGUF tensor directory/raw placement, per-layer model/KV device assignments, and model/KV requested buffer bytes versus printed rounding intervals",
            "not_validated": "driver physical allocation granularity, process VRAM peak, inference quality, throughput, capacity-policy calibration, general GPU/model compatibility",
            "device_sampling": "nvidia-smi GPU-index-0 total device memory before/during/after includes other processes, driver and CUDA context; sampled maximum is not a process peak",
            "unsupported_telemetry": "WSL/WDDM unavailable fields are null with a reason; absent measurements are never numeric zero or marked passed",
            "planning_RAM": "Linux sysconf visible RAM is used as capacity; on WSL this is the VM limit, not Windows physical RAM or currently free memory",
        },
        "allocation_oracle": {
            "CPU_model": "sum(align_up(raw_tensor_bytes,32)) over unique actual CPU tensor objects",
            "CUDA_model": "sum(align_up(raw_tensor_bytes + quant_tail,128)); quant_tail=(((-ne0)%512)/block_elements)*block_bytes for quantized tensors, otherwise 0; added once per tensor",
            "FP16_KV": "S=align_up(requested_context,256); one K or V tensor=2*kv_heads*head_dim*S; sum both tensors per layer aligned to its CPU32/CUDA128 buffer; output slot has no KV",
            "tied_weights": "CPU alias shares original allocation when ngl=0; when output offloads, original CPU embedding remains and one independently padded CUDA duplicate is allocated",
            "rounding": "reported decimal buffer sizes define a half-last-digit interval, plus one byte numerical slack; no broad percentage tolerance",
            "upstream_INFO_caveat": "pinned max-offload repeating-layer INFO count underreports by one; acceptance uses DEBUG per-layer assignments, not that summary",
            "source_links": [
                "https://github.com/ggml-org/llama.cpp/blob/" + REVISION + "/ggml/src/ggml-backend.cpp#L57-L88",
                "https://github.com/ggml-org/llama.cpp/blob/" + REVISION + "/ggml/src/ggml-cuda/ggml-cuda.cu#L942-L965",
                "https://github.com/ggml-org/llama.cpp/blob/" + REVISION + "/src/llama-model.cpp#L1653-L1684",
                "https://github.com/ggml-org/llama.cpp/blob/" + REVISION + "/src/llama-kv-cache.cpp#L210-L291",
                "https://github.com/ggml-org/llama.cpp/blob/" + REVISION + "/src/llama-model-loader.cpp#L1369-L1388",
            ],
        },
        "removed_environment_variable_names": removed,
    }
    exit_code = 1
    try:
        require(platform.system() == "Linux", "this GPU collector supports Linux and WSL2 only")
        ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        require(ram_bytes > 0, "Linux visible RAM could not be determined")
        ram_capacity = format(ram_bytes / GIB, ".9f") + "GiB"
        report["host_memory"] = {"visible_total_bytes": ram_bytes,
                                 "source": "sysconf SC_PAGE_SIZE * SC_PHYS_PAGES",
                                 "planning_capacity": ram_capacity}
        require(calculator.is_file() and os.access(calculator, os.X_OK), "calculator executable missing")
        require(backend.is_file() and os.access(backend, os.X_OK), "backend executable missing")
        require(model.is_file() and model.stat().st_size == MODEL_SIZE and sha256(model) == MODEL_SHA256,
                "main model differs from the fixed stories15M-q4_0 source artifact")
        tied_directory = verify_tied_fixture(tied) if tied else None
        report["calculator"] = {"binary": calculator.name, "sha256": sha256(calculator)}
        report["backend"] = {"binary": backend.name, "sha256": sha256(backend)}
        version_run = subprocess.run([str(backend), "--version"], capture_output=True, text=True, timeout=15,
                                     stdin=subprocess.DEVNULL, env=env)
        version = (version_run.stdout + version_run.stderr).strip()
        require(version_run.returncode == 0 and REVISION in version, "backend --version does not identify the full fixed source revision")
        report["backend"]["version"] = version
        require(smi is not None, "nvidia-smi executable unavailable; refusing inference")
        preflight = smi_sample(smi, "preflight", time.monotonic())
        report["gpu_preflight"] = preflight
        gpu = available_gpu(preflight)
        # UUID maps physical nvidia-smi index 0 to the backend's only logical CUDA0.
        env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
        report["device_selection"] = {
            "nvidia_smi_index": 0, "uuid_sha256": hashlib.sha256(gpu["uuid"].encode()).hexdigest(),
            "binding": "CUDA_VISIBLE_DEVICES is set to the physical GPU 0 UUID; the public identifier is hashed",
            "backend_logical_device": "CUDA0",
        }
        devices_run = subprocess.run([str(backend), "--list-devices"], capture_output=True, text=True,
                                     stdin=subprocess.DEVNULL, timeout=15, env=env)
        devices = devices_run.stdout + devices_run.stderr
        require(devices_run.returncode == 0 and re.search(r"\bCUDA0\b", devices), "selected backend does not enumerate CUDA0")
        report["backend"]["list_devices"] = devices.strip()
        inputs = [(model, False)] + ([(tied, True)] if tied else [])
        directory_cache = {}
        report["models"] = []
        for path, is_tied in inputs:
            directory = tied_directory if is_tied else read_directory(path)
            require(directory["layers"] == (5 if is_tied else 6) and directory["tied_output"] == is_tied,
                    "model does not match the intended untied/tied acceptance fixture")
            directory_cache[path] = directory
            model_report = {"model": path.name, "sha256": sha256(path), "file_size": directory["file_size"],
                            "is_derived_tied_fixture": is_tied, "tensor_count": len(directory["tensors"]),
                            "tensor_payload_bytes": directory["payload_bytes"],
                            "independent_inspection_checks": inspect_checks(calculator, path, directory, env)}
            report["models"].append(model_report)
            require(all(c["passed"] for c in model_report["independent_inspection_checks"]), "calculator inspect differs from the independent directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        for case_index, (path, context, ngl) in enumerate(cases, 1):
            case = {"case": case_index, "model": path.name, "context_requested": context,
                    "n_gpu_layers": ngl, "passed": False, "status": "preflight"}
            report["runs"].append(case)
            started = time.monotonic()
            before = smi_sample(smi, "before", started)
            case["gpu_before"] = before
            current_gpu = available_gpu(before)
            require(current_gpu["uuid"] == gpu["uuid"], "GPU index 0 changed between runs")
            expected = oracle(directory_cache[path], context, ngl)
            vram = format(current_gpu["memory.total"] / 1024, ".9f") + "GiB"
            plan_command = [str(calculator), "plan", str(path), "--vram=" + vram, "--ram=" + ram_capacity,
                            "--context=" + str(context), "--kv=FP16", "--n-gpu-layers=" + str(ngl), "--json"]
            prediction = calculator_json(plan_command, env, 30)
            checks = plan_checks(prediction, directory_cache[path], expected, context, ngl)
            case["calculator_checks"] = checks
            require(all(c["passed"] for c in checks), "calculator fixed plan differs from independent raw placement oracle")
            launch_guard = smi_sample(smi, "immediately_before_launch", started)
            case["gpu_launch_guard"] = launch_guard
            require(available_gpu(launch_guard)["uuid"] == gpu["uuid"], "GPU index 0 changed before launch")
            command = backend_command(backend, path, context, ngl)
            observed, stdout, stderr = run_backend(command, env, smi, args.timeout, args.sample_interval, started)
            case["observed_process"] = {key: value for key, value in observed.items() if key != "during_samples"}
            stderr_path, stdout_path = case_logs[case_index - 1]
            write_new_text(stderr_path, sanitize_text(stderr, private_paths))
            write_new_text(stdout_path, sanitize_text(stdout, private_paths))
            case["logs"] = {"stderr": stderr_path.name, "stdout": stdout_path.name}
            case["command_template"] = [backend.name if i == 0 else path.name if v == str(path) else v for i, v in enumerate(command)]
            case["calculator_command_template"] = [calculator.name if i == 0 else path.name if v == str(path) else v for i, v in enumerate(plan_command)]
            backend_check, logged, components = backend_checks(stderr + "\n" + stdout, expected)
            add_check(backend_check, "backend.exit_code", observed["exit_code"], 0)
            add_check(backend_check, "backend.timed_out", observed["timed_out"], False)
            case["backend_checks"], case["backend_buffers"], case["component_comparisons"] = backend_check, logged, components
            case["independent_oracle"] = {key: value for key, value in expected.items() if key != "placements"}
            after = smi_sample(smi, "after", started)
            samples = observed["during_samples"]
            valid = [sample for sample in samples if sample.get("backend_alive_at_query_start")
                     and sample.get("backend_alive_at_query_end") and sample.get("values")
                     and sample["values"].get("memory.used") is not None]
            used = [sample["values"]["memory.used"] for sample in valid]
            case["device_telemetry"] = {
                "before": before, "during": samples, "after": after,
                "in_run_memory_sample_status": "observed" if valid else "unavailable",
                "unavailable_reason": None if valid else "no valid memory.used query completed while the backend remained alive",
                "in_run_memory_sample_count": len(valid),
                "sampled_device_used_max_mib": max(used) if used else None,
                "sampled_device_used_last_mib": used[-1] if used else None,
                "process_vram_peak_bytes": None,
                "process_vram_peak_unavailable_reason": "nvidia-smi device totals do not establish a process peak; WSL/WDDM process accounting is not assumed",
            }
            add_check(backend_check, "device.in_run_memory_sample_observed", bool(valid), True)
            case["passed"] = all(c["passed"] for c in checks + backend_check) and all(c["within_reported_rounding_interval"] for c in components)
            case["status"] = "passed" if case["passed"] else "failed"
            print(f"CUDA case {case_index}/{len(cases)}: ctx={context} ngl={ngl} " + case["status"], flush=True)
            if observed["exit_code"] != 0 or observed["timed_out"]:
                break
        for model_report, (path, _) in zip(report["models"], inputs):
            model_report["unchanged_after_runs"] = sha256(path) == model_report["sha256"]
        report["backend"]["unchanged_after_runs"] = sha256(backend) == report["backend"]["sha256"]
        report["calculator"]["unchanged_after_runs"] = sha256(calculator) == report["calculator"]["sha256"]
        report["all_passed"] = (len(report["runs"]) == report["expected_run_count"]
                                and all(case["passed"] for case in report["runs"])
                                and all(m["unchanged_after_runs"] for m in report["models"])
                                and report["backend"]["unchanged_after_runs"]
                                and report["calculator"]["unchanged_after_runs"])
        report["status"] = "passed" if report["all_passed"] else "failed"
        exit_code = 0 if report["all_passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
        report["status"] = "failed" if report["runs"] else "preflight_refused"
        if report["runs"] and report["runs"][-1]["status"] == "preflight":
            report["runs"][-1].update({"status": "refused_or_failed", "error": str(error)})
        print("CUDA backend validation failed: " + sanitize_text(str(error), private_paths), file=sys.stderr)
    safe_report = sanitize(report, private_paths)
    encoded = json.dumps(safe_report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_new_text(output, encoded)
    except OSError as error:
        print("CUDA validation report not written: " + sanitize_text(str(error), private_paths), file=sys.stderr)
        return 1
    print(f"CUDA allocation comparison: {sum(case['passed'] for case in report['runs'])}/{report['expected_run_count']} passed; status={report['status']}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
