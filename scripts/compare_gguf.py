#!/usr/bin/env python3
"""Compare Native inspection with the pinned official gguf-py GGUFReader.

The driver uses stdlib; the selected upstream package requires NumPy and its
normal import dependencies. No dependency is installed by this script. Only
the explicitly supplied --gguf-python directory is inserted into sys.path.
Reports contain model basenames and hashes, never local absolute paths.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys


REVISION = "88aaf325a7026ac1127d0897394b0a3d6413dc2a"
PINNED_FILES = {
    "gguf_reader.py": "d0ea743200e19d7ef0a4c969edcc4a6dba8e3656a790b217e3bdc7b31170718e",
    "constants.py": "3e6108928a33ac8d0c5c836d1854207f9137cd2b21825b7613c3705d29c5f183",
    "quants.py": "2c927a1b3d9f0920dcf4007fb686e1b0999333e9f65ce43dcc689900c0beae8b",
    "lazy.py": "dbc98e3ee9ef8606df34e9d91f98ab29c2697e6824995f1dd8952938c135ad85",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(text):
    def reject_constant(value):
        raise ValueError("nonstandard JSON numeric constant: " + value)
    return json.loads(text, parse_constant=reject_constant)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def raw_integer(value, label):
    require(isinstance(value, str) and value.isdigit(), label + " must be a decimal JSON string")
    return int(value)


def load_reader(directory):
    files = {name: sha256(directory / "gguf" / name) for name in PINNED_FILES}
    require(files == PINNED_FILES, "gguf-py reader/dependency source hashes differ from the fixed upstream revision")
    numpy = importlib.import_module("numpy")
    sys.path.insert(0, str(directory))
    module = importlib.import_module("gguf.gguf_reader")
    require(Path(module.__file__).resolve() == directory / "gguf" / "gguf_reader.py", "GGUFReader was imported from a different package")
    return module.GGUFReader, numpy.__version__, files


def difference(differences, field, calculator, official):
    if calculator != official:
        differences.append({"field": field, "calculator": calculator, "official": official})


def selected_keys(architecture):
    keys = {"general.architecture", "general.name", "general.file_type", "general.alignment"}
    for suffix in (
        "block_count", "context_length", "embedding_length", "feed_forward_length",
        "attention.head_count", "attention.head_count_kv", "attention.key_length",
        "attention.value_length", "attention.causal", "rope.dimension_count", "rope.freq_base",
    ):
        keys.add(architecture + "." + suffix)
    return sorted(keys)


def reference_metadata(field):
    type_id = int(field.types[0])
    if type_id == 9:
        return {"kind": "array", "total_count": str(int(field.parts[4][0])), "element_type_id": int(field.parts[3][0])}
    contents = field.contents()
    if type_id in (0, 2, 4, 10):
        return {"kind": "unsigned_integer", "value": str(int(contents))}
    if type_id in (1, 3, 5, 11):
        return {"kind": "signed_integer", "value": str(int(contents))}
    if type_id == 8:
        return {"kind": "string", "value": str(contents)}
    if type_id == 7:
        return {"kind": "boolean", "value": bool(contents)}
    if type_id in (6, 12):
        number = float(contents)
        if not math.isfinite(number):
            number = "NaN" if math.isnan(number) else ("Infinity" if number > 0 else "-Infinity")
        return {"kind": "float", "value": number}
    raise ValueError("unexpected selected metadata type " + str(type_id))


def compare_model(calculator, path, reader_class):
    input_hash = sha256(path)
    size = path.stat().st_size
    result = subprocess.run([str(calculator), "inspect", str(path), "--json"], capture_output=True, text=True, timeout=60)
    require(result.returncode == 0, f"calculator inspect exited {result.returncode}: {result.stderr.strip()}")
    require(not result.stderr, "calculator wrote unexpected stderr: " + result.stderr.strip())
    calculated = strict_json(result.stdout)
    require(calculated["command"] == "inspect", "calculator returned a different command")
    require(calculated["artifact"]["complete"], "calculator tensor inventory is incomplete")
    reader = reader_class(path)
    differences = []
    artifact = calculated["artifact"]
    version = int(reader.get_field("GGUF.version").contents())
    difference(differences, "gguf_version", artifact["gguf_version"], version)
    difference(differences, "file_size", raw_integer(artifact["file_size"], "file_size"), size)
    difference(differences, "mapped_file_size", int(reader.data.nbytes), size)
    difference(differences, "data_offset", raw_integer(artifact["data_offset"], "data_offset"), int(reader.data_offset))
    difference(differences, "tensor_count", artifact["tensor_count"], len(reader.tensors))
    difference(differences, "metadata_count", artifact["metadata_count"], int(reader.get_field("GGUF.kv_count").contents()))
    by_name = {tensor["name"]: tensor for tensor in calculated["tensors"]}
    require(len(by_name) == len(calculated["tensors"]), "calculator returned duplicate tensor names")
    reference_names = {tensor.name for tensor in reader.tensors}
    difference(differences, "tensor_names", sorted(by_name), sorted(reference_names))
    records = []
    by_type = {}
    total_bytes = total_elements = 0
    for tensor in reader.tensors:
        type_id = int(tensor.tensor_type)
        shape = [str(int(dimension)) for dimension in tensor.shape.tolist()]
        # ReaderTensor.shape is the original GGML directory order. Its data.shape
        # is reversed and quantized data additionally has a packed byte shape.
        relative_offset = int(tensor.field.parts[-1][0])
        reference = {
            "ggml_dimensions": shape, "type_id": type_id,
            "element_count": str(int(tensor.n_elements)), "byte_size": str(int(tensor.n_bytes)),
            "relative_offset": str(relative_offset), "absolute_offset": str(int(tensor.data_offset)),
        }
        found = by_name.get(tensor.name)
        before = len(differences)
        if found is not None:
            actual = {
                "ggml_dimensions": found["dimensions"], "type_id": found["type_id"],
                "element_count": found["element_count"], "byte_size": found["byte_size"],
                "relative_offset": found["offset"], "absolute_offset": found["absolute_offset"],
            }
            for field, expected in reference.items():
                difference(differences, "tensor." + tensor.name + "." + field, actual[field], expected)
            for field in ("element_count", "byte_size", "relative_offset", "absolute_offset"):
                raw_integer(actual[field], "tensor." + tensor.name + "." + field)
            require(all(isinstance(dim, str) and dim.isdigit() for dim in actual["ggml_dimensions"]), "tensor dimensions must be exact decimal strings")
        records.append({"name": tensor.name, **reference, "numpy_payload_shape": [int(dim) for dim in tensor.data.shape], "match": found is not None and len(differences) == before})
        total_bytes += int(tensor.n_bytes)
        total_elements += int(tensor.n_elements)
        group = by_type.setdefault(str(type_id), {"type_name": tensor.tensor_type.name, "tensor_count": 0, "payload_bytes": 0, "element_count": 0})
        group["tensor_count"] += 1
        group["payload_bytes"] += int(tensor.n_bytes)
        group["element_count"] += int(tensor.n_elements)
    for field in ("total_tensor_bytes", "known_tensor_bytes"):
        difference(differences, field, raw_integer(artifact[field], field), total_bytes)
    difference(differences, "sum_calculator_tensor_bytes", sum(raw_integer(t["byte_size"], "tensor.byte_size") for t in calculated["tensors"]), total_bytes)
    difference(differences, "sum_calculator_tensor_elements", sum(raw_integer(t["element_count"], "tensor.element_count") for t in calculated["tensors"]), total_elements)
    metadata = {entry["key"]: entry["value"] for entry in calculated["metadata"]}
    architecture_field = reader.get_field("general.architecture")
    architecture = str(architecture_field.contents()) if architecture_field else ""
    compared_metadata = {}
    for key in selected_keys(architecture):
        field = reader.get_field(key)
        actual = metadata.get(key)
        if field is None and actual is None:
            continue
        expected = reference_metadata(field) if field is not None else None
        comparison = None if actual is None else {name: actual.get(name) for name in expected or ()}
        difference(differences, "metadata." + key, comparison, expected)
        compared_metadata[key] = {"official": expected, "calculator": comparison, "match": comparison == expected}
    require(sha256(path) == input_hash, "model contents changed during comparison")
    for group in by_type.values():
        group["payload_bytes"] = str(group["payload_bytes"])
        group["element_count"] = str(group["element_count"])
    return {
        "model": path.name, "input_sha256": input_hash, "file_size": str(size),
        "gguf_version": version, "data_offset": str(int(reader.data_offset)),
        "summary": {"tensor_count": len(reader.tensors), "stored_tensor_elements": str(total_elements), "tensor_payload_bytes": str(total_bytes), "by_type": by_type},
        "selected_metadata": compared_metadata, "tensors": records,
        "calculator_io": artifact["io"], "match": not differences, "differences": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calculator", type=Path, required=True)
    parser.add_argument("--gguf-python", type=Path, required=True, help="explicit upstream gguf-py directory containing gguf/gguf_reader.py")
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calculator, upstream = args.calculator.resolve(), args.gguf_python.resolve()
    paths = [path.resolve() for path in args.model]
    private_paths = [calculator, upstream, *paths]
    def safe_message(error):
        message = str(error)
        for path in private_paths:
            message = message.replace(str(path), path.name)
        return message
    try:
        require(calculator.is_file(), "calculator executable is missing")
        reader_class, numpy_version, hashes = load_reader(upstream)
    except (ImportError, ValueError, OSError) as error:
        print("Comparison dependency/source check failed: " + safe_message(error), file=sys.stderr)
        return 2
    models = []
    for path in paths:
        try:
            models.append(compare_model(calculator, path, reader_class))
        except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
            models.append({"model": path.name, "match": False, "error": safe_message(error)})
    report = {
        "schema_version": "1", "comparison": "Native inspect versus official GGUFReader",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream": {"repository": "https://github.com/ggml-org/llama.cpp", "revision": REVISION, "reader_source_sha256": hashes["gguf_reader.py"], "source_files_sha256": hashes},
        "runtime": {"python_version": sys.version.split()[0], "numpy_version": numpy_version},
        "calculator": {"binary": calculator.name, "sha256": sha256(calculator)},
        "shape_convention": "GGML dimensions use ReaderTensor.shape; NumPy payload data.shape is reversed and may be quantization-byte-packed",
        "scope": "tensor directory and selected scalar structure metadata comparison; not tensor-content validation, inference accuracy, performance, or universal GGUF compatibility",
        "models": models, "all_match": all(model["match"] for model in models),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for path in private_paths:
        require(str(path) not in encoded, "report unexpectedly contains an absolute input path")
    require("/Users/" not in encoded and "\\Users\\" not in encoded, "report unexpectedly contains a user home path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    for model in models:
        if not model["match"]:
            print("Mismatch: " + model["model"] + ": " + json.dumps(model.get("differences", model.get("error")), ensure_ascii=False), file=sys.stderr)
    print(f"GGUF comparison: {sum(model['match'] for model in models)}/{len(models)} models matched; {sum(len(model.get('tensors', [])) for model in models)} tensors compared")
    return 0 if report["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
