#!/usr/bin/env python3
"""Derive a clearly labeled tied-output validation fixture from pinned stories260K.

This is a structural test artifact, not a newly trained model or a quality claim.
The source weights are not redistributed; the pinned official GGUF writer copies
every retained tensor unchanged and removes only output.weight.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

from compare_gguf import load_reader, require, sha256, REVISION

SOURCE_SHA256 = "047bf46455a544931cff6fef14d7910154c56afbc23ab1c5e56a72e69912c04b"
WRITER_SHA256 = "123158b033180f70f85638a0c654f6c2f269c7ada9238de37dd678e3e2627423"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--gguf-python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    require(len({args.source.resolve(), args.output.resolve(), args.manifest.resolve()}) == 3,
            "source, output and manifest must be distinct paths")
    require(not args.output.exists() and not args.output.is_symlink(),
            "output already exists; use a new destination")
    require(not args.manifest.exists() and not args.manifest.is_symlink(),
            "manifest already exists; use a new destination")
    require(sha256(args.source) == SOURCE_SHA256, "source must be the fixed official stories260K.gguf")
    require(sha256(args.gguf_python / "gguf/gguf_writer.py") == WRITER_SHA256,
            "GGUFWriter source differs from the pinned upstream revision")
    reader_class, numpy_version, reader_hashes = load_reader(args.gguf_python.resolve())
    writer_module = importlib.import_module("gguf.gguf_writer")
    require(Path(writer_module.__file__).resolve() == args.gguf_python.resolve() / "gguf/gguf_writer.py",
            "GGUFWriter was imported from a different package")
    reader = reader_class(args.source)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    require(len(tensors) == 48, "unexpected source tensor inventory")
    require("output.weight" in tensors and "token_embd.weight" in tensors, "source must have untied output")
    require(tensors["output.weight"].shape.tolist() == tensors["token_embd.weight"].shape.tolist(),
            "embedding/output shapes do not match")
    require(int(tensors["token_embd.weight"].tensor_type) == 0, "this fixture requires F32 embedding")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = writer_module.GGUFWriter(args.output, "llama")
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key in ("general.architecture", "general.name", "general.description"):
            continue
        writer.add_key_value(key, field.contents(), field.types[0],
                             field.types[1] if len(field.types) > 1 else None)
    writer.add_name("stories260K-derived-tied-validation")
    writer.add_description("Derived structural fixture: output.weight removed; input embedding reused as output. Not a trained tied model; generation quality is not evaluated.")
    for tensor in reader.tensors:
        if tensor.name != "output.weight":
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    derived = reader_class(args.output)
    require(len(derived.tensors) == 47, "derived tensor count mismatch")
    require(not any(t.name == "output.weight" for t in derived.tensors), "output tensor was not removed")
    for tensor in derived.tensors:
        original = tensors[tensor.name]
        require(tensor.shape.tolist() == original.shape.tolist() and tensor.tensor_type == original.tensor_type,
                "retained descriptor changed: " + tensor.name)
        require(tensor.data.tobytes() == original.data.tobytes(), "retained payload changed: " + tensor.name)
    report = {
        "schema_version": "1", "purpose": "derived tied-output structural validation fixture; not an official or trained tied model",
        "source": {"filename": args.source.name, "sha256": SOURCE_SHA256, "tensors": 48},
        "derivation": "official GGUFWriter copies retained tensors byte-for-byte, removes output.weight, and labels general.name/description",
        "reference_revision": REVISION, "reference_reader_sha256": reader_hashes,
        "reference_writer_sha256": sha256(args.gguf_python / "gguf/gguf_writer.py"),
        "numpy_version": numpy_version,
        "derived": {"filename": args.output.name, "sha256": sha256(args.output),
                    "file_size": args.output.stat().st_size, "tensors": len(derived.tensors),
                    "tensor_payload_bytes": sum(int(t.n_bytes) for t in derived.tensors)},
        "all_retained_tensor_payloads_unchanged": True,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x") as manifest:
        manifest.write(json.dumps(report, indent=2) + "\n")
    print("Derived tied fixture: 47 retained tensors, all retained payloads unchanged")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print("fixture derivation failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)
