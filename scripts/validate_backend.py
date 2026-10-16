#!/usr/bin/env python3
"""Reconcile tensor/KV predictions with a pinned CPU backend, not GPU accuracy.

Uses only the Python standard library. wait4 records per-child OS peak RSS;
periodic ps samples are separately labeled and are not called steady-state RSS.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

REVISION = "88aaf325a7026ac1127d0897394b0a3d6413dc2a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(text: str):
    def invalid(value):
        raise ValueError("nonstandard JSON number: " + value)
    return json.loads(text, parse_constant=invalid)


def run_sampled(command: list[str], logfile: Path, timeout: int) -> dict:
    samples = []
    sampling_error = None
    started = time.monotonic()
    with logfile.open("wb") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr,
                                   stdin=subprocess.DEVNULL)
        try:
            while True:
                child, status, usage = os.wait4(process.pid, os.WNOHANG)
                if child:
                    process.returncode = os.waitstatus_to_exitcode(status)
                    break
                if time.monotonic() - started > timeout:
                    process.kill()
                    _, status, usage = os.wait4(process.pid, 0)
                    process.returncode = os.waitstatus_to_exitcode(status)
                    raise TimeoutError("backend exceeded timeout")
                if sampling_error is None:
                    try:
                        observed = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)],
                                                  capture_output=True, text=True, timeout=5)
                        if observed.returncode == 0 and observed.stdout.strip().isdigit():
                            samples.append(int(observed.stdout.strip()) * 1024)
                    except OSError as error:
                        sampling_error = f"ps sampling unavailable (OS error {error.errno}); wait4 peak remains available"
                time.sleep(0.01)
        except BaseException:
            if process.returncode is None:
                process.kill()
                process.wait()
            raise
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    peak = int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    return {
        "exit_code": process.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "os_peak_rss_bytes": peak,
        "rss_sample_count": len(samples),
        "sampled_peak_rss_bytes": max(samples) if samples else None,
        "last_sampled_rss_bytes": samples[-1] if samples else None,
        "rss_sampling_error": sampling_error,
        "rss_note": "wait4 OS process high-water RSS; samples are not a separately identified steady-state phase",
    }


def buffers(text: str) -> list[dict]:
    values = []
    pattern = re.compile(r"\b(CPU\w*)\s+(model|KV|compute|output) buffer size\s*=\s*([0-9.]+)\s+(KiB|MiB|GiB)")
    factors = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            device, kind, value, unit = match.groups()
            # Backend decimal output is rounded to two places, not exact bytes.
            values.append({"device": device, "kind": kind, "reported_value": float(value),
                           "reported_unit": unit, "rounded_bytes": float(value) * factors[unit],
                           "rounding_half_interval_bytes": 0.005 * factors[unit]})
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calculator", required=True, type=Path)
    parser.add_argument("--backend", required=True, type=Path)
    parser.add_argument("--model", required=True, action="append", type=Path)
    parser.add_argument("--context", action="append", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    if not hasattr(os, "wait4") or platform.system() not in ("Darwin", "Linux"):
        parser.error("this RSS collector supports macOS and Linux only")
    if args.timeout < 1 or any(c < 1 or c > 1048576 for c in (args.context or [])):
        parser.error("positive timeout and contexts within 1..1048576 are required")
    inputs = [args.calculator, args.backend, *args.model]
    if args.manifest:
        inputs.append(args.manifest)
    destinations = [args.output] + [
        args.output.parent / (model.stem + "-ctx" + str(context) + ".log")
        for model in args.model for context in args.context or [128, 512]
    ]
    for destination in destinations:
        for source in inputs:
            if (destination.resolve() == source.resolve()
                    or destination.exists() and source.exists() and destination.samefile(source)):
                raise ValueError("validation output would overwrite an input: " + destination.name)
    version_run = subprocess.run([str(args.backend), "--version"], capture_output=True,
                                 text=True, check=True, timeout=15)
    version = (version_run.stdout + version_run.stderr).strip()
    if REVISION not in version:
        raise ValueError("backend version does not identify the pinned source revision " + REVISION)
    manifest = strict_json(args.manifest.read_text()) if args.manifest else None
    expected = {m["filename"]: m for m in manifest["models"]} if manifest else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backend_hash = sha256(args.backend)
    calculator_hash = sha256(args.calculator)
    results = []
    for model in args.model:
        model_hash = sha256(model)
        if manifest:
            entry = expected.get(model.name)
            if not entry or entry["sha256"] != model_hash or entry["file_size"] != model.stat().st_size:
                raise ValueError("model does not match the source manifest: " + model.name)
        for context in args.context or [128, 512]:
            prediction = strict_json(subprocess.run([
                str(args.calculator), "plan", str(model), "--vram=0GiB", "--ram=64GiB",
                "--context=" + str(context), "--json"], check=True, capture_output=True,
                text=True, timeout=30).stdout)
            plan = prediction["fit"]["plan"]
            if plan is None or plan["n_gpu_layers"] != 0:
                raise ValueError("the CPU-only prediction is unavailable")
            logfile = args.output.parent / (model.stem + "-ctx" + str(context) + ".log")
            command = [str(args.backend), "-m", str(model), "-c", str(context),
                       "-n", "32", "-t", "2", "-b", "64", "-ub", "64", "-ngl", "0",
                       "--device", "none", "--fit", "off", "--no-repack", "--load-mode", "none",
                       "--lazy-mode", "off", "-fa", "off", "-ctk", "f16", "-ctv", "f16",
                       "-no-cnv", "-p", "Once upon a time", "-lv", "4", "--no-display-prompt"]
            observed = run_sampled(command, logfile, args.timeout)
            raw = logfile.read_text(errors="replace")
            sanitized = raw.replace(str(model), model.name).replace(str(args.backend), args.backend.name)
            logfile.write_text("\n".join(line.rstrip() for line in sanitized.splitlines()) + "\n")
            logged = buffers(raw)
            weight = int(plan["cpu_weight_bytes"])
            allocated_kv = plan["allocated_kv_bytes"]
            checks = []
            for kind, predicted in [("model", weight), ("KV", allocated_kv)]:
                found = [v for v in logged if v["kind"] == kind]
                if len(found) == 1:
                    difference = found[0]["rounded_bytes"] - predicted
                    within_rounding = abs(difference) <= found[0]["rounding_half_interval_bytes"] + 1
                else:
                    difference, within_rounding = None, False
                checks.append({"component": kind, "prediction_bytes": predicted,
                               "difference_from_rounded_backend_bytes": difference,
                               "within_reported_rounding_interval": within_rounding})
            result = {
                "model": model.name, "model_sha256": model_hash, "file_size": model.stat().st_size,
                "context_requested": context, "context_allocated": plan["allocated_context_tokens"],
                "raw_kv_payload_bytes": plan["kv_bytes"], "allocated_kv_payload_bytes": allocated_kv,
                "tensor_payload_bytes": int(prediction["profile"]["tensor_payload_bytes"]),
                "backend_buffers": logged, "component_comparisons": checks,
                "observed_process": observed, "log": logfile.name,
                "command_template": [args.backend.name if i == 0 else model.name if v == str(model) else v
                                     for i, v in enumerate(command)],
                "warnings": plan["warnings"],
                "passed": observed["exit_code"] == 0 and all(c["within_reported_rounding_interval"] for c in checks),
            }
            results.append(result)
    report = {
        "schema_version": "1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend_revision": REVISION, "backend_version": version,
        "backend_binary_sha256": backend_hash, "calculator_binary_sha256": calculator_hash,
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "mode": "CPU only; GPU and offload hardware behavior were not measured",
        "measurement_boundary": "exact file tensor payload and raw/padded KV predictions versus rounded backend logs; process RSS includes code, allocator and other buffers, and is not compared with policy OS reserves",
        "all_passed": all(r["passed"] for r in results), "runs": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"CPU backend comparison: {sum(r['passed'] for r in results)}/{len(results)} passed")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError, TimeoutError, KeyError) as error:
        print("backend validation failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)
