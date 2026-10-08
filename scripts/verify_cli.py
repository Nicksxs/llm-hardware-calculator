#!/usr/bin/env python3
"""Check the real Native CLI process and public JSON contract using stdlib.

Usage: python3 scripts/verify_cli.py PATH_TO_LLMCALC_BINARY

This is boundary verification, not a second placement calculator. It checks
published raw-byte examples, conservation, and budget inequalities without
reimplementing architecture selection or the layer search.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


GIB = 1073741824
STATUSES = {"FULL_GPU", "GPU+CPU OFFLOAD", "CPU ONLY", "INSUFFICIENT"}
LEDGER_BYTES = (
    "gpu_weights_bytes", "gpu_kv_bytes", "gpu_runtime_bytes",
    "gpu_margin_bytes", "gpu_staging_bytes", "host_weights_bytes",
    "host_kv_bytes", "host_runtime_bytes", "os_reserve_bytes",
    "loading_workspace_bytes", "gpu_steady_bytes", "gpu_loading_peak_bytes",
    "host_steady_bytes", "host_loading_peak_bytes",
)
POLICY_FIELDS = (
    "gpu_runtime_gib", "gpu_margin_gib", "host_runtime_gib", "os_reserve_gib",
    "loading_workspace_gib", "gpu_staging_gib",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def number(value, name):
    require(type(value) in (int, float), f"{name} must be a JSON number")
    require(math.isfinite(value) and value >= 0, f"{name} must be finite/nonnegative")
    return value


def close(actual, expected, name):
    require(
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-6),
        f"{name}: expected {expected}, got {actual}",
    )


def object_fields(value, fields, name):
    require(isinstance(value, dict), f"{name} must be an object")
    require(set(fields) <= value.keys(), f"{name} missing {set(fields) - value.keys()}")
    return value


def check_architecture(shape):
    object_fields(shape, ("layers", "kv_heads", "head_dim", "query_heads", "source"), "architecture")
    for name in ("layers", "kv_heads", "head_dim", "query_heads"):
        require(type(shape[name]) is int and shape[name] > 0, f"architecture.{name} must be positive integer")
    require(shape["query_heads"] % shape["kv_heads"] == 0, "GQA query/KV head divisibility")
    require(isinstance(shape["source"], str) and shape["source"], "architecture source must be disclosed")


def check_fit(fit, shape, weights_bytes, kv_bytes, input_values):
    object_fields(fit, ("status", "reason", "ledger"), "fit")
    require(fit["status"] in STATUSES, "fit status must use a canonical string")
    require(isinstance(fit["reason"], str) and fit["reason"], "fit reason must be present")
    ledger = fit["ledger"]
    if fit["status"] == "INSUFFICIENT":
        require(ledger is None, "INSUFFICIENT ledger must be JSON null")
        return
    object_fields(ledger, ("gpu_layers", "cpu_layers") + LEDGER_BYTES, "ledger")
    for name in LEDGER_BYTES:
        number(ledger[name], name)
    for name in ("gpu_layers", "cpu_layers"):
        require(type(ledger[name]) is int and ledger[name] >= 0, f"{name} must be nonnegative integer")
    require(ledger["gpu_layers"] + ledger["cpu_layers"] == shape["layers"], "layer split must conserve layers")
    close(ledger["gpu_weights_bytes"] + ledger["host_weights_bytes"], weights_bytes, "weight conservation")
    close(ledger["gpu_kv_bytes"] + ledger["host_kv_bytes"], kv_bytes, "KV conservation")
    for device, capacity in (("gpu", input_values["vram_gib"]), ("host", input_values["ram_gib"])):
        steady = ledger[f"{device}_steady_bytes"]
        loading = ledger[f"{device}_loading_peak_bytes"]
        require(steady <= capacity * GIB + 1e-6, f"{device} steady exceeds capacity")
        require(loading <= capacity * GIB + 1e-6, f"{device} loading peak exceeds capacity")
        require(loading + 1e-6 >= steady, f"{device} loading peak below steady budget")
    close(
        ledger["gpu_steady_bytes"],
        sum(ledger[key] for key in ("gpu_weights_bytes", "gpu_kv_bytes", "gpu_runtime_bytes", "gpu_margin_bytes")),
        "GPU steady ledger sum",
    )
    close(
        ledger["host_steady_bytes"],
        sum(ledger[key] for key in ("host_weights_bytes", "host_kv_bytes", "host_runtime_bytes", "os_reserve_bytes")),
        "RAM steady ledger sum",
    )
    if fit["status"] == "FULL_GPU":
        require(ledger["cpu_layers"] == 0, "FULL_GPU must have no CPU layers")
    elif fit["status"] == "CPU ONLY":
        require(ledger["gpu_layers"] == 0, "CPU ONLY must have no GPU layers")
        close(ledger["gpu_loading_peak_bytes"], 0, "CPU-only GPU budget")
    else:
        require(ledger["gpu_layers"] > 0 and ledger["cpu_layers"] > 0, "offload must use both devices")


def check_report(report):
    object_fields(report, ("schema_version", "input", "architecture_mode", "architecture", "weights", "kv", "policy", "fit", "sensitivity", "assumptions"), "report")
    require(report["schema_version"] == "1", "unexpected JSON schema version")
    values = object_fields(report["input"], ("params_b", "quant", "vram_gib", "ram_gib", "context_tokens"), "input")
    for name in ("params_b", "vram_gib", "ram_gib"):
        number(values[name], name)
    require(type(values["context_tokens"]) is int and values["context_tokens"] > 0, "context must be integer tokens")
    require(isinstance(values["quant"], str), "weight quantization must be a string")
    require(isinstance(report["architecture_mode"], str) and report["architecture_mode"], "architecture mode must be disclosed")
    check_architecture(report["architecture"])
    weights = object_fields(report["weights"], ("baseline_bytes", "planning_bytes", "low_bytes", "high_bytes", "baseline_bpw", "effective_bpw", "policy"), "weights")
    for name in ("baseline_bytes", "planning_bytes", "low_bytes", "high_bytes", "baseline_bpw", "effective_bpw"):
        number(weights[name], name)
    require(weights["low_bytes"] <= weights["planning_bytes"] <= weights["high_bytes"], "planning weights must be within policy scenarios")
    require(isinstance(weights["policy"], str) and "uncalibrated" in weights["policy"], "weight policy must disclose calibration boundary")
    kv = object_fields(report["kv"], ("precision", "bytes", "gib", "formula", "sequences"), "KV")
    number(kv["bytes"], "KV bytes")
    close(kv["gib"] * GIB, kv["bytes"], "KV GiB conversion")
    require(kv["precision"] in {"FP16", "Q8_0", "Q4_0"}, "KV precision must use canonical string")
    require(kv["sequences"] == 1 and isinstance(kv["formula"], str), "single-sequence KV formula must be disclosed")
    policy = object_fields(report["policy"], ("id",) + POLICY_FIELDS, "policy")
    for name in POLICY_FIELDS:
        number(policy[name], name)
    require(isinstance(policy["id"], str) and "uncalibrated" in policy["id"], "runtime policy must disclose calibration boundary")
    check_fit(report["fit"], report["architecture"], weights["planning_bytes"], kv["bytes"], values)
    require(isinstance(report["sensitivity"], list) and len(report["sensitivity"]) >= 2, "weight sensitivity scenarios must be present")
    for scenario in report["sensitivity"]:
        object_fields(scenario, ("label", "architecture", "weights_bytes", "kv_bytes", "fit"), "sensitivity")
        require(isinstance(scenario["label"], str) and scenario["label"], "scenario label must be present")
        check_architecture(scenario["architecture"])
        number(scenario["weights_bytes"], "scenario weights")
        number(scenario["kv_bytes"], "scenario KV")
        check_fit(scenario["fit"], scenario["architecture"], scenario["weights_bytes"], scenario["kv_bytes"], values)
    object_fields(report["assumptions"], ("scope", "offload", "loading", "kv_runtime", "weight_range", "units"), "assumptions")


def basic(params="27B", quant="Q4_K_M", context="8K", vram="16GiB", ram="64GiB"):
    return ["--params", params, "--quant", quant, "--vram", vram, "--ram", ram, "--context", context]


class NativeChecks:
    def __init__(self, binary):
        self.binary = binary
        self.processes = 0
        self.cases = 0
        self.failures = []

    def run(self, args):
        self.processes += 1
        return subprocess.run([str(self.binary), *args], capture_output=True, text=True, timeout=10)

    def case(self, name, action):
        self.cases += 1
        try:
            action()
        except (AssertionError, ValueError, KeyError, TypeError, OSError, subprocess.TimeoutExpired) as error:
            self.failures.append(f"{name}: {error}")

    def json_report(self, args):
        result = self.run([*args, "--json"])
        require(result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()}")
        require(not result.stderr, f"unexpected stderr: {result.stderr.strip()}")
        def reject_constant(value):
            raise ValueError(f"nonstandard JSON numeric constant: {value}")
        report = json.loads(result.stdout, parse_constant=reject_constant)
        check_report(report)
        return report

    def raw_case(self, args, params_b, quant, context, baseline_bytes, kv_bytes):
        report = self.json_report(args)
        close(report["input"]["params_b"], params_b, "parameter units")
        require(report["input"]["quant"] == quant, "quant name mismatch")
        require(report["input"]["context_tokens"] == context, "context normalization")
        close(report["weights"]["baseline_bytes"], baseline_bytes, "raw GGML weight oracle")
        close(report["kv"]["bytes"], kv_bytes, "raw KV oracle")

    def text_matches(self, args):
        report = self.json_report(args)
        result = self.run(args)
        require(result.returncode == 0 and not result.stderr, "text output must succeed without stderr")
        status = [line.removeprefix("Fit: ") for line in result.stdout.splitlines() if line.startswith("Fit: ")]
        require(status == [report["fit"]["status"]], "text and JSON fit status disagree")
        for detail in ("Source:", "loading peak:", "RAM steady:", "uncalibrated", "K=1024"):
            require(detail in result.stdout, f"text output lacks {detail}")

    def error_case(self, args):
        result = self.run(args)
        require(result.returncode in (1, 2), f"expected parse/calculate exit 1 or 2, got {result.returncode}")
        require(not result.stdout, "failed command must leave stdout empty")
        require(result.stderr.startswith("llmcalc:"), "failed command must explain error on stderr")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="path to an already-built Native llmcalc executable")
    binary = parser.parse_args().binary.resolve()
    if not binary.is_file():
        parser.error(f"executable does not exist: {binary}")
    checks = NativeChecks(binary)

    # Exact published block-payload bytes for 27 decimal billion parameters;
    # planning allowances and layer placement are deliberately not duplicated.
    weight_oracles = {"Q4_K_M": 15187500000, "IQ4_XS": 14343750000, "Q5_K_M": 18562500000, "Q8_0": 28687500000}
    for quant, weight_bytes in weight_oracles.items():
        for context_k in (8, 16, 32, 64):
            args = basic(quant=quant, context=f"{context_k}K")
            checks.case(
                f"27B {quant} {context_k}K",
                lambda args=args, quant=quant, context_k=context_k, weight_bytes=weight_bytes: checks.raw_case(
                    args, 27, quant, context_k * 1024, weight_bytes, (context_k // 8) * 2 * GIB,
                ),
            )
    for params_b, weight_bytes, kv_bytes in (
        (7, 3937500000, 234881024), (14, 7875000000, 805306368),
        (32, 18000000000, 1073741824), (70, 39375000000, 1342177280),
    ):
        checks.case(
            f"{params_b}B baseline 4K",
            lambda params_b=params_b, weight_bytes=weight_bytes, kv_bytes=kv_bytes: checks.raw_case(
                basic(params=f"{params_b}B", context="4K"), params_b, "Q4_K_M", 4096, weight_bytes, kv_bytes,
            ),
        )

    def advanced_q8():
        report = checks.json_report(basic() + ["--layers=64", "--kv-heads=8", "--head-dim=128", "--query-heads=40", "--kv=Q8_0"])
        require(report["kv"]["precision"] == "Q8_0", "advanced Q8 precision name")
        close(report["kv"]["bytes"], 1140850688, "Q8_0 KV raw bytes (34/32 B per element)")
        close(report["kv"]["gib"], 1.0625, "Q8_0 KV oracle in GiB")
        require("Advanced" in report["architecture_mode"], "advanced mode disclosure")
    checks.case("Advanced Q8 KV 1.0625 GiB oracle", advanced_q8)

    def cpu_only():
        report = checks.json_report(basic(vram="0GiB"))
        require(report["fit"]["status"] == "CPU ONLY", "zero VRAM must permit normal CPU-only result")
    checks.case("zero VRAM CPU-only success", cpu_only)

    def insufficient():
        report = checks.json_report(basic(vram="0GiB", ram="0GiB"))
        require(report["fit"]["status"] == "INSUFFICIENT" and report["fit"]["ledger"] is None, "insufficient result must succeed with null ledger")
    checks.case("insufficient is valid JSON result", insufficient)

    def decimal_capacity():
        report = checks.json_report(basic(vram="16GB", ram="64GB") + ["--os-reserve=0", "--gpu-staging=0"])
        close(report["input"]["vram_gib"], 16000000000 / GIB, "decimal VRAM conversion")
        close(report["input"]["ram_gib"], 64000000000 / GIB, "decimal RAM conversion")
        require(report["policy"]["os_reserve_gib"] == 0, "already-free RAM override")
        require("custom CLI overrides" in report["policy"]["id"], "custom budget policy disclosure")
    checks.case("GB conversion and already-free RAM policy", decimal_capacity)

    def all_overrides():
        report = checks.json_report(basic() + ["--gpu-runtime=0", "--gpu-margin=0", "--host-runtime=0", "--os-reserve=0", "--loading-workspace=0", "--gpu-staging=0"])
        require(all(report["policy"][key] == 0 for key in POLICY_FIELDS), "all six budget overrides must reach calculation")
    checks.case("six policy overrides reach Native calculation", all_overrides)
    checks.case("text matches JSON for offload", lambda: checks.text_matches(basic()))
    checks.case("text matches JSON for full GPU", lambda: checks.text_matches(basic(params="7B", context="4K")))

    def help_case():
        result = checks.run(["--help"])
        require(result.returncode == 0 and not result.stderr, "help must exit successfully without stderr")
        require("Usage:" in result.stdout and "--os-reserve" in result.stdout, "help must describe input and policy flags")
    checks.case("help without required inputs", help_case)
    error_cases = (
        ("unknown flag", basic() + ["--unknown"]),
        ("duplicate basic option", basic() + ["--params=28"]),
        ("missing required options", ["--params=27B"]),
        ("missing flag value", basic() + ["--kv"]),
        ("negative parameter count", basic(params="-1B")),
        ("zero context", basic(context="0")),
        ("NaN capacity", basic(vram="NaNGiB")),
        ("NaN policy", basic() + ["--os-reserve=NaN"]),
        ("unknown weight quantization", basic(quant="Q3_K_M")),
        ("ambiguous IQ4 recipe", basic(quant="IQ4")),
        ("partial architecture", basic() + ["--layers=64"]),
        ("query heads not divisible by KV heads", basic() + ["--layers=32", "--kv-heads=3", "--head-dim=128", "--query-heads=5"]),
        ("quantized KV incomplete block", basic() + ["--layers=64", "--kv-heads=8", "--head-dim=100", "--kv=Q8_0"]),
        ("out-of-range proxy requires advanced structure", basic(params="100B")),
    )
    for name, args in error_cases:
        checks.case(name, lambda args=args: checks.error_case(args))

    for failure in checks.failures:
        print("FAIL: " + failure, file=sys.stderr)
    print(f"Native CLI verification: {checks.cases - len(checks.failures)}/{checks.cases} cases passed; {checks.processes} real process checks")
    require(checks.processes >= 30, "verification must exercise at least 30 real processes")
    return 1 if checks.failures else 0


if __name__ == "__main__":
    sys.exit(main())
