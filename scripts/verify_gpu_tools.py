#!/usr/bin/env python3
"""Regress validation-tool safety and allocation checks without GPU hardware.

Run: python3 scripts/verify_gpu_tools.py

Only temporary sentinel files, synthetic tensor descriptors, and source-format
log text are used. No model, NumPy, CUDA, SSH, or external executable is needed.
Every test blocks real subprocess.run/Popen calls; NVML responses are mocked.
These tests verify collector behavior, not actual backend or GPU compatibility.
"""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

# Local imports must not create cache files when this standalone check runs.
sys.dont_write_bytecode = True
import make_tied_fixture as fixture
import validate_backend as cpu
import validate_gpu_backend as gpu


class NoExternalProcesses(unittest.TestCase):
    def setUp(self):
        for name in ("run", "Popen"):
            patcher = mock.patch.object(
                subprocess, name,
                side_effect=AssertionError("external subprocess is forbidden in this regression suite"),
            )
            setattr(self, "blocked_" + name, patcher.start())
            self.addCleanup(patcher.stop)

    def assert_no_processes(self):
        self.blocked_run.assert_not_called()
        self.blocked_Popen.assert_not_called()


class OutputSafetyTests(NoExternalProcesses):
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory(prefix="llmcalc-tool-regression-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.inputs = {}
        self.original = {}
        for name in ("calculator", "backend", "model.gguf", "tied.gguf", "nvidia-smi", "manifest.json"):
            path = self.directory / name
            data = ("untouched sentinel: " + name + "\n").encode()
            path.write_bytes(data)
            self.inputs[name] = path
            self.original[path] = data

    def assert_sources_unchanged(self):
        for path, data in self.original.items():
            self.assertEqual(path.read_bytes(), data, str(path))

    def gpu_arguments(self, output):
        return ["validate_gpu_backend.py", "--calculator", str(self.inputs["calculator"]),
                "--backend", str(self.inputs["backend"]), "--model", str(self.inputs["model.gguf"]),
                "--tied-model", str(self.inputs["tied.gguf"]),
                "--nvidia-smi", str(self.inputs["nvidia-smi"]), "--output", str(output)]

    def assert_gpu_refused_before_launch(self, output):
        with mock.patch.object(sys, "argv", self.gpu_arguments(output)), \
                mock.patch.object(gpu.shutil, "which", return_value=str(self.inputs["nvidia-smi"])), \
                mock.patch.object(gpu, "write_new_text") as writer, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(gpu.main(), 1)
        writer.assert_not_called()
        self.assert_no_processes()
        self.assert_sources_unchanged()

    def test_gpu_report_cannot_be_any_direct_input(self):
        for source in self.inputs.values():
            # manifest is not an argument to the GPU runner, so only its actual
            # protected inputs are included in this matrix.
            if source.name == "manifest.json":
                continue
            with self.subTest(input=source.name):
                self.assert_gpu_refused_before_launch(source)

    def test_gpu_report_symlinks_and_hardlinks_cannot_alias_inputs(self):
        for kind in ("symlink", "hardlink"):
            for source in list(self.inputs.values())[:-1]:
                with self.subTest(kind=kind, input=source.name):
                    output = self.directory / (kind + "-" + source.name + ".json")
                    if kind == "symlink":
                        output.symlink_to(source)
                    else:
                        os.link(source, output)
                    self.assert_gpu_refused_before_launch(output)

    def test_gpu_log_symlinks_and_hardlinks_are_refused_before_launch(self):
        model = self.inputs["model.gguf"].resolve()
        tied = self.inputs["tied.gguf"].resolve()
        for kind in ("symlink", "hardlink"):
            for stream_index in (0, 1):
                with self.subTest(kind=kind, stream=stream_index):
                    output = self.directory / (kind + "-log-" + str(stream_index) + ".json")
                    _, logs = gpu.validate_output_targets(
                        output, list(self.inputs.values()), gpu.acceptance_cases(model, tied))
                    target = logs[0][stream_index]
                    if kind == "symlink":
                        target.symlink_to(self.inputs["backend"])
                    else:
                        os.link(self.inputs["backend"], target)
                    self.assert_gpu_refused_before_launch(output)
                    self.assertFalse(output.exists())

    def test_gpu_dangling_report_symlink_is_not_a_fresh_destination(self):
        missing = self.directory / "missing-report-target.json"
        output = self.directory / "dangling-report.json"
        output.symlink_to(missing)
        self.assert_gpu_refused_before_launch(output)
        self.assertFalse(missing.exists())

    def test_gpu_fresh_target_guard_is_read_only(self):
        output = self.directory / "fresh.json"
        cases = gpu.acceptance_cases(self.inputs["model.gguf"], self.inputs["tied.gguf"])
        actual, logs = gpu.validate_output_targets(output, list(self.inputs.values()), cases)
        self.assertEqual(actual, output.resolve())
        self.assertEqual(len(logs), 7)
        self.assertFalse(output.exists())
        self.assertTrue(all(not path.exists() for pair in logs for path in pair))
        self.assert_sources_unchanged()
        self.assert_no_processes()

    def test_exclusive_writer_refuses_links_created_after_preflight(self):
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                output = self.directory / ("late-" + kind + ".json")
                gpu.validate_output_targets(output, list(self.inputs.values()), [])
                if kind == "symlink":
                    output.symlink_to(self.inputs["model.gguf"])
                else:
                    os.link(self.inputs["model.gguf"], output)
                with self.assertRaises(FileExistsError):
                    gpu.write_new_text(output, "must not replace source")
                self.assert_sources_unchanged()
        self.assert_no_processes()

    def test_fixture_existing_manifest_links_refuse_before_source_or_numpy(self):
        for kind in ("existing", "symlink", "hardlink", "dangling"):
            with self.subTest(kind=kind):
                manifest = self.directory / ("fixture-" + kind + ".json")
                if kind == "existing":
                    manifest.write_bytes(b"old manifest must survive")
                elif kind == "hardlink":
                    os.link(self.inputs["model.gguf"], manifest)
                else:
                    manifest.symlink_to(self.inputs["model.gguf"] if kind == "symlink"
                                        else self.directory / "missing-manifest-target")
                output = self.directory / ("fixture-" + kind + ".gguf")
                argv = ["make_tied_fixture.py", "--source", str(self.inputs["model.gguf"]),
                        "--gguf-python", str(self.directory / "no-upstream-needed"),
                        "--output", str(output), "--manifest", str(manifest)]
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(fixture, "sha256") as source_hash, \
                        mock.patch.object(fixture, "load_reader") as reader:
                    with self.assertRaises(ValueError):
                        fixture.main()
                source_hash.assert_not_called()
                reader.assert_not_called()
                self.assertFalse(output.exists())
                if kind == "existing":
                    self.assertEqual(manifest.read_bytes(), b"old manifest must survive")
                self.assert_sources_unchanged()
        self.assert_no_processes()

    def test_cpu_report_hardlink_refuses_before_any_subprocess(self):
        output = self.directory / "cpu-report.json"
        os.link(self.inputs["model.gguf"], output)
        self.assert_cpu_refused_before_launch(output)

    def test_cpu_log_symlink_refuses_before_any_subprocess(self):
        output = self.directory / "cpu-new-report.json"
        log = self.directory / "model-ctx128.log"
        log.symlink_to(self.inputs["backend"])
        self.assert_cpu_refused_before_launch(output)
        self.assertFalse(output.exists())

    def assert_cpu_refused_before_launch(self, output):
        argv = ["validate_backend.py", "--calculator", str(self.inputs["calculator"]),
                "--backend", str(self.inputs["backend"]), "--model", str(self.inputs["model.gguf"]),
                "--manifest", str(self.inputs["manifest.json"]), "--context", "128", "--output", str(output)]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, "overwrite an input"):
                cpu.main()
        self.assert_no_processes()
        self.assert_sources_unchanged()

    def test_unrecognized_tied_fixture_refuses_before_directory_parsing(self):
        candidate = self.directory / "wrong-same-size-tied.gguf"
        with candidate.open("wb") as stream:
            stream.truncate(gpu.TIED_MODEL_SIZE)
        before = gpu.sha256(candidate)
        self.assertNotEqual(before, gpu.TIED_MODEL_SHA256)
        with mock.patch.object(gpu, "read_directory") as reader:
            with self.assertRaisesRegex(ValueError, "fixed derived"):
                gpu.verify_tied_fixture(candidate)
        reader.assert_not_called()
        self.assertEqual(gpu.sha256(candidate), before)
        self.assert_sources_unchanged()
        self.assert_no_processes()


class TelemetryTests(NoExternalProcesses):
    def sample(self, free="12288", utilization="N/A"):
        row = "0, GPU-00000000-0000-0000-0000-000000000001, RTX 5060 Ti, 581.57, 16384, 4096, " + free + ", " + utilization + "\n"
        result = subprocess.CompletedProcess([], 0, row, "")
        with mock.patch.object(subprocess, "run", return_value=result) as query:
            sample = gpu.smi_sample("mock-nvidia-smi", "test", time.monotonic())
        self.assertEqual(query.call_count, 1)
        self.assert_no_processes()
        return sample

    def test_unavailable_free_memory_stays_null_and_refuses(self):
        for unavailable in ("N/A", "[N/A]", "[Not Supported]", "Unknown Error", "nan"):
            with self.subTest(field=unavailable):
                sample = self.sample(free=unavailable)
                self.assertIsNone(sample["values"]["memory.free"])
                self.assertIn("memory.free", sample["unavailable_fields"])
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    gpu.available_gpu(sample)

    def test_low_free_memory_is_preserved_and_refused(self):
        for free in ("1", "2047", "2047.5"):
            with self.subTest(free=free):
                sample = self.sample(free=free)
                self.assertGreater(sample["values"]["memory.free"], 0)
                with self.assertRaisesRegex(ValueError, "less than 2 GiB"):
                    gpu.available_gpu(sample)

    def test_exact_free_threshold_accepts_unavailable_utilization(self):
        sample = self.sample(free="2048")
        self.assertEqual(gpu.available_gpu(sample)["memory.free"], 2048)
        self.assertIsNone(sample["values"]["utilization.gpu"])
        self.assertIn("utilization.gpu", sample["unavailable_fields"])

    def test_nvml_utilization_query_failure_retries_memory_only(self):
        failed = subprocess.CompletedProcess([], 1, "", "utilization unsupported")
        identity = subprocess.CompletedProcess([], 0,
            "0, GPU-00000000-0000-0000-0000-000000000001, RTX 5060 Ti, 581.57, 16384, 4096, 12288\n", "")
        with mock.patch.object(subprocess, "run", side_effect=[failed, identity]) as query:
            sample = gpu.smi_sample("mock-nvidia-smi", "test", time.monotonic())
        self.assertEqual(query.call_count, 2)
        self.assertNotIn("utilization.gpu", query.call_args_list[1].args[0][3])
        self.assertIsNone(sample["values"]["utilization.gpu"])
        self.assertEqual(sample["unavailable_fields"]["utilization.gpu"], "utilization unsupported")
        self.assertEqual(gpu.available_gpu(sample)["memory.free"], 12288)
        self.assert_no_processes()

    def test_failed_nvml_queries_never_become_zero_measurements(self):
        failed = subprocess.CompletedProcess([], 1, "", "NVML unavailable")
        with mock.patch.object(subprocess, "run", return_value=failed) as query:
            sample = gpu.smi_sample("mock-nvidia-smi", "test", time.monotonic())
        self.assertEqual(query.call_count, 2)
        self.assertIsNone(sample["values"])
        self.assertIn("NVML unavailable", sample["unavailable_reason"])
        with self.assertRaises(ValueError):
            gpu.available_gpu(sample)
        self.assert_no_processes()


class AllocationAndLogTests(NoExternalProcesses):
    # Synthetic values in the exact DEBUG/INFO formats emitted by the pinned
    # llama-model.cpp, llama-kv-cache.cpp and model buffer logging. They are not
    # measurements captured from hardware.
    LOG = """load_tensors: layer   0 assigned to device CPU, is_swa = 0
load_tensors: layer   1 assigned to device CUDA0, is_swa = 0
load_tensors: layer   2 assigned to device CUDA0, is_swa = 0
load_tensors:        CPU model buffer size =     1.25 MiB
load_tensors:      CUDA0 model buffer size =     2.50 MiB
llama_kv_cache: layer   0: dev = CPU
llama_kv_cache: layer   1: dev = CUDA0
llama_kv_cache:        CPU KV buffer size =     0.50 MiB
llama_kv_cache:      CUDA0 KV buffer size =     1.00 MiB
llama_context:      CUDA0 compute buffer size =     4.00 MiB
llama_context:        CPU output buffer size =     0.25 MiB
"""
    EXPECTED = {"model_layer_devices": {0: "CPU", 1: "CUDA0", 2: "CUDA0"},
                "kv_layer_devices": {0: "CPU", 1: "CUDA0"},
                "model": {"CPU": 1310720, "CUDA0": 2621440},
                "kv": {"CPU": 524288, "CUDA0": 1048576}}

    def outcome(self, text, expected=None):
        checks, buffers, components = gpu.backend_checks(text, expected or self.EXPECTED)
        passed = all(check["passed"] for check in checks) and all(
            component["within_reported_rounding_interval"] for component in components)
        self.assert_no_processes()
        return passed, {check["check"]: check for check in checks}, buffers, components

    def test_source_format_log_and_ansi_are_parsed_correctly(self):
        passed, checks, buffers, components = self.outcome("\x1b[32m" + self.LOG + "\x1b[0m")
        self.assertTrue(passed)
        self.assertEqual(len(buffers), 6)
        self.assertEqual(len(components), 4)
        self.assertTrue(checks["backend.kv_layer_devices"]["passed"])
        self.assertEqual(buffers[0]["rounded_bytes"], 1310720)
        self.assertEqual(buffers[0]["rounding_half_interval_bytes"], 5242.88)

    def test_wrong_kv_device_is_a_failure(self):
        passed, checks, _, _ = self.outcome(self.LOG.replace("layer   1: dev = CUDA0", "layer   1: dev = CPU"))
        self.assertFalse(passed)
        self.assertFalse(checks["backend.kv_layer_devices"]["passed"])

    def test_fallback_and_allocation_errors_are_failures(self):
        for error in ("tensor cannot be used with preferred buffer type", "CUDA error: allocation failed",
                      "failed to allocate buffer", "no usable GPU found"):
            with self.subTest(error=error):
                passed, checks, _, _ = self.outcome(self.LOG + error + "\n")
                self.assertFalse(passed)
                self.assertFalse(checks["backend.no_fallback_or_allocation_error"]["passed"])

    def test_incorrect_allocation_is_outside_printed_rounding_interval(self):
        passed, _, _, components = self.outcome(self.LOG.replace("2.50 MiB", "2.52 MiB"))
        self.assertFalse(passed)
        cuda_model = next(c for c in components if c["device"] == "CUDA0" and c["component"] == "model")
        self.assertFalse(cuda_model["within_reported_rounding_interval"])

    def test_duplicate_layer_log_cannot_hide_behind_equal_device_dictionary(self):
        passed, checks, _, _ = self.outcome(self.LOG + self.LOG.splitlines()[0] + "\n")
        self.assertFalse(passed)
        self.assertTrue(checks["backend.model_layer_devices"]["passed"])
        self.assertFalse(checks["backend.model_layer_log_count"]["passed"])

    def test_missing_nonzero_buffer_is_not_a_zero_measurement(self):
        text = "\n".join(line for line in self.LOG.splitlines() if "CUDA0 model buffer" not in line)
        passed, _, _, components = self.outcome(text)
        self.assertFalse(passed)
        missing = next(c for c in components if c["device"] == "CUDA0" and c["component"] == "model")
        self.assertEqual(missing["matching_log_count"], 0)
        self.assertIsNone(missing["difference_from_rounded_backend_bytes"])

    @staticmethod
    def tiny_directory(type_id=2, dimensions=None, payload=216, tied=False):
        # Partial synthetic inventory for exercising the allocation function,
        # deliberately not a complete/loadable LLM or a parser fixture.
        tensors = [
            {"name": "token_embd.weight", "type_id": 0, "dimensions": [4], "byte_size": 16},
            {"name": "output_norm.weight", "type_id": 0, "dimensions": [4], "byte_size": 16},
            {"name": "blk.0.attn_q.weight", "type_id": type_id,
             "dimensions": dimensions or [128, 3], "byte_size": payload},
        ]
        if not tied:
            tensors.append({"name": "output.weight", "type_id": 0, "dimensions": [4, 4], "byte_size": 64})
        return {"layers": 1, "tensors": tensors, "tied_output": tied,
                "payload_bytes": sum(t["byte_size"] for t in tensors), "kv_heads": 1, "head_dim": 4}

    def test_quantized_cuda_tail_is_added_once_per_tensor(self):
        # Q4_0 [128,3]: raw 216, one trailing 384-element row pad 216,
        # then CUDA128 alignment => 512, not a pad for each of the 3 rows.
        expected = gpu.oracle(self.tiny_directory(), 256, 2)
        record = next(a for a in expected["allocations"] if a["name"] == "blk.0.attn_q.weight")
        self.assertEqual((record["raw_payload_bytes"], record["quantized_tail_padding_bytes"], record["allocated_bytes"]),
                         (216, 216, 512))
        self.assertEqual(expected["model"], {"CPU": 32, "CUDA0": 768})
        self.assert_no_processes()

    def test_k_quantized_tail_uses_its_256_element_block(self):
        # Q4_K [256,4]: raw 576, one 256-element pad 144 => 720 => 768.
        expected = gpu.oracle(self.tiny_directory(12, [256, 4], 576), 256, 2)
        record = next(a for a in expected["allocations"] if a["name"] == "blk.0.attn_q.weight")
        self.assertEqual((record["quantized_tail_padding_bytes"], record["allocated_bytes"]), (144, 768))
        self.assert_no_processes()

    def test_cpu_quantized_tensor_has_no_cuda_tail(self):
        expected = gpu.oracle(self.tiny_directory(), 256, 0)
        record = next(a for a in expected["allocations"] if a["name"] == "blk.0.attn_q.weight")
        self.assertEqual((record["quantized_tail_padding_bytes"], record["allocated_bytes"]), (0, 224))
        self.assert_no_processes()

    def test_tied_output_first_slot_duplicates_embedding_without_gpu_kv(self):
        directory = self.tiny_directory(tied=True)
        cpu_only, output_only = gpu.oracle(directory, 256, 0), gpu.oracle(directory, 256, 1)
        self.assertEqual(cpu_only["duplicated_bytes"], 0)
        self.assertEqual(output_only["duplicated_bytes"], 16)
        self.assertEqual(output_only["gpu_blocks"], 0)
        self.assertEqual(cpu_only["kv"], {"CPU": 4096, "CUDA0": 0})
        self.assertEqual(output_only["kv"], cpu_only["kv"])
        self.assertEqual(output_only["raw"], {"CPU": 232, "CUDA0": 32})
        self.assertEqual(output_only["model"], {"CPU": 256, "CUDA0": 256})
        self.assert_no_processes()


if __name__ == "__main__":
    unittest.main(verbosity=2)
