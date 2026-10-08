///|
fn architecture_json(architecture : @core.Architecture) -> Json {
  {
    "layers": Json(architecture.layers),
    "kv_heads": Json(architecture.kv_heads),
    "head_dim": Json(architecture.head_dim),
    "query_heads": Json(architecture.query_heads),
    "source": Json(architecture.source),
  }
}

///|
fn ledger_json(ledger : @core.MemoryLedger) -> Json {
  {
    "gpu_layers": Json(ledger.gpu_layers),
    "cpu_layers": Json(ledger.cpu_layers),
    "gpu_weights_bytes": Json(ledger.gpu_weights_bytes),
    "gpu_kv_bytes": Json(ledger.gpu_kv_bytes),
    "gpu_runtime_bytes": Json(ledger.gpu_runtime_bytes),
    "gpu_margin_bytes": Json(ledger.gpu_margin_bytes),
    "gpu_staging_bytes": Json(ledger.gpu_staging_bytes),
    "host_weights_bytes": Json(ledger.host_weights_bytes),
    "host_kv_bytes": Json(ledger.host_kv_bytes),
    "host_runtime_bytes": Json(ledger.host_runtime_bytes),
    "os_reserve_bytes": Json(ledger.os_reserve_bytes),
    "loading_workspace_bytes": Json(ledger.loading_workspace_bytes),
    "gpu_steady_bytes": Json(ledger.gpu_steady_bytes),
    "gpu_loading_peak_bytes": Json(ledger.gpu_loading_peak_bytes),
    "host_steady_bytes": Json(ledger.host_steady_bytes),
    "host_loading_peak_bytes": Json(ledger.host_loading_peak_bytes),
  }
}

///|
fn fit_json(fit : @core.FitResult) -> Json {
  let ledger = match fit.ledger {
    Some(ledger) => ledger_json(ledger)
    None => Json::null()
  }
  {
    "status": Json(@core.fit_status_name(fit.status)),
    "reason": Json(fit.reason),
    "ledger": ledger,
  }
}

///|
/// Stable names and explicit units avoid enum-derived serialization surprises.
pub fn report_json(report : @core.Report) -> Json {
  let input = report.input
  let weights = report.weights
  let policy = report.policy
  let sensitivity : Array[Json] = report.sensitivity.map(scenario => {
    "label": Json(scenario.label),
    "architecture": architecture_json(scenario.architecture),
    "weights_bytes": Json(scenario.weights_bytes),
    "kv_bytes": Json(scenario.kv_bytes),
    "fit": fit_json(scenario.fit),
  })
  {
    "schema_version": "1",
    "input": {
      "params_b": Json(input.params_b),
      "quant": Json(@core.weight_quant_name(input.quant)),
      "vram_gib": Json(input.vram_gib),
      "ram_gib": Json(input.ram_gib),
      "context_tokens": Json(input.context),
    },
    "architecture_mode": Json(report.architecture_mode),
    "architecture": architecture_json(report.architecture),
    "weights": {
      "baseline_bytes": Json(weights.baseline_bytes),
      "planning_bytes": Json(weights.planning_bytes),
      "low_bytes": Json(weights.low_bytes),
      "high_bytes": Json(weights.high_bytes),
      "baseline_bpw": Json(weights.baseline_bpw),
      "effective_bpw": Json(weights.effective_bpw),
      "policy": Json(weights.policy),
    },
    "kv": {
      "precision": Json(@core.kv_precision_name(report.kv_precision)),
      "bytes": Json(report.kv_bytes),
      "gib": Json(@core.bytes_to_gib(report.kv_bytes)),
      "formula": "context * layers * kv_heads * head_dim * 2 * bytes_per_element",
      "sequences": 1,
    },
    "policy": {
      "id": Json(policy.id),
      "gpu_runtime_gib": Json(policy.gpu_runtime_gib),
      "gpu_margin_gib": Json(policy.gpu_margin_gib),
      "host_runtime_gib": Json(policy.host_runtime_gib),
      "os_reserve_gib": Json(policy.os_reserve_gib),
      "loading_workspace_gib": Json(policy.loading_workspace_gib),
      "gpu_staging_gib": Json(policy.gpu_staging_gib),
    },
    "fit": fit_json(report.fit),
    "sensitivity": Json::array(sensitivity),
    "assumptions": {
      "scope": "one GPU, dense decoder-only full attention, MHA/GQA, one sequence",
      "offload": "uniform layer weight and KV split; not a runtime tensor placement guarantee",
      "loading": "full host weight copy plus CPU KV, host runtime, OS reserve and loading workspace",
      "kv_runtime": "quantized KV is a memory scenario; runtime/weight/KV compatibility is not guaranteed",
      "weight_range": "uncalibrated planning scenarios, not measured accuracy or confidence bounds",
      "units": "B parameters = 1e9; GiB = 2^30 bytes; GB input = 1e9 bytes; K context = 1024",
    },
  }
}

///|
pub fn render_json(report : @core.Report) -> String {
  @json.to_json(report_json(report)).stringify(indent=2)
}

///|
fn architecture_text(architecture : @core.Architecture) -> String {
  "layers=\{architecture.layers}, KV heads=\{architecture.kv_heads}, head dim=\{architecture.head_dim}, query heads=\{architecture.query_heads}\nSource: \{architecture.source}"
}

///|
fn gib(bytes : Double) -> String {
  "\{@core.bytes_to_gib(bytes)} GiB"
}

///|
fn ledger_text(ledger : @core.MemoryLedger) -> String {
  "Layer split: GPU \{ledger.gpu_layers}, CPU \{ledger.cpu_layers}\n" +
  "GPU: weights \{gib(ledger.gpu_weights_bytes)}, KV \{gib(ledger.gpu_kv_bytes)}, runtime \{gib(ledger.gpu_runtime_bytes)}, margin \{gib(ledger.gpu_margin_bytes)}\n" +
  "GPU steady: \{gib(ledger.gpu_steady_bytes)}; loading peak: \{gib(ledger.gpu_loading_peak_bytes)} (staging \{gib(ledger.gpu_staging_bytes)})\n" +
  "RAM: weights \{gib(ledger.host_weights_bytes)}, KV \{gib(ledger.host_kv_bytes)}, runtime \{gib(ledger.host_runtime_bytes)}, OS reserve \{gib(ledger.os_reserve_bytes)}\n" +
  "RAM steady: \{gib(ledger.host_steady_bytes)}; loading peak: \{gib(ledger.host_loading_peak_bytes)} (full host weights plus loading workspace \{gib(ledger.loading_workspace_bytes)})"
}

///|
pub fn render_text(report : @core.Report) -> String {
  let input = report.input
  let weights = report.weights
  let policy = report.policy
  let ledger = match report.fit.ledger {
    Some(ledger) => ledger_text(ledger)
    None => "No feasible layer placement under the steady and loading budgets."
  }
  let lines : Array[String] = [
    "LLM Hardware Calculator",
    "Fit: \{@core.fit_status_name(report.fit.status)}",
    "Reason: \{report.fit.reason}",
    "Input: \{input.params_b}B, \{@core.weight_quant_name(input.quant)}, VRAM \{input.vram_gib} GiB, RAM \{input.ram_gib} GiB, context \{input.context} tokens",
    "Architecture mode: \{report.architecture_mode}",
    architecture_text(report.architecture),
    "Weights: block baseline \{gib(weights.baseline_bytes)} (\{weights.baseline_bpw} bpw); planning \{gib(weights.planning_bytes)} (\{weights.effective_bpw} bpw); low/high \{gib(weights.low_bytes)} / \{gib(weights.high_bytes)}",
    "Weight policy: \{weights.policy}",
    "KV: \{@core.kv_precision_name(report.kv_precision)}, \{gib(report.kv_bytes)}; one sequence, K and V stored separately",
    "Budget policy: \{policy.id}",
    "Allowances (GiB): GPU runtime \{policy.gpu_runtime_gib}, GPU margin \{policy.gpu_margin_gib}, staging \{policy.gpu_staging_gib}; host runtime \{policy.host_runtime_gib}, OS reserve \{policy.os_reserve_gib}, loading workspace \{policy.loading_workspace_gib}",
    ledger,
  ]
  if !report.sensitivity.is_empty() {
    lines.push("Sensitivity scenarios:")
    for scenario in report.sensitivity {
      lines.push(
        "  \{scenario.label}: weights \{gib(scenario.weights_bytes)}, KV \{gib(scenario.kv_bytes)}, \{@core.fit_status_name(scenario.fit.status)}; layers=\{scenario.architecture.layers}, KV heads=\{scenario.architecture.kv_heads}, head dim=\{scenario.architecture.head_dim}, query heads=\{scenario.architecture.query_heads}",
      )
      lines.push("  Source: \{scenario.architecture.source}")
      lines.push("  Reason: \{scenario.fit.reason}")
    }
  }
  lines.push(
    "Assumptions: one GPU; dense decoder-only full attention; MHA/GQA; one sequence. Layer weights and KV are split uniformly. Loading retains a full host weight copy. Quantized KV represents memory scenarios, not guaranteed runtime combinations. Weight ranges and runtime allowances are uncalibrated policies, not measured accuracy bounds.",
  )
  lines.push(
    "Units: parameters B=1e9; GiB=2^30 bytes; decimal GB input is converted to GiB; context K=1024 tokens.",
  )
  lines.join("\n")
}

///|
pub fn help_text() -> String {
  "LLM Hardware Calculator\n\nUsage:\n  llmcalc --params 27B --quant Q4_K_M --vram 16GiB --ram 64GiB --context 8K [options]\n\nBasic (all five required):\n  --params NUMBER[B]  Decimal billions; a plain number also means billions\n  --quant NAME        FP16, BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS, IQ4_NL\n  --vram NUMBERGiB    GPU memory; decimal GB also accepted\n  --ram NUMBERGiB     System RAM; decimal GB also accepted\n  --context INTEGER[K] Tokens; K means 1024\n\nAdvanced:\n  --layers INTEGER --kv-heads INTEGER --head-dim INTEGER\n                      Supply all three together; otherwise use a disclosed proxy\n  --query-heads INTEGER  Defaults to --kv-heads for an explicit architecture\n  --kv NAME           FP16 (default), Q8_0, Q4_0; memory scenarios only\n\nBudget policy overrides (plain numbers mean GiB; GiB/GB also accepted):\n  --gpu-runtime NUMBER       Default 0.5 GiB\n  --gpu-margin NUMBER        Default 1 GiB\n  --host-runtime NUMBER      Default 1 GiB\n  --os-reserve NUMBER        Default 4 GiB; use 0 for already-free RAM\n  --loading-workspace NUMBER Default 1 GiB\n  --gpu-staging NUMBER       Default 0.25 GiB\n\nOutput:\n  --json              Emit a machine-readable JSON report\n  --help              Show this help\n\nFlags accept --flag=value. Duplicate or unknown flags are errors.\nScope: one GPU, dense decoder-only full attention, MHA/GQA, one sequence.\nFit includes full GPU, CPU offload, CPU only, and insufficient memory, with\nseparate GPU/RAM steady and loading budgets under uncalibrated policies."
}
