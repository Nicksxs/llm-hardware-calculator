///|
fn basic_args() -> Array[String] {
  [
    "--params", "27B", "--quant", "Q4_K_M", "--vram", "16GiB", "--ram", "64GiB",
    "--context", "8K",
  ]
}

///|
fn with_extra(extra : Array[String]) -> Result[@cli.CliOptions, String] {
  let args = basic_args()
  args.append(extra[:])
  @cli.parse_args(args)
}

///|
test "basic options normalize units and use default FP16 KV" {
  let options = match @cli.parse_args(basic_args()) {
    Ok(value) => value
    Err(error) => fail(error)
  }
  assert_eq(options.input.params_b, 27.0)
  assert_eq(options.input.vram_gib, 16.0)
  assert_eq(options.input.ram_gib, 64.0)
  assert_eq(options.input.context, 8192)
  assert_eq(options.input.quant, @core.Q4_K_M)
  assert_eq(options.kv_precision, @core.FP16KV)
  assert_eq(options.policy, @core.default_policy())
  assert_eq(options.architecture, None)
  assert_eq(options.help, false)
  assert_eq(options.json, false)
}

///|
test "equals syntax accepts decimal GB and raw numeric parameters" {
  let options = match
    @cli.parse_args([
      "--params=27", "--quant=iq4_xs", "--vram=16GB", "--ram=64gb", "--context=8192",
      "--json",
    ]) {
    Ok(value) => value
    Err(error) => fail(error)
  }
  assert_eq(options.input.params_b, 27.0)
  assert_eq(options.input.vram_gib, 16000000000.0 / 1073741824.0)
  assert_eq(options.input.ram_gib, 64000000000.0 / 1073741824.0)
  assert_eq(options.input.context, 8192)
  assert_eq(options.input.quant, @core.IQ4_XS)
  assert_eq(options.json, true)
}

///|
test "advanced architecture defaults query heads to KV heads" {
  let options = match
    with_extra([
      "--layers", "64", "--kv-heads", "8", "--head-dim", "128", "--kv", "Q8_0",
    ]) {
    Ok(value) => value
    Err(error) => fail(error)
  }
  guard options.architecture is Some(architecture) else {
    fail("missing explicit architecture")
  }
  assert_eq(architecture.layers, 64)
  assert_eq(architecture.kv_heads, 8)
  assert_eq(architecture.head_dim, 128)
  assert_eq(architecture.query_heads, 8)
  assert_eq(architecture.source, "explicit CLI architecture override")
  assert_eq(options.kv_precision, @core.Q8KV)
}

///|
test "explicit query heads and Q4 KV are retained" {
  let options = match
    with_extra([
      "--layers=64", "--kv-heads=8", "--head-dim=128", "--query-heads=40", "--kv=Q4_0",
    ]) {
    Ok(value) => value
    Err(error) => fail(error)
  }
  guard options.architecture is Some(architecture) else {
    fail("missing explicit architecture")
  }
  assert_eq(architecture.query_heads, 40)
  assert_eq(options.kv_precision, @core.Q4KV)
}

///|
test "help works without input and still validates supplied syntax" {
  guard @cli.parse_args(["--help"]) is Ok(options) else {
    fail("help must work without basic input")
  }
  assert_eq(options.help, true)
  assert_true(@cli.parse_args(["--help", "--unknown"]) is Err(_))
  assert_true(@cli.parse_args(["--help", "--params"]) is Err(_))
  assert_true(@cli.parse_args(["--help", "--params=bad"]) is Err(_))
  assert_true(@cli.parse_args(["--help=yes"]) is Err(_))
}

///|
test "every basic option is required" {
  for omitted in 0..<5 {
    let args = basic_args()
    let reduced : Array[String] = []
    for index, value in args {
      if index / 2 != omitted {
        reduced.push(value)
      }
    }
    assert_true(@cli.parse_args(reduced) is Err(_))
  }
}

///|
test "unknown duplicate positional empty and missing options fail" {
  for
    extra in [
      ["--unknown", "1"],
      ["positional"],
      ["--params", "30B"],
      ["--json", "--json"],
      ["--kv"],
      ["--kv", "--json"],
      ["--kv="],
      ["--json=true"],
    ] {
    assert_true(with_extra(extra) is Err(_))
  }
}

///|
test "partial and invalid architecture overrides fail" {
  for
    extra in [
      ["--layers=64"],
      ["--kv-heads=8", "--head-dim=128"],
      ["--query-heads=40"],
      ["--layers=0", "--kv-heads=8", "--head-dim=128"],
      ["--layers=64", "--kv-heads=0", "--head-dim=128"],
      ["--layers=64", "--kv-heads=8", "--head-dim=0"],
      ["--layers=64", "--kv-heads=8", "--head-dim=128", "--query-heads=4"],
      ["--layers=1.5", "--kv-heads=8", "--head-dim=128"],
    ] {
    assert_true(with_extra(extra) is Err(_))
  }
}

///|
test "numeric bounds malformed units and unknown quantization fail" {
  for
    (option, value) in [
      ("--params", "0B"),
      ("--params", "NaN"),
      ("--params", "inf"),
      ("--params", "not-a-number"),
      ("--params", "1001"),
      ("--quant", "IQ4"),
      ("--quant", "Q3_K_M"),
      ("--vram", "16"),
      ("--vram", "-1GiB"),
      ("--ram", "NaNGiB"),
      ("--context", "0"),
      ("--context", "1.5K"),
      ("--context", "1025K"),
      ("--context", "2147483648"),
    ] {
    let args = basic_args()
    for index in 0..<args.length() {
      if args[index] == option {
        args[index + 1] = value
      }
    }
    assert_true(@cli.parse_args(args) is Err(_))
  }
  assert_true(with_extra(["--kv", "BF16"]) is Err(_))
}

///|
test "policy overrides preserve units and disclose a custom policy" {
  let options = match
    with_extra([
      "--gpu-runtime=1", "--gpu-margin=2GiB", "--host-runtime=3GB", "--os-reserve=0",
      "--loading-workspace=0.5GiB", "--gpu-staging=0.125",
    ]) {
    Ok(value) => value
    Err(error) => fail(error)
  }
  assert_eq(options.policy.gpu_runtime_gib, 1.0)
  assert_eq(options.policy.gpu_margin_gib, 2.0)
  assert_eq(options.policy.host_runtime_gib, 3000000000.0 / 1073741824.0)
  assert_eq(options.policy.os_reserve_gib, 0.0)
  assert_eq(options.policy.loading_workspace_gib, 0.5)
  assert_eq(options.policy.gpu_staging_gib, 0.125)
  assert_true(options.policy.id.has_prefix("custom CLI overrides"))
  assert_true(options.policy.id.contains("--os-reserve"))
}

///|
test "zero OS reserve supports capacities already reported as free RAM" {
  let options = match with_extra(["--os-reserve=0"]) {
    Ok(value) => value
    Err(error) => fail(error)
  }
  let report = match
    @core.calculate(
      options.input,
      options.architecture,
      options.kv_precision,
      options.policy,
    ) {
    Ok(value) => value
    Err(error) => fail(error)
  }
  guard report.fit.ledger is Some(ledger) else {
    fail("expected a feasible placement")
  }
  assert_eq(ledger.os_reserve_bytes, 0.0)
  assert_eq(report.policy.os_reserve_gib, 0.0)
  assert_true(report.policy.id.contains("custom CLI overrides"))
}

///|
test "policy overrides reject negative nonfinite malformed and duplicate values" {
  for
    option in [
      "--gpu-runtime", "--gpu-margin", "--host-runtime", "--os-reserve", "--loading-workspace",
      "--gpu-staging",
    ] {
    for value in ["-1", "NaN", "inf", "1000001", "bad", "1MB"] {
      assert_true(with_extra([option, value]) is Err(_))
    }
    assert_true(with_extra([option, "1", option + "=2"]) is Err(_))
  }
}

///|
fn sample_report() -> @core.Report raise {
  let options = match @cli.parse_args(basic_args()) {
    Ok(value) => value
    Err(error) => fail(error)
  }
  match
    @core.calculate(
      options.input,
      options.architecture,
      options.kv_precision,
      options.policy,
    ) {
    Ok(value) => value
    Err(error) => fail(error)
  }
}

///|
test "JSON report has canonical names all budgets and sensitivity weights" {
  let report = sample_report()
  let json = @json.parse(@cli.render_json(report))
  guard json is Object(fields) else { fail("expected JSON object") }
  assert_eq(fields.get("schema_version"), Some(Json("1")))
  guard fields.get("input") is Some(Object(input)) else {
    fail("missing input")
  }
  assert_eq(input.get("quant"), Some(Json("Q4_K_M")))
  assert_eq(input.get("context_tokens"), Some(Json(8192)))
  guard fields.get("fit") is Some(Object(fit)) else { fail("missing fit") }
  assert_eq(
    fit.get("status"),
    Some(Json(@core.fit_status_name(report.fit.status))),
  )
  guard fit.get("ledger") is Some(Object(ledger)) else {
    fail("missing feasible ledger")
  }
  for
    key in [
      "gpu_layers", "cpu_layers", "gpu_weights_bytes", "gpu_kv_bytes", "gpu_runtime_bytes",
      "gpu_margin_bytes", "gpu_staging_bytes", "host_weights_bytes", "host_kv_bytes",
      "host_runtime_bytes", "os_reserve_bytes", "loading_workspace_bytes", "gpu_steady_bytes",
      "gpu_loading_peak_bytes", "host_steady_bytes", "host_loading_peak_bytes",
    ] {
    assert_true(ledger.get(key) is Some(Number(_, ..)))
  }
  guard fields.get("sensitivity") is Some(Array(scenarios)) else {
    fail("missing sensitivity")
  }
  assert_eq(scenarios.length(), 3)
  for scenario in scenarios {
    guard scenario is Object(scenario) else { fail("expected scenario object") }
    assert_true(scenario.get("weights_bytes") is Some(Number(_, ..)))
    assert_true(scenario.get("architecture") is Some(Object(_)))
    assert_true(scenario.get("fit") is Some(Object(_)))
  }
}

///|
test "insufficient JSON output uses a null ledger" {
  let input = { ..sample_report().input, vram_gib: 0.0, ram_gib: 0.0, }
  let report = match
    @core.calculate(input, None, @core.FP16KV, @core.default_policy()) {
    Ok(value) => value
    Err(error) => fail(error)
  }
  guard @json.parse(@cli.render_json(report)) is Object(fields) else {
    fail("expected JSON object")
  }
  guard fields.get("fit") is Some(Object(fit)) else { fail("missing fit") }
  assert_eq(fit.get("status"), Some(Json("INSUFFICIENT")))
  assert_eq(fit.get("ledger"), Some(Json::null()))
}

///|
test "JSON escapes architecture sources and text exposes assumptions" {
  let original = sample_report()
  let source = "explicit \"quoted\" source \\ unicode 架构"
  let report = {
    ..original,
    architecture: { ..original.architecture, source, },
  }
  guard @json.parse(@cli.render_json(report)) is Object(fields) else {
    fail("expected JSON object")
  }
  guard fields.get("architecture") is Some(Object(architecture)) else {
    fail("missing architecture")
  }
  assert_eq(architecture.get("source"), Some(Json(source)))
  let text = @cli.render_text(report)
  for
    detail in [
      "GPU steady:", "loading peak:", "RAM steady:", "Weight policy:", "Architecture mode:",
      "Source:", "uncalibrated", "uniformly", "not guaranteed runtime combinations",
      "K=1024",
    ] {
    assert_true(text.contains(detail[:]))
  }
}
