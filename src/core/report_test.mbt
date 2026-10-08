///|
fn report_input() -> @core.BasicInput {
  {
    params_b: 27.0,
    quant: @core.Q4_K_M,
    vram_gib: 16.0,
    ram_gib: 64.0,
    context: 8192,
  }
}

///|
test "Basic report retains all inputs proxy sources and sensitivity assumptions" {
  let input = report_input()
  let report = @core.calculate(
    input,
    None,
    @core.FP16KV,
    @core.default_policy(),
  ).unwrap()
  assert_eq(report.input, input)
  assert_true(report.architecture_mode.contains("proxy"))
  assert_true(
    report.architecture.source.contains(
      "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    ),
  )
  assert_eq(report.kv_bytes, @core.gib_to_bytes(2.0))
  assert_eq(report.sensitivity.length(), 3)
  assert_eq(report.sensitivity[0].weights_bytes, report.weights.low_bytes)
  assert_eq(report.sensitivity[1].weights_bytes, report.weights.high_bytes)
  assert_eq(report.sensitivity[2].kv_bytes, @core.gib_to_bytes(10.0))
}

///|
test "Advanced bypasses proxy range but validates quantized row alignment" {
  let arch : @core.Architecture = {
    layers: 80,
    kv_heads: 8,
    query_heads: 64,
    head_dim: 128,
    source: "explicit test structure",
  }
  let large = { ..report_input(), params_b: 100.0, ram_gib: 512.0, }
  assert_true(
    @core.calculate(large, None, @core.FP16KV, @core.default_policy()) is Err(_),
  )
  let report = @core.calculate(
    large,
    Some(arch),
    @core.Q8KV,
    @core.default_policy(),
  ).unwrap()
  assert_eq(report.architecture, arch)
  assert_true(report.architecture_mode.contains("Advanced"))
  assert_true(
    @core.calculate(
      large,
      Some({ ..arch, head_dim: 33, }),
      @core.Q8KV,
      @core.default_policy(),
    )
    is Err(_),
  )
}

///|
test "integration matrix preserves independent budgets and allocation totals" {
  let cases = [
    (7.0, 8.0),
    (14.0, 12.0),
    (27.0, 16.0),
    (32.0, 24.0),
    (70.0, 24.0),
  ]
  for (params_b, vram_gib) in cases {
    let mut previous_kv = 0.0
    for context in [4096, 8192, 32768, 65536] {
      let input = { ..report_input(), params_b, vram_gib, context, }
      let report = @core.calculate(
        input,
        None,
        @core.FP16KV,
        @core.default_policy(),
      ).unwrap()
      assert_true(report.kv_bytes > previous_kv)
      previous_kv = report.kv_bytes
      if report.fit.ledger is Some(ledger) {
        assert_true(ledger.gpu_steady_bytes <= @core.gib_to_bytes(vram_gib))
        assert_true(
          ledger.gpu_loading_peak_bytes <= @core.gib_to_bytes(vram_gib),
        )
        assert_true(
          ledger.host_steady_bytes <= @core.gib_to_bytes(input.ram_gib),
        )
        assert_true(
          ledger.host_loading_peak_bytes <= @core.gib_to_bytes(input.ram_gib),
        )
        assert_true(
          (ledger.gpu_weights_bytes +
          ledger.host_weights_bytes -
          report.weights.planning_bytes).abs() <
          0.01,
        )
        assert_eq(ledger.gpu_kv_bytes + ledger.host_kv_bytes, report.kv_bytes)
      } else {
        assert_eq(report.fit.status, @core.Insufficient)
      }
    }
  }
}

///|
test "invalid input and policy propagate to report errors" {
  assert_true(
    @core.calculate(
      { ..report_input(), context: 0, },
      None,
      @core.FP16KV,
      @core.default_policy(),
    )
    is Err(_),
  )
  let bad_policy = { ..@core.default_policy(), os_reserve_gib: -1.0, }
  assert_true(
    @core.calculate(report_input(), None, @core.FP16KV, bad_policy) is Err(_),
  )
}
