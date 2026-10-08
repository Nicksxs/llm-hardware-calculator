///|
test "weight block payload uses scales rather than nominal bit count" {
  // Independent byte-size fixtures for one decimal billion parameters.
  let fixtures = [
    (@core.Q4_K_M, 562500000.0, 606250000.0, 650000000.0),
    (@core.Q5_K_M, 687500000.0, 718750000.0, 762500000.0),
    (@core.Q6_K, 820312500.0, 831250000.0, 862500000.0),
    (@core.Q8_0, 1062500000.0, 1075000000.0, 1100000000.0),
    (@core.IQ4_XS, 531250000.0, 575000000.0, 637500000.0),
    (@core.IQ4_NL, 562500000.0, 587500000.0, 650000000.0),
  ]
  for (quant, expected, expected_planning, expected_high) in fixtures {
    let actual = @core.estimate_weights(1.0, quant).unwrap()
    assert_eq(actual.baseline_bytes, expected)
    assert_eq(actual.low_bytes, expected)
    assert_true((actual.planning_bytes - expected_planning).abs() < 0.01)
    assert_true((actual.high_bytes - expected_high).abs() < 0.01)
  }
}

///|
test "27B Q4 baseline and planning policy remain distinct" {
  let actual = @core.estimate_weights(27.0, @core.Q4_K_M).unwrap()
  assert_eq(actual.baseline_bytes, 15187500000.0)
  assert_eq(actual.low_bytes, 15187500000.0)
  assert_true((actual.planning_bytes - 16368750000.0).abs() < 0.01)
  assert_true((actual.high_bytes - 17550000000.0).abs() < 0.01)
  assert_eq(actual.baseline_bpw, 4.5)
  assert_eq(actual.effective_bpw, 4.85)
  assert_true(actual.policy.contains("uncalibrated"))
  assert_true(actual.policy.contains("not a confidence interval"))
  assert_true(@core.bytes_to_gib(actual.baseline_bytes) > 14.14)
  assert_true(@core.bytes_to_gib(actual.baseline_bytes) < 14.15)
}

///|
test "FP16 and BF16 store two bytes per parameter without quant uplift" {
  let fp16 = @core.estimate_weights(7.0, @core.FP16).unwrap()
  let bf16 = @core.estimate_weights(7.0, @core.BF16).unwrap()
  assert_eq(fp16, bf16)
  assert_eq(fp16.baseline_bytes, 14000000000.0)
  assert_eq(fp16.low_bytes, fp16.planning_bytes)
  assert_eq(fp16.high_bytes, fp16.planning_bytes)
}

///|
test "planning ranges are ordered and estimates scale with parameter count" {
  let recipes = [
    @core.FP16,
    @core.BF16,
    @core.Q8_0,
    @core.Q6_K,
    @core.Q5_K_M,
    @core.Q4_K_M,
    @core.IQ4_XS,
    @core.IQ4_NL,
  ]
  for quant in recipes {
    let small = @core.estimate_weights(7.0, quant).unwrap()
    let large = @core.estimate_weights(14.0, quant).unwrap()
    assert_true(small.baseline_bytes <= small.low_bytes)
    assert_true(small.low_bytes <= small.planning_bytes)
    assert_true(small.planning_bytes <= small.high_bytes)
    assert_eq(large.baseline_bytes, small.baseline_bytes * 2.0)
    assert_eq(large.planning_bytes, small.planning_bytes * 2.0)
    assert_eq(large.high_bytes, small.high_bytes * 2.0)
  }
}

///|
test "quant names round trip and aliases resolve to canonical recipes" {
  let recipes = [
    @core.FP16,
    @core.BF16,
    @core.Q8_0,
    @core.Q6_K,
    @core.Q5_K_M,
    @core.Q4_K_M,
    @core.IQ4_XS,
    @core.IQ4_NL,
  ]
  for quant in recipes {
    assert_eq(
      @core.parse_weight_quant(@core.weight_quant_name(quant)),
      Ok(quant),
    )
  }
  assert_eq(@core.parse_weight_quant("q4"), Ok(@core.Q4_K_M))
  assert_eq(@core.parse_weight_quant("q5"), Ok(@core.Q5_K_M))
  assert_eq(@core.parse_weight_quant("q6"), Ok(@core.Q6_K))
  assert_eq(@core.parse_weight_quant("q8"), Ok(@core.Q8_0))
  assert_eq(@core.parse_weight_quant("f16"), Ok(@core.FP16))
  assert_eq(@core.parse_weight_quant("iQ4_xS"), Ok(@core.IQ4_XS))
}

///|
test "unknown and ambiguous quantization strings are rejected" {
  assert_true(@core.parse_weight_quant("iq4") is Err(_))
  assert_true(@core.parse_weight_quant("q3_k_m") is Err(_))
  assert_true(@core.parse_weight_quant("") is Err(_))
  assert_true(@core.parse_weight_quant("q4_k_m_extra") is Err(_))
}

///|
test "weight estimation rejects non-finite and out-of-range parameter counts" {
  let invalid = [0.0, -1.0, 1001.0, 0.0 / 0.0, 1.0 / 0.0, -1.0 / 0.0]
  for params_b in invalid {
    assert_true(@core.estimate_weights(params_b, @core.Q4_K_M) is Err(_))
  }
  assert_true(@core.estimate_weights(0.25, @core.Q4_K_M) is Ok(_))
  assert_true(@core.estimate_weights(1000.0, @core.Q4_K_M) is Ok(_))
}
