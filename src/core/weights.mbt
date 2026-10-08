///|
/// Tensor block payload and an explicitly uncalibrated whole-model planning
/// allowance. Bytes exclude GGUF metadata and runtime allocations. The range
/// describes policy scenarios, not a statistical confidence interval.
pub(all) struct WeightEstimate {
  baseline_bytes : Double
  planning_bytes : Double
  low_bytes : Double
  high_bytes : Double
  baseline_bpw : Double
  effective_bpw : Double
  policy : String
} derive(Eq, Debug)

///|
pub extend WeightEstimate with Eq::{equal, not_equal}

///|
pub extend WeightEstimate with @debug.Debug::{to_repr}

///|
/// Canonical recipe names are also used by reports and JSON output.
pub fn weight_quant_name(quant : WeightQuant) -> String {
  match quant {
    FP16 => "FP16"
    BF16 => "BF16"
    Q8_0 => "Q8_0"
    Q6_K => "Q6_K"
    Q5_K_M => "Q5_K_M"
    Q4_K_M => "Q4_K_M"
    IQ4_XS => "IQ4_XS"
    IQ4_NL => "IQ4_NL"
  }
}

///|
/// Short Q4/Q5/Q6/Q8 names resolve to a disclosed GGUF recipe. IQ4 is ambiguous
/// and requires selecting XS or NL explicitly.
pub fn parse_weight_quant(value : String) -> Result[WeightQuant, String] {
  match value.to_lower() {
    "fp16" | "f16" => Ok(FP16)
    "bf16" => Ok(BF16)
    "q8" | "q8_0" => Ok(Q8_0)
    "q6" | "q6_k" => Ok(Q6_K)
    "q5" | "q5_k_m" => Ok(Q5_K_M)
    "q4" | "q4_k_m" => Ok(Q4_K_M)
    "iq4_xs" => Ok(IQ4_XS)
    "iq4_nl" => Ok(IQ4_NL)
    "iq4" => Err("IQ4 is ambiguous; select iq4_xs or iq4_nl")
    _ => Err("unknown weight quantization: " + value)
  }
}

///|
/// Parameters are decimal billions. Baseline BPW comes from GGML block storage:
/// Q4_K=144/256 bytes, Q5_K=176/256, Q6_K=210/256, Q8_0=34/32,
/// IQ4_XS=136/256, IQ4_NL=18/32. Q4_K_M/Q5_K_M and IQ recipes can
/// contain higher-precision tensors, so planning BPW is a separate project
/// policy. These v1 allowances are not measured error bounds.
pub fn estimate_weights(
  params_b : Double,
  quant : WeightQuant,
) -> Result[WeightEstimate, String] {
  if !(params_b > 0.0 && params_b <= 1000.0) {
    return Err("params must be finite and in (0, 1000] decimal billions")
  }
  // Tuple fields: baseline, planning, low, high bits per parameter.
  let (baseline, planning, low, high) = match quant {
    FP16 | BF16 => (16.0, 16.0, 16.0, 16.0)
    Q8_0 => (8.5, 8.6, 8.5, 8.8)
    Q6_K => (6.5625, 6.65, 6.5625, 6.9)
    Q5_K_M => (5.5, 5.75, 5.5, 6.1)
    Q4_K_M => (4.5, 4.85, 4.5, 5.2)
    IQ4_XS => (4.25, 4.6, 4.25, 5.1)
    IQ4_NL => (4.5, 4.7, 4.5, 5.2)
  }
  let bytes_per_bpw = params_b * 125000000.0
  Ok({
    baseline_bytes: bytes_per_bpw * baseline,
    planning_bytes: bytes_per_bpw * planning,
    low_bytes: bytes_per_bpw * low,
    high_bytes: bytes_per_bpw * high,
    baseline_bpw: baseline,
    effective_bpw: planning,
    policy: "planning-policy-v1 (uncalibrated; excludes metadata and runtime; range is not a confidence interval)",
  })
}
