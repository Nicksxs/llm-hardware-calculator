///|
/// Weight recipes; whole-model storage estimates are separate from block costs.
pub(all) enum WeightQuant {
  FP16
  BF16
  Q8_0
  Q6_K
  Q5_K_M
  Q4_K_M
  IQ4_XS
  IQ4_NL
} derive(Eq, Debug)

///|
pub extend WeightQuant with Eq::{equal, not_equal}

///|
pub extend WeightQuant with @debug.Debug::{to_repr}

///|
/// Parameters are decimal billions; memory capacities are binary GiB.
pub(all) struct BasicInput {
  params_b : Double
  quant : WeightQuant
  vram_gib : Double
  ram_gib : Double
  context : Int
} derive(Eq, Debug)

///|
pub extend BasicInput with Eq::{equal, not_equal}

///|
pub extend BasicInput with @debug.Debug::{to_repr}

///|
pub fn gib_to_bytes(gib : Double) -> Double {
  gib * 1073741824.0
}

///|
pub fn bytes_to_gib(bytes : Double) -> Double {
  bytes / 1073741824.0
}

///|
/// Explicit bounds also reject NaN and infinities through ordered comparisons.
pub fn validate_input(input : BasicInput) -> Result[BasicInput, String] {
  if !(input.params_b > 0.0 && input.params_b <= 1000.0) {
    return Err("params must be finite and in (0, 1000] decimal billions")
  }
  if !(input.vram_gib >= 0.0 && input.vram_gib <= 1000000.0) {
    return Err("vram must be finite and in [0, 1000000] GiB")
  }
  if !(input.ram_gib >= 0.0 && input.ram_gib <= 1000000.0) {
    return Err("ram must be finite and in [0, 1000000] GiB")
  }
  if input.context < 1 || input.context > 1048576 {
    return Err("context must be in [1, 1048576] tokens")
  }
  Ok(input)
}
