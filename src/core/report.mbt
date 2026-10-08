///|
pub(all) struct SensitivityResult {
  label : String
  architecture : Architecture
  weights_bytes : Double
  kv_bytes : Double
  fit : FitResult
} derive(Eq, Debug)

///|
pub extend SensitivityResult with Eq::{equal, not_equal}

///|
pub extend SensitivityResult with @debug.Debug::{to_repr}

///|
/// All assumptions and intermediate values accompany the placement decision.
pub(all) struct Report {
  input : BasicInput
  architecture : Architecture
  architecture_mode : String
  weights : WeightEstimate
  kv_bytes : Double
  kv_precision : KvPrecision
  policy : BudgetPolicy
  fit : FitResult
  sensitivity : Array[SensitivityResult]
} derive(Eq, Debug)

///|
pub extend Report with Eq::{equal, not_equal}

///|
pub extend Report with @debug.Debug::{to_repr}

///|
pub fn calculate(
  input : BasicInput,
  architecture : Architecture?,
  kv_precision : KvPrecision,
  policy : BudgetPolicy,
) -> Result[Report, String] {
  match validate_input(input) {
    Err(message) => return Err(message)
    Ok(_) => ()
  }
  let shape = match architecture {
    Some(explicit) =>
      match validate_architecture(explicit) {
        Err(message) => return Err(message)
        Ok(valid) => valid
      }
    None =>
      match proxy_architecture(input.params_b) {
        Err(message) => return Err(message)
        Ok(proxy) => proxy
      }
  }
  let weights = match estimate_weights(input.params_b, input.quant) {
    Err(message) => return Err(message)
    Ok(value) => value
  }
  let kv_bytes = match estimate_kv(input.context, shape, kv_precision) {
    Err(message) => return Err(message)
    Ok(value) => value
  }
  let fit = match
    search_fit(
      weights.planning_bytes,
      kv_bytes,
      shape.layers,
      input.vram_gib,
      input.ram_gib,
      policy,
    ) {
    Err(message) => return Err(message)
    Ok(value) => value
  }
  let sensitivity : Array[SensitivityResult] = []
  for
    (label, weight_bytes) in [
      ("weight-policy-low (not a confidence bound)", weights.low_bytes),
      ("weight-policy-high (not a confidence bound)", weights.high_bytes),
    ] {
    let scenario_fit = match
      search_fit(
        weight_bytes,
        kv_bytes,
        shape.layers,
        input.vram_gib,
        input.ram_gib,
        policy,
      ) {
      Err(message) => return Err(message)
      Ok(value) => value
    }
    sensitivity.push({
      label,
      architecture: shape,
      weights_bytes: weight_bytes,
      kv_bytes,
      fit: scenario_fit,
    })
  }
  if shape.query_heads > shape.kv_heads {
    let mha = {
      ..shape,
      kv_heads: shape.query_heads,
      source: shape.source +
      " | hypothetical MHA sensitivity with KV heads = query heads",
    }
    let mha_kv = match estimate_kv(input.context, mha, kv_precision) {
      Err(message) => return Err(message)
      Ok(value) => value
    }
    let mha_fit = match
      search_fit(
        weights.planning_bytes,
        mha_kv,
        mha.layers,
        input.vram_gib,
        input.ram_gib,
        policy,
      ) {
      Err(message) => return Err(message)
      Ok(value) => value
    }
    sensitivity.push({
      label: "MHA-structure (hypothetical; not architecture detection)",
      architecture: mha,
      weights_bytes: weights.planning_bytes,
      kv_bytes: mha_kv,
      fit: mha_fit,
    })
  }
  Ok({
    input,
    architecture: shape,
    architecture_mode: if architecture is Some(_) {
      "Advanced supplied structure"
    } else {
      "Basic upper-size-bucket proxy (not model detection)"
    },
    weights,
    kv_bytes,
    kv_precision,
    policy,
    fit,
    sensitivity,
  })
}
