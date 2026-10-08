///|
/// Capacity-planning allowances, not measured runtime requirements.
pub(all) struct BudgetPolicy {
  gpu_runtime_gib : Double
  gpu_margin_gib : Double
  host_runtime_gib : Double
  os_reserve_gib : Double
  loading_workspace_gib : Double
  gpu_staging_gib : Double
  id : String
} derive(Eq, Debug)

///|
pub extend BudgetPolicy with Eq::{equal, not_equal}

///|
pub extend BudgetPolicy with @debug.Debug::{to_repr}

///|
pub fn default_policy() -> BudgetPolicy {
  {
    gpu_runtime_gib: 0.5,
    gpu_margin_gib: 1.0,
    host_runtime_gib: 1.0,
    os_reserve_gib: 4.0,
    loading_workspace_gib: 1.0,
    gpu_staging_gib: 0.25,
    id: "capacity-policy-v1 (uncalibrated; full host weights retained during loading)",
  }
}

///|
pub fn validate_policy(policy : BudgetPolicy) -> Result[BudgetPolicy, String] {
  for
    allowance in [
      policy.gpu_runtime_gib,
      policy.gpu_margin_gib,
      policy.host_runtime_gib,
      policy.os_reserve_gib,
      policy.loading_workspace_gib,
      policy.gpu_staging_gib,
    ] {
    if !(allowance >= 0.0 && allowance <= 1000000.0) {
      return Err("policy allowances must be finite and in [0, 1000000] GiB")
    }
  }
  Ok(policy)
}

///|
/// Uniform-layer split with a separate, conservative startup ledger.
pub(all) struct MemoryLedger {
  gpu_layers : Int
  cpu_layers : Int
  gpu_weights_bytes : Double
  gpu_kv_bytes : Double
  gpu_runtime_bytes : Double
  gpu_margin_bytes : Double
  gpu_staging_bytes : Double
  host_weights_bytes : Double
  host_kv_bytes : Double
  host_runtime_bytes : Double
  os_reserve_bytes : Double
  loading_workspace_bytes : Double
  gpu_steady_bytes : Double
  gpu_loading_peak_bytes : Double
  host_steady_bytes : Double
  host_loading_peak_bytes : Double
} derive(Eq, Debug)

///|
pub extend MemoryLedger with Eq::{equal, not_equal}

///|
pub extend MemoryLedger with @debug.Debug::{to_repr}

///|
pub fn build_ledger(
  weights_bytes : Double,
  kv_bytes : Double,
  layers : Int,
  gpu_layers : Int,
  policy : BudgetPolicy,
) -> Result[MemoryLedger, String] {
  if !(weights_bytes > 0.0 && weights_bytes <= gib_to_bytes(1000000.0)) {
    return Err(
      "weight bytes must be finite and positive within supported bounds",
    )
  }
  if !(kv_bytes >= 0.0 && kv_bytes <= gib_to_bytes(1000000.0)) {
    return Err(
      "KV bytes must be finite and nonnegative within supported bounds",
    )
  }
  if layers < 1 || layers > 512 || gpu_layers < 0 || gpu_layers > layers {
    return Err("layers must be in [1, 512] and GPU layers in [0, layers]")
  }
  match validate_policy(policy) {
    Err(message) => return Err(message)
    Ok(_) => ()
  }
  let fraction = gpu_layers.to_double() / layers.to_double()
  let gpu_weights = weights_bytes * fraction
  let gpu_kv = kv_bytes * fraction
  let host_weights = weights_bytes - gpu_weights
  let host_kv = kv_bytes - gpu_kv
  let gpu_runtime = if gpu_layers > 0 {
    gib_to_bytes(policy.gpu_runtime_gib)
  } else {
    0.0
  }
  let gpu_margin = if gpu_layers > 0 {
    gib_to_bytes(policy.gpu_margin_gib)
  } else {
    0.0
  }
  let gpu_staging = if gpu_layers > 0 {
    gib_to_bytes(policy.gpu_staging_gib)
  } else {
    0.0
  }
  let host_runtime = gib_to_bytes(policy.host_runtime_gib)
  let os_reserve = gib_to_bytes(policy.os_reserve_gib)
  let workspace = gib_to_bytes(policy.loading_workspace_gib)
  let gpu_steady = gpu_weights + gpu_kv + gpu_runtime + gpu_margin
  let host_steady = host_weights + host_kv + host_runtime + os_reserve
  Ok({
    gpu_layers,
    cpu_layers: layers - gpu_layers,
    gpu_weights_bytes: gpu_weights,
    gpu_kv_bytes: gpu_kv,
    gpu_runtime_bytes: gpu_runtime,
    gpu_margin_bytes: gpu_margin,
    gpu_staging_bytes: gpu_staging,
    host_weights_bytes: host_weights,
    host_kv_bytes: host_kv,
    host_runtime_bytes: host_runtime,
    os_reserve_bytes: os_reserve,
    loading_workspace_bytes: workspace,
    gpu_steady_bytes: gpu_steady,
    gpu_loading_peak_bytes: gpu_steady + gpu_staging,
    host_steady_bytes: host_steady,
    // Replaces host-resident weights with the full weight copy, rather than
    // adding a second full copy to steady-state weights.
    host_loading_peak_bytes: weights_bytes +
    host_kv +
    host_runtime +
    os_reserve +
    workspace,
  })
}
