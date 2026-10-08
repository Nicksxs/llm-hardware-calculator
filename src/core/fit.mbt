///|
/// Classification under the selected capacity-planning policies.
pub(all) enum FitStatus {
  FullGPU
  Offload
  CPUOnly
  Insufficient
} derive(Eq, Debug)

///|
pub extend FitStatus with Eq::{equal, not_equal}

///|
pub extend FitStatus with @debug.Debug::{to_repr}

///|
pub(all) struct FitResult {
  status : FitStatus
  ledger : MemoryLedger?
  reason : String
} derive(Eq, Debug)

///|
pub extend FitResult with Eq::{equal, not_equal}

///|
pub extend FitResult with @debug.Debug::{to_repr}

///|
pub fn fit_status_name(status : FitStatus) -> String {
  match status {
    FullGPU => "FULL_GPU"
    Offload => "GPU+CPU OFFLOAD"
    CPUOnly => "CPU ONLY"
    Insufficient => "INSUFFICIENT"
  }
}

///|
/// Search integer layer placements from most GPU-resident to CPU-only. Both
/// steady-state and loading peaks must fit each device independently. This is
/// a uniform-layer capacity estimate, not an exact backend placement plan.
pub fn search_fit(
  weights_bytes : Double,
  kv_bytes : Double,
  layers : Int,
  vram_gib : Double,
  ram_gib : Double,
  policy : BudgetPolicy,
) -> Result[FitResult, String] {
  if !(vram_gib >= 0.0 && vram_gib <= 1000000.0) {
    return Err("vram must be finite and in [0, 1000000] GiB")
  }
  if !(ram_gib >= 0.0 && ram_gib <= 1000000.0) {
    return Err("ram must be finite and in [0, 1000000] GiB")
  }
  // Validate even when an invalid layer count would skip the search loop.
  match build_ledger(weights_bytes, kv_bytes, layers, 0, policy) {
    Err(message) => return Err(message)
    Ok(_) => ()
  }
  let vram_bytes = gib_to_bytes(vram_gib)
  let ram_bytes = gib_to_bytes(ram_gib)
  let assumptions = "uniform-layer allocation and full-host-weights-loading-policy; " +
    policy.id
  for gpu_layers = layers; gpu_layers >= 0; gpu_layers = gpu_layers - 1 {
    let ledger = match
      build_ledger(weights_bytes, kv_bytes, layers, gpu_layers, policy) {
      Err(message) => return Err(message)
      Ok(value) => value
    }
    if ledger.gpu_steady_bytes <= vram_bytes &&
      ledger.gpu_loading_peak_bytes <= vram_bytes &&
      ledger.host_steady_bytes <= ram_bytes &&
      ledger.host_loading_peak_bytes <= ram_bytes {
      let status = if gpu_layers == layers {
        FullGPU
      } else if gpu_layers == 0 {
        CPUOnly
      } else {
        Offload
      }
      return Ok({
        status,
        ledger: Some(ledger),
        reason: "Estimated " +
        fit_status_name(status) +
        " fits separate GPU/RAM steady-state and loading budgets under " +
        assumptions,
      })
    }
  }
  Ok({
    status: Insufficient,
    ledger: None,
    reason: "No integer-layer placement fits separate GPU/RAM steady-state and loading budgets under " +
    assumptions +
    ". This policy result does not establish that every backend configuration is unable to run.",
  })
}
