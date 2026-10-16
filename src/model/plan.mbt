///|
fn validate_profile(profile : Profile) -> Result[Unit, String] {
  match @core.validate_architecture(profile.shape) {
    Err(message) => return Err(message)
    Ok(_) => ()
  }
  if (profile.architecture != "llama" && profile.architecture != "qwen2") ||
    profile.backend_revision != backend_revision() ||
    profile.layer_bytes.length() != profile.shape.layers ||
    profile.input_bytes == 0UL {
    return Err("invalid profile architecture, revision or layer inventory")
  }
  let mut feed_forward = 0UL
  for tensor in profile.tensors {
    if tensor.name == "blk.0.ffn_gate.weight" && tensor.dimensions.length() == 2 {
      feed_forward = tensor.dimensions[1]
    }
  }
  if feed_forward == 0UL || feed_forward > 1048576UL {
    return Err("profile has no supported feed-forward tensor shape")
  }
  // Profile is publicly constructible. Re-derive its grouping, shape checks and
  // storage sizes from retained raw descriptors instead of trusting mutable
  // summary fields or caller-supplied role/block labels.
  let prefix = profile.architecture + "."
  let metadata : Array[@gguf.Metadata] = [
    {
      key: "general.architecture",
      value: @gguf.StringValue(profile.architecture),
    },
    {
      key: prefix + "block_count",
      value: @gguf.UIntValue(profile.shape.layers.to_uint64()),
    },
    {
      key: prefix + "embedding_length",
      value: @gguf.UIntValue(profile.embedding_length.to_uint64()),
    },
    {
      key: prefix + "context_length",
      value: @gguf.UIntValue(profile.context_length.to_uint64()),
    },
    {
      key: prefix + "attention.head_count",
      value: @gguf.UIntValue(profile.shape.query_heads.to_uint64()),
    },
    {
      key: prefix + "attention.head_count_kv",
      value: @gguf.UIntValue(profile.shape.kv_heads.to_uint64()),
    },
    {
      key: prefix + "attention.key_length",
      value: @gguf.UIntValue(profile.shape.head_dim.to_uint64()),
    },
    {
      key: prefix + "attention.value_length",
      value: @gguf.UIntValue(profile.shape.head_dim.to_uint64()),
    },
    {
      key: prefix + "feed_forward_length",
      value: @gguf.UIntValue(feed_forward),
    },
  ]
  let tensors : Array[@gguf.Tensor] = profile.tensors.map(tensor => {
    name: tensor.name,
    dimensions: tensor.dimensions,
    type_id: tensor.type_id,
    element_count: tensor.element_count,
    byte_size: Some(tensor.byte_size),
    offset: 0UL,
    absolute_offset: 0UL,
  })
  let reconstructed : @gguf.Model = {
    version: 3,
    file_size: 0UL,
    data_offset: 0UL,
    metadata,
    tensors,
    complete: true,
    warnings: [],
  }
  let derived = match extract_profile(reconstructed) {
    Ok(value) => value
    Err(message) =>
      return Err("profile raw descriptors are inconsistent: " + message)
  }
  if derived.shape != profile.shape ||
    derived.layer_bytes != profile.layer_bytes ||
    derived.tensor_payload_bytes != profile.tensor_payload_bytes ||
    derived.parameter_count != profile.parameter_count ||
    derived.input_bytes != profile.input_bytes ||
    derived.output_bytes != profile.output_bytes ||
    derived.tied_output != profile.tied_output ||
    derived.tensors != profile.tensors {
    return Err(
      "profile summary, grouping or structure disagrees with raw tensor descriptors",
    )
  }
  Ok(())
}

///|
/// Single-device payload placement for the pinned output-first llama.cpp
/// revision. n_gpu_layers includes the output slot: 1 offloads output only;
/// L+1 offloads every repeating block plus output. Embedding remains on CPU.
pub fn plan(
  profile : Profile,
  context : Int,
  precision : @core.KvPrecision,
  n_gpu_layers : Int,
  policy : @core.BudgetPolicy,
) -> Result[Plan, String] {
  match validate_profile(profile) {
    Ok(_) => ()
    Err(m) => return Err(m)
  }
  match @core.validate_policy(policy) {
    Ok(_) => ()
    Err(m) => return Err(m)
  }
  let layers = profile.shape.layers
  if n_gpu_layers < 0 || n_gpu_layers > layers + 1 {
    return Err(
      "n_gpu_layers must be in [0, block_count+1], including the output slot",
    )
  }
  let kv_bytes = match @core.estimate_kv(context, profile.shape, precision) {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  // The pinned backend pads n_ctx and single-sequence n_ctx_seq to 256 tokens.
  // estimate_kv above validates context <= 1,048,576, so this addition is safe
  // and the largest allocated context remains inside the same core domain.
  let allocated_context = (context + 255) / 256 * 256
  let allocated_kv_bytes = match
    @core.estimate_kv(allocated_context, profile.shape, precision) {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let output_on_gpu = n_gpu_layers > 0
  let gpu_block_count = if output_on_gpu { n_gpu_layers - 1 } else { 0 }
  let first_gpu_block = layers + 1 - n_gpu_layers
  let placements : Array[TensorPlacement] = []
  let mut gpu_weight_bytes = 0UL
  let mut cpu_weight_bytes = 0UL
  for tensor in profile.tensors {
    let on_gpu = match (tensor.role, tensor.block) {
      ("output", _) => output_on_gpu
      ("block", Some(index)) => index >= first_gpu_block
      _ => false
    }
    if on_gpu {
      gpu_weight_bytes = match exact_add(gpu_weight_bytes, tensor.byte_size) {
        Ok(v) => v
        Err(m) => return Err(m)
      }
    } else {
      cpu_weight_bytes = match exact_add(cpu_weight_bytes, tensor.byte_size) {
        Ok(v) => v
        Err(m) => return Err(m)
      }
    }
    placements.push({
      name: tensor.name,
      role: tensor.role,
      device: if on_gpu {
        "GPU"
      } else {
        "CPU"
      },
      byte_size: tensor.byte_size,
      is_duplicate: false,
    })
  }
  let duplicated_bytes = if profile.tied_output && output_on_gpu {
    gpu_weight_bytes = match exact_add(gpu_weight_bytes, profile.input_bytes) {
      Ok(v) => v
      Err(m) => return Err(m)
    }
    placements.push({
      name: "token_embd.weight",
      role: "tied-output",
      device: "GPU",
      byte_size: profile.input_bytes,
      is_duplicate: true,
    })
    profile.input_bytes
  } else {
    0UL
  }
  let total = match exact_add(gpu_weight_bytes, cpu_weight_bytes) {
    Ok(v) => v
    Err(m) => return Err(m)
  }
  let expected = match
    exact_add(profile.tensor_payload_bytes, duplicated_bytes) {
    Ok(v) => v
    Err(m) => return Err(m)
  }
  if total != expected {
    return Err("internal payload placement conservation failure")
  }
  let gpu_kv = allocated_kv_bytes /
    layers.to_double() *
    gpu_block_count.to_double()
  let host_kv = allocated_kv_bytes - gpu_kv
  // Output-only offload still creates a GPU backend and therefore incurs GPU
  // allowances, even though no repeating block or KV cache is on the device.
  let gpu_runtime = if output_on_gpu {
    @core.gib_to_bytes(policy.gpu_runtime_gib)
  } else {
    0.0
  }
  let gpu_margin = if output_on_gpu {
    @core.gib_to_bytes(policy.gpu_margin_gib)
  } else {
    0.0
  }
  let gpu_staging = if output_on_gpu {
    @core.gib_to_bytes(policy.gpu_staging_gib)
  } else {
    0.0
  }
  let host_runtime = @core.gib_to_bytes(policy.host_runtime_gib)
  let os_reserve = @core.gib_to_bytes(policy.os_reserve_gib)
  let workspace = @core.gib_to_bytes(policy.loading_workspace_gib)
  let gpu_weights = gpu_weight_bytes.to_double()
  let host_weights = cpu_weight_bytes.to_double()
  let gpu_steady = gpu_weights + gpu_kv + gpu_runtime + gpu_margin
  let host_steady = host_weights + host_kv + host_runtime + os_reserve
  let ledger : @core.MemoryLedger = {
    gpu_layers: gpu_block_count,
    cpu_layers: layers - gpu_block_count,
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
    host_loading_peak_bytes: profile.tensor_payload_bytes.to_double() +
    host_kv +
    host_runtime +
    os_reserve +
    workspace,
  }
  let warnings = profile.warnings.copy()
  if allocated_context > profile.context_length {
    warnings.push(
      "Allocated context after 256-token padding exceeds GGUF training context; supported extrapolation is not established.",
    )
  }
  if precision != @core.FP16KV {
    warnings.push(
      "Quantized KV requires compatible backend Flash Attention/type support; capacity arithmetic does not establish compatibility.",
    )
  }
  warnings.push(
    "kv_bytes describes requested-context payload; allocated_context and allocated_kv_bytes include the pinned single-sequence 256-token padding and drive the ledger. Buffer alignment remains separate.",
  )
  Ok({
    n_gpu_layers,
    gpu_block_count,
    output_on_gpu,
    gpu_weight_bytes,
    cpu_weight_bytes,
    duplicated_bytes,
    kv_bytes,
    allocated_context,
    allocated_kv_bytes,
    ledger,
    placements,
    backend_policy: "llama.cpp@" +
    backend_revision() +
    "; single discrete GPU; output-first ngl; single-sequence context padded to 256 tokens; plain CPU buffers; no ACCEL/lazy/repack/host/tensor overrides; KV follows blocks; preferred GPU kernels assumed; full-host-weights-loading-policy",
    warnings,
  })
}

///|
fn fits_budgets(candidate : Plan, vram : Double, ram : Double) -> Bool {
  let ledger = candidate.ledger
  ledger.gpu_steady_bytes <= vram &&
  ledger.gpu_loading_peak_bytes <= vram &&
  ledger.host_steady_bytes <= ram &&
  ledger.host_loading_peak_bytes <= ram
}

///|
/// Evaluate exactly one requested output/block placement. A failed budget
/// check returns no plan and never searches for a smaller or larger request.
pub fn evaluate_plan(
  profile : Profile,
  context : Int,
  precision : @core.KvPrecision,
  n_gpu_layers : Int,
  vram_gib : Double,
  ram_gib : Double,
  policy : @core.BudgetPolicy,
) -> Result[PlanSearch, String] {
  if !(vram_gib >= 0.0 && vram_gib <= 1000000.0) ||
    !(ram_gib >= 0.0 && ram_gib <= 1000000.0) {
    return Err("GPU/RAM budgets must be finite and in [0, 1000000] GiB")
  }
  let candidate = match
    plan(profile, context, precision, n_gpu_layers, policy) {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let request = "Fixed n_gpu_layers=\{n_gpu_layers} request"
  if !fits_budgets(
      candidate,
      @core.gib_to_bytes(vram_gib),
      @core.gib_to_bytes(ram_gib),
    ) {
    return Ok({
      status: @core.Insufficient,
      plan: None,
      reason: request +
      " does not fit the separate GPU/RAM steady/loading policy budgets; no alternative placement was searched. Runtime allowances remain uncalibrated.",
    })
  }
  let status = if n_gpu_layers == profile.shape.layers + 1 {
    @core.FullGPU
  } else if n_gpu_layers == 0 {
    @core.CPUOnly
  } else {
    @core.Offload
  }
  Ok({
    status,
    plan: Some(candidate),
    reason: request +
    " fits the separate GPU/RAM steady/loading policy budgets. Stored tensor payloads and explicit duplication are checked; runtime allowances remain uncalibrated.",
  })
}

///|
pub fn search_plan(
  profile : Profile,
  context : Int,
  precision : @core.KvPrecision,
  vram_gib : Double,
  ram_gib : Double,
  policy : @core.BudgetPolicy,
) -> Result[PlanSearch, String] {
  if !(vram_gib >= 0.0 && vram_gib <= 1000000.0) ||
    !(ram_gib >= 0.0 && ram_gib <= 1000000.0) {
    return Err("GPU/RAM budgets must be finite and in [0, 1000000] GiB")
  }
  let cpu = match plan(profile, context, precision, 0, policy) {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let vram = @core.gib_to_bytes(vram_gib)
  let ram = @core.gib_to_bytes(ram_gib)
  for ngl = profile.shape.layers + 1; ngl >= 0; ngl = ngl - 1 {
    let candidate = if ngl == 0 {
      cpu
    } else {
      match plan(profile, context, precision, ngl, policy) {
        Ok(v) => v
        Err(m) => return Err(m)
      }
    }
    if fits_budgets(candidate, vram, ram) {
      let status = if ngl == profile.shape.layers + 1 {
        @core.FullGPU
      } else if ngl == 0 {
        @core.CPUOnly
      } else {
        @core.Offload
      }
      let placement_reason = match status {
        @core.FullGPU =>
          "Maximum block/output offload fits; the input embedding remains on CPU. "
        @core.CPUOnly => "CPU-only payload placement fits. "
        _ => "Partial output/block offload fits. "
      }
      return Ok({
        status,
        plan: Some(candidate),
        reason: placement_reason +
        "Stored tensor payloads and explicit duplication are checked against separate GPU/RAM steady/loading policy budgets; runtime allowances remain uncalibrated.",
      })
    }
  }
  Ok({
    status: @core.Insufficient,
    plan: None,
    reason: "No supported integer output/block placement fits these budgets under the full-host-weights-loading-policy. This does not establish that every backend configuration is unable to run.",
  })
}
