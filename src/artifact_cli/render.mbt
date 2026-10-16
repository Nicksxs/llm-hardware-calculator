///|
/// Raw artifact integers remain decimal strings even above JavaScript's 2^53.
fn exact(value : UInt64) -> Json {
  Json(value.to_string())
}

///|
fn optional_exact(value : UInt64?) -> Json {
  match value {
    Some(value) => exact(value)
    None => Json::null()
  }
}

///|
fn io_json(io : @nativeio.IoStats) -> Json {
  {
    "read_calls": Json(io.read_calls),
    "bytes_read": exact(io.bytes_read),
    "max_read_end": exact(io.max_read_end),
    "max_request": Json(io.max_request),
  }
}

///|
fn metadata_json(metadata : @gguf.Metadata) -> Json {
  let value : Json = match metadata.value {
    @gguf.UIntValue(value) =>
      { "kind": "unsigned_integer", "value": exact(value) }
    @gguf.IntValue(value) =>
      { "kind": "signed_integer", "value": Json(value.to_string()) }
    @gguf.FloatValue(value) => { "kind": "float", "value": Json(value) }
    @gguf.BoolValue(value) => { "kind": "boolean", "value": Json(value) }
    @gguf.StringValue(value) =>
      if value.length() > 512 {
        {
          "kind": "string",
          "value": null,
          "omitted": true,
          "utf16_length": Json(value.length()),
        }
      } else {
        { "kind": "string", "value": Json(value), "omitted": false }
      }
    @gguf.ArrayValue(type_id, count, values, complete) =>
      {
        "kind": "array",
        "element_type_id": Json(type_id),
        "total_count": exact(count),
        "retained_value_count": Json(values.length()),
        "retention_complete": Json(complete),
        "value": null,
        "omitted": true,
        "summary": "array values omitted from report",
      }
  }
  { "key": Json(metadata.key), "value": value }
}

///|
fn tensor_json(tensor : @gguf.Tensor) -> Json {
  {
    "name": Json(tensor.name),
    "dimensions": Json::array(tensor.dimensions.map(exact)),
    "type_id": Json(tensor.type_id),
    "offset": exact(tensor.offset),
    "absolute_offset": exact(tensor.absolute_offset),
    "element_count": exact(tensor.element_count),
    "byte_size": optional_exact(tensor.byte_size),
  }
}

///|
fn tensor_totals(model : @gguf.Model) -> (UInt64?, Int) {
  let mut known = Some(0UL)
  let mut unknown = 0
  for tensor in model.tensors {
    match tensor.byte_size {
      None => unknown += 1
      Some(size) =>
        match known {
          None => ()
          Some(sum) =>
            known = if sum > 18446744073709551615UL - size {
              None
            } else {
              Some(sum + size)
            }
        }
    }
  }
  (known, unknown)
}

///|
fn artifact_json(scan : @nativeio.Scan) -> Json {
  let model = scan.model
  let (known, unknown) = tensor_totals(model)
  {
    "gguf_version": Json(model.version),
    "file_size": exact(model.file_size),
    "data_offset": exact(model.data_offset),
    "metadata_count": Json(model.metadata.length()),
    "tensor_count": Json(model.tensors.length()),
    "complete": Json(model.complete),
    "known_tensor_bytes": optional_exact(known),
    "unknown_tensor_count": Json(unknown),
    "total_tensor_bytes": if model.complete && unknown == 0 {
      optional_exact(known)
    } else {
      Json::null()
    },
    "warnings": Json(model.warnings),
    "io": io_json(scan.io),
  }
}

///|
pub fn render_inspect_json(scan : @nativeio.Scan) -> String {
  let report : Json = {
    "schema_version": "1",
    "command": "inspect",
    "artifact": artifact_json(scan),
    "metadata": Json::array(scan.model.metadata.map(metadata_json)),
    "tensors": Json::array(scan.model.tensors.map(tensor_json)),
    "integer_encoding": "raw UInt64 bytes, offsets, dimensions and element counts are decimal strings",
    "scope": "metadata and tensor descriptors only; tensor payload contents, checksums and inference compatibility are not verified",
  }
  @json.to_json(report).stringify(indent=2)
}

///|
fn exact_text(value : UInt64?) -> String {
  match value {
    Some(value) => value.to_string() + " bytes"
    None => "unknown"
  }
}

///|
pub fn render_inspect_text(scan : @nativeio.Scan) -> String {
  let model = scan.model
  let (known, unknown) = tensor_totals(model)
  let lines = [
    "GGUF inspection",
    "Version: \{model.version}; file size: \{model.file_size} bytes; data offset: \{model.data_offset}",
    "Metadata entries: \{model.metadata.length()}; tensors: \{model.tensors.length()}",
    "Complete tensor sizes: \{model.complete}; known tensor payload: \{exact_text(known)}; tensors with unknown byte size: \{unknown}",
    "I/O: \{scan.io.read_calls} reads, \{scan.io.bytes_read} bytes read, maximum read end \{scan.io.max_read_end}, maximum request \{scan.io.max_request} bytes",
  ]
  lines.push(
    if !model.complete || unknown > 0 {
      "Total tensor payload: unknown; incomplete sizes cannot be used for planning."
    } else {
      "Total tensor payload: \{exact_text(known)} (excludes file padding and metadata)."
    },
  )
  for warning in model.warnings {
    lines.push("Warning: " + warning)
  }
  lines.push("Metadata summaries:")
  for metadata in model.metadata {
    lines.push("  " + @json.to_json(metadata_json(metadata)).stringify())
  }
  lines.push("Tensor descriptors:")
  for tensor in model.tensors {
    let shape = tensor.dimensions.map(value => value.to_string()).join(" x ")
    lines.push(
      "  \{tensor.name}: shape [\{shape}], type id \{tensor.type_id}, elements \{tensor.element_count}, payload \{exact_text(tensor.byte_size)}, offset \{tensor.offset}, absolute offset \{tensor.absolute_offset}",
    )
  }
  lines.push(
    "Scope: metadata and tensor descriptors only. Payload contents, checksums and inference compatibility are not verified.",
  )
  lines.join("\n")
}

///|
fn shape_json(shape : @core.Architecture) -> Json {
  {
    "layers": Json(shape.layers),
    "kv_heads": Json(shape.kv_heads),
    "head_dim": Json(shape.head_dim),
    "query_heads": Json(shape.query_heads),
    "source": Json(shape.source),
  }
}

///|
fn policy_json(policy : @core.BudgetPolicy) -> Json {
  {
    "id": Json(policy.id),
    "gpu_runtime_gib": Json(policy.gpu_runtime_gib),
    "gpu_margin_gib": Json(policy.gpu_margin_gib),
    "host_runtime_gib": Json(policy.host_runtime_gib),
    "os_reserve_gib": Json(policy.os_reserve_gib),
    "loading_workspace_gib": Json(policy.loading_workspace_gib),
    "gpu_staging_gib": Json(policy.gpu_staging_gib),
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
fn tensor_record_json(tensor : @model.TensorRecord) -> Json {
  let block = match tensor.block {
    Some(block) => Json(block)
    None => Json::null()
  }
  {
    "name": Json(tensor.name),
    "byte_size": exact(tensor.byte_size),
    "role": Json(tensor.role),
    "block": block,
    "dimensions": Json::array(tensor.dimensions.map(exact)),
    "type_id": Json(tensor.type_id),
    "element_count": exact(tensor.element_count),
  }
}

///|
fn profile_json(profile : @model.Profile) -> Json {
  {
    "architecture": Json(profile.architecture),
    "shape": shape_json(profile.shape),
    "context_length": Json(profile.context_length),
    "embedding_length": Json(profile.embedding_length),
    "tensor_payload_bytes": exact(profile.tensor_payload_bytes),
    "parameter_count": exact(profile.parameter_count),
    "parameter_count_basis": "sum of stored tensor elements; not a certified trainable parameter count",
    "layer_bytes": Json::array(profile.layer_bytes.map(exact)),
    "input_bytes": exact(profile.input_bytes),
    "output_bytes": exact(profile.output_bytes),
    "tied_output": Json(profile.tied_output),
    "backend_revision": Json(profile.backend_revision),
    "warnings": Json(profile.warnings),
    "tensors": Json::array(profile.tensors.map(tensor_record_json)),
  }
}

///|
fn placement_json(placement : @model.TensorPlacement) -> Json {
  {
    "name": Json(placement.name),
    "role": Json(placement.role),
    "device": Json(placement.device),
    "byte_size": exact(placement.byte_size),
    "is_duplicate": Json(placement.is_duplicate),
  }
}

///|
fn plan_json(plan : @model.Plan) -> Json {
  {
    "n_gpu_layers": Json(plan.n_gpu_layers),
    "gpu_block_count": Json(plan.gpu_block_count),
    "output_on_gpu": Json(plan.output_on_gpu),
    "gpu_weight_bytes": exact(plan.gpu_weight_bytes),
    "cpu_weight_bytes": exact(plan.cpu_weight_bytes),
    "duplicated_bytes": exact(plan.duplicated_bytes),
    "kv_bytes": Json(plan.kv_bytes),
    "allocated_context_tokens": Json(plan.allocated_context),
    "allocated_kv_bytes": Json(plan.allocated_kv_bytes),
    "ledger": ledger_json(plan.ledger),
    "ledger_layer_unit": "transformer blocks; the output offload slot is not counted",
    "placements": Json::array(plan.placements.map(placement_json)),
    "backend_policy": Json(plan.backend_policy),
    "warnings": Json(plan.warnings),
  }
}

///|
fn artifact_status(status : @core.FitStatus) -> String {
  match status {
    @core.FullGPU => "MAX_OFFLOAD"
    _ => @core.fit_status_name(status)
  }
}

///|
pub fn render_plan_json(
  scan : @nativeio.Scan,
  profile : @model.Profile,
  search : @model.PlanSearch,
  options : PlanOptions,
) -> String {
  let plan = match search.plan {
    Some(plan) => plan_json(plan)
    None => Json::null()
  }
  let requested_layers = match options.n_gpu_layers {
    Some(value) => Json(value)
    None => Json::null()
  }
  let report : Json = {
    "schema_version": "1",
    "command": "plan",
    "input": {
      "path": Json(options.path),
      "vram_gib": Json(options.vram_gib),
      "ram_gib": Json(options.ram_gib),
      "context_tokens": Json(options.context),
      "kv_precision": Json(@core.kv_precision_name(options.kv_precision)),
      "n_gpu_layers": requested_layers,
    },
    "artifact": artifact_json(scan),
    "profile": profile_json(profile),
    "policy": policy_json(options.policy),
    "fit": {
      "status": Json(artifact_status(search.status)),
      "reason": Json(search.reason),
      "plan": plan,
    },
    "integer_encoding": "raw UInt64 artifact and placement counts are decimal strings; GiB and runtime/KV budgets are presentation/planning numbers",
    "assumptions": {
      "scope": "one GPU, supported dense decoder-only full attention, one sequence",
      "n_gpu_layers": "pinned output-first policy: output occupies the first offload slot, followed by transformer blocks; input embeddings remain on CPU",
      "max_offload": "all supported blocks and output offloaded; CPU input embeddings may remain, so this is not FULL_GPU residency",
      "payload": "exact tensor payload bytes exclude metadata and alignment padding; runtime allocations are separate policy estimates",
      "loading": "conservative full-host-weight retention with separate GPU/RAM loading peak budgets",
      "kv": "KV precision is a memory scenario and does not guarantee backend runtime compatibility",
      "verification": "tensor payload contents, checksums, inference correctness and throughput are not verified",
    },
  }
  @json.to_json(report).stringify(indent=2)
}

///|
fn gib(bytes : Double) -> String {
  "\{@core.bytes_to_gib(bytes)} GiB"
}

///|
pub fn render_plan_text(
  scan : @nativeio.Scan,
  profile : @model.Profile,
  search : @model.PlanSearch,
  options : PlanOptions,
) -> String {
  let shape = profile.shape
  let lines = [
    "GGUF artifact plan",
    "Fit: " + artifact_status(search.status),
    "Reason: " + search.reason,
    "File: \{options.path}; GGUF v\{scan.model.version}; \{scan.model.file_size} bytes",
    "Architecture: \{profile.architecture}; layers=\{shape.layers}, KV heads=\{shape.kv_heads}, head dim=\{shape.head_dim}, query heads=\{shape.query_heads}",
    "Source: \{shape.source}",
    "Model context metadata: \{profile.context_length}; requested context: \{options.context}; KV precision: \{@core.kv_precision_name(options.kv_precision)}",
    "Exact tensor payload: \{profile.tensor_payload_bytes} bytes; tensor elements: \{profile.parameter_count}; tied output: \{profile.tied_output}",
    "Input embeddings: \{profile.input_bytes} bytes; output tensors: \{profile.output_bytes} bytes",
    "Capacities: VRAM \{options.vram_gib} GiB, RAM \{options.ram_gib} GiB",
    "Budget policy: \{options.policy.id}",
    "Policy allowances (GiB): GPU runtime \{options.policy.gpu_runtime_gib}, margin \{options.policy.gpu_margin_gib}, staging \{options.policy.gpu_staging_gib}; host runtime \{options.policy.host_runtime_gib}, OS reserve \{options.policy.os_reserve_gib}, loading workspace \{options.policy.loading_workspace_gib}",
    "Backend revision: \{profile.backend_revision}",
  ]
  lines.push(
    match options.n_gpu_layers {
      Some(value) => "Offload selection: fixed n_gpu_layers=\{value}"
      None => "Offload selection: search for the largest fitting placement"
    },
  )
  for warning in scan.model.warnings {
    lines.push("Artifact warning: " + warning)
  }
  for warning in profile.warnings {
    lines.push("Profile warning: " + warning)
  }
  match search.plan {
    None => lines.push("No feasible placement under the selected budgets.")
    Some(plan) => {
      let ledger = plan.ledger
      lines.push(
        "Offload: n_gpu_layers=\{plan.n_gpu_layers} (output-first slots), GPU transformer blocks=\{plan.gpu_block_count}, output on GPU=\{plan.output_on_gpu}",
      )
      lines.push(
        "Exact placement weights: GPU \{plan.gpu_weight_bytes} bytes, CPU \{plan.cpu_weight_bytes} bytes, duplicated \{plan.duplicated_bytes} bytes",
      )
      lines.push("KV payload: \{gib(plan.kv_bytes)}")
      lines.push(
        "Allocated KV: \{gib(plan.allocated_kv_bytes)} for \{plan.allocated_context} tokens (pinned backend pads context to 256 tokens)",
      )
      lines.push(
        "GPU budget: weights \{gib(ledger.gpu_weights_bytes)}, KV \{gib(ledger.gpu_kv_bytes)}, runtime \{gib(ledger.gpu_runtime_bytes)}, margin \{gib(ledger.gpu_margin_bytes)}, staging \{gib(ledger.gpu_staging_bytes)}",
      )
      lines.push(
        "GPU steady: \{gib(ledger.gpu_steady_bytes)}; loading peak: \{gib(ledger.gpu_loading_peak_bytes)}",
      )
      lines.push(
        "RAM budget: weights \{gib(ledger.host_weights_bytes)}, KV \{gib(ledger.host_kv_bytes)}, runtime \{gib(ledger.host_runtime_bytes)}, OS reserve \{gib(ledger.os_reserve_bytes)}, loading workspace \{gib(ledger.loading_workspace_bytes)}",
      )
      lines.push(
        "RAM steady: \{gib(ledger.host_steady_bytes)}; loading peak: \{gib(ledger.host_loading_peak_bytes)}",
      )
      lines.push("Backend placement policy: " + plan.backend_policy)
      for warning in plan.warnings {
        lines.push("Plan warning: " + warning)
      }
      lines.push("Tensor placements:")
      for placement in plan.placements {
        lines.push(
          "  \{placement.name}: role=\{placement.role}, device=\{placement.device}, \{placement.byte_size} bytes, duplicate=\{placement.is_duplicate}",
        )
      }
    }
  }
  lines.push(
    "MAX_OFFLOAD means all supported blocks/output can be offloaded while input embeddings remain on CPU. n_gpu_layers=1 may place only output on GPU. Artifact byte counts are exact; runtime and KV budgets are estimates under the disclosed policy. Payload contents, checksums, inference compatibility and throughput are not verified.",
  )
  lines.join("\n")
}
