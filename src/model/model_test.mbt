///|
fn fixture_tensor(
  name : String,
  dimensions : Array[UInt64],
  fp16 : Bool,
) -> @gguf.Tensor {
  let mut elements = 1UL
  for dimension in dimensions {
    elements *= dimension
  }
  {
    name,
    dimensions,
    type_id: if fp16 {
      1
    } else {
      0
    },
    offset: 0UL,
    absolute_offset: 4096UL,
    element_count: elements,
    byte_size: Some(elements * (if fp16 { 2UL } else { 4UL })),
  }
}

///|
fn inventory_fixture(
  architecture : String,
  tied : Bool,
  uneven : Bool,
) -> @gguf.Model {
  let prefix = architecture + "."
  let metadata : Array[@gguf.Metadata] = [
    { key: "general.architecture", value: @gguf.StringValue(architecture), },
    { key: prefix + "block_count", value: @gguf.UIntValue(2UL), },
    { key: prefix + "embedding_length", value: @gguf.UIntValue(4UL), },
    { key: prefix + "context_length", value: @gguf.UIntValue(8192UL), },
    { key: prefix + "feed_forward_length", value: @gguf.UIntValue(8UL), },
    { key: prefix + "attention.head_count", value: @gguf.UIntValue(2UL), },
    { key: prefix + "attention.head_count_kv", value: @gguf.UIntValue(1UL), },
  ]
  let tensors = [
    fixture_tensor("token_embd.weight", [4UL, 8UL], false),
    fixture_tensor("output_norm.weight", [4UL], false),
  ]
  if !tied {
    tensors.push(fixture_tensor("output.weight", [4UL, 8UL], false))
  }
  for index in 0..<2 {
    let block = "blk." + index.to_string() + "."
    for
      (leaf, shape) in [
        ("attn_norm.weight", [4UL]),
        ("attn_q.weight", [4UL, 4UL]),
        ("attn_k.weight", [4UL, 2UL]),
        ("attn_v.weight", [4UL, 2UL]),
        ("attn_output.weight", [4UL, 4UL]),
        ("ffn_norm.weight", [4UL]),
        ("ffn_gate.weight", [4UL, 8UL]),
        ("ffn_down.weight", [8UL, 4UL]),
        ("ffn_up.weight", [4UL, 8UL]),
      ] {
      tensors.push(fixture_tensor(block + leaf, shape, uneven && index == 0))
    }
  }
  let mut offset = 0UL
  let located = tensors.map(tensor => {
    let located = { ..tensor, offset, absolute_offset: 4096UL + offset, }
    let length = tensor.byte_size.unwrap()
    offset += (length + 31UL) / 32UL * 32UL
    located
  })
  {
    version: 3,
    file_size: 4096UL + offset,
    data_offset: 4096UL,
    metadata,
    tensors: located,
    complete: true,
    warnings: [],
  }
}

///|
fn replace_metadata(
  model : @gguf.Model,
  key : String,
  value : @gguf.Value,
) -> @gguf.Model {
  let mut replaced = false
  let metadata = model.metadata.map(entry => {
    if entry.key == key {
      replaced = true
      { ..entry, value, }
    } else {
      entry
    }
  })
  if !replaced {
    metadata.push({ key, value, })
  }
  { ..model, metadata, }
}

///|
fn model_zero_policy() -> @core.BudgetPolicy {
  {
    gpu_runtime_gib: 0.0,
    gpu_margin_gib: 0.0,
    host_runtime_gib: 0.0,
    os_reserve_gib: 0.0,
    loading_workspace_gib: 0.0,
    gpu_staging_gib: 0.0,
    id: "synthetic zero-overhead model fixture",
  }
}

///|
test "real tensor inventory retains nonuniform byte costs without BPW presets" {
  let profile = @model.extract_profile(inventory_fixture("qwen2", false, true)).unwrap()
  assert_eq(profile.layer_bytes, [304UL, 608UL])
  assert_eq(profile.input_bytes, 128UL)
  assert_eq(profile.output_bytes, 144UL)
  assert_eq(profile.tensor_payload_bytes, 1184UL)
  assert_eq(profile.parameter_count, 372UL)
  assert_eq(profile.shape.layers, 2)
  assert_eq(profile.shape.query_heads, 2)
  assert_eq(profile.shape.kv_heads, 1)
  assert_eq(profile.shape.head_dim, 2)
  assert_eq(profile.context_length, 8192)
  assert_false(profile.tied_output)
  assert_eq(
    profile.backend_revision,
    "88aaf325a7026ac1127d0897394b0a3d6413dc2a",
  )
}

///|
test "output vocabulary checks are independent of tensor directory order" {
  let original = inventory_fixture("llama", false, false)
  let tensors = original.tensors.copy()
  let first = tensors[0]
  tensors[0] = tensors[2]
  tensors[2] = first
  assert_true(@model.extract_profile({ ..original, tensors, }) is Ok(_))
  let broken = tensors.map(tensor => {
    if tensor.name == "output.weight" {
      { ..tensor, dimensions: [4UL, 9UL], }
    } else {
      tensor
    }
  })
  assert_true(
    @model.extract_profile({ ..original, tensors: broken, }) is Err(_),
  )
}

///|
test "profile rejects missing fields unsupported structures and inconsistent shapes" {
  let model = inventory_fixture("llama", false, false)
  assert_true(@model.extract_profile({ ..model, complete: false, }) is Err(_))
  assert_true(
    @model.extract_profile(
      replace_metadata(
        model,
        "general.architecture",
        @gguf.StringValue("gemma3"),
      ),
    )
    is Err(_),
  )
  for
    (key, value) in [
      ("llama.expert_count", @gguf.UIntValue(8UL)),
      ("llama.attention.sliding_window", @gguf.UIntValue(4096UL)),
      ("llama.attention.causal", @gguf.BoolValue(false)),
      ("llama.nextn_predict_layers", @gguf.UIntValue(1UL)),
      ("llama.rope.dimension_count", @gguf.UIntValue(1UL)),
      ("llama.attention.value_length", @gguf.UIntValue(4UL)),
      ("llama.block_count", @gguf.ArrayValue(4, 2UL, [], false)),
    ] {
    assert_true(
      @model.extract_profile(replace_metadata(model, key, value)) is Err(_),
    )
  }
  assert_true(
    @model.extract_profile({
      ..model,
      metadata: model.metadata.filter(e => e.key != "llama.block_count"),
    })
    is Err(_),
  )
  assert_true(
    @model.extract_profile({
      ..model,
      tensors: model.tensors.filter(t => t.name != "blk.1.ffn_down.weight"),
    })
    is Err(_),
  )
  let bad_shape = model.tensors.map(tensor => {
    if tensor.name == "blk.0.attn_k.weight" {
      { ..tensor, dimensions: [4UL, 4UL], }
    } else {
      tensor
    }
  })
  assert_true(
    @model.extract_profile({ ..model, tensors: bad_shape, }) is Err(_),
  )
}

///|
test "planner refuses unknown names and types while retaining inspectable model" {
  let model = inventory_fixture("qwen2", false, false)
  for
    name in [
      "blk.0.foo.weight", "blk.2.attn_q.weight", "blk.01.attn_q.weight", "blk.0.attn_qkv.weight",
      "rope_freqs.weight", "blk.0.ffn_gate_exps.weight", "blk.0.attn_q.scale",
    ] {
    let tensors = model.tensors.copy()
    tensors.push(fixture_tensor(name, [4UL], false))
    assert_true(@model.extract_profile({ ..model, tensors, }) is Err(_))
    assert_true(tensors.length() > model.tensors.length())
  }
  for type_id in [9, 15, 999] {
    let tensors = model.tensors.map(tensor => {
      if tensor.name == "blk.0.attn_q.weight" {
        { ..tensor, type_id, }
      } else {
        tensor
      }
    })
    assert_true(@model.extract_profile({ ..model, tensors, }) is Err(_))
  }
}

///|
test "architecture-specific optional biases are counted only when backend reads them" {
  let llama = inventory_fixture("llama", false, false)
  let qwen = inventory_fixture("qwen2", false, false)
  let extra = fixture_tensor("blk.0.attn_output.bias", [4UL], false)
  let llama_tensors = llama.tensors.copy()
  llama_tensors.push(extra)
  let llama_profile = @model.extract_profile({
    ..llama,
    tensors: llama_tensors,
  }).unwrap()
  assert_eq(llama_profile.layer_bytes[0], 624UL)
  let qwen_tensors = qwen.tensors.copy()
  qwen_tensors.push(extra)
  assert_true(
    @model.extract_profile({ ..qwen, tensors: qwen_tensors, }) is Err(_),
  )
  let qwen_output = qwen.tensors.copy()
  qwen_output.push(fixture_tensor("output.bias", [8UL], false))
  assert_eq(
    @model.extract_profile({ ..qwen, tensors: qwen_output, }).unwrap().output_bytes,
    176UL,
  )
  let tokens = @gguf.ArrayValue(8, 9UL, [], false)
  assert_true(
    @model.extract_profile(
      replace_metadata(qwen, "tokenizer.ggml.tokens", tokens),
    )
    is Err(_),
  )
}

///|
test "fixed backend offloads output first and embedding stays CPU at maximum" {
  let profile = @model.extract_profile(inventory_fixture("qwen2", false, true)).unwrap()
  let expected = [
    (0, 0, 0UL, 1184UL),
    (1, 0, 144UL, 1040UL),
    (2, 1, 752UL, 432UL),
    (3, 2, 1056UL, 128UL),
  ]
  for (ngl, blocks, gpu, cpu) in expected {
    let plan = @model.plan(profile, 16, @core.FP16KV, ngl, model_zero_policy()).unwrap()
    assert_eq(plan.gpu_block_count, blocks)
    assert_eq(plan.ledger.gpu_layers, blocks)
    assert_eq(plan.gpu_weight_bytes, gpu)
    assert_eq(plan.cpu_weight_bytes, cpu)
    assert_eq(gpu + cpu, profile.tensor_payload_bytes)
    assert_eq(plan.kv_bytes, 256.0)
    assert_eq(plan.allocated_context, 256)
    assert_eq(plan.allocated_kv_bytes, 4096.0)
    assert_eq(plan.ledger.gpu_kv_bytes, blocks.to_double() * 2048.0)
    for placed in plan.placements {
      if placed.name == "token_embd.weight" {
        assert_eq(placed.device, "CPU")
      }
      if placed.name.has_prefix("blk.0.") && ngl < 3 {
        assert_eq(placed.device, "CPU")
      }
      if placed.name.has_prefix("blk.1.") && ngl >= 2 {
        assert_eq(placed.device, "GPU")
      }
    }
  }
}

///|
test "tied output duplicates only GPU allocation not file payload or CPU reuse" {
  let profile = @model.extract_profile(inventory_fixture("llama", true, true)).unwrap()
  assert_true(profile.tied_output)
  assert_eq(profile.tensor_payload_bytes, 1056UL)
  assert_eq(profile.output_bytes, 16UL)
  for ngl in 0..<4 {
    let plan = @model.plan(profile, 16, @core.FP16KV, ngl, model_zero_policy()).unwrap()
    let duplicate = if ngl > 0 { 128UL } else { 0UL }
    assert_eq(plan.duplicated_bytes, duplicate)
    assert_eq(
      plan.gpu_weight_bytes + plan.cpu_weight_bytes,
      profile.tensor_payload_bytes + duplicate,
    )
    let copies = plan.placements.filter(p => p.name == "token_embd.weight")
    assert_eq(copies.length(), if ngl > 0 { 2 } else { 1 })
    if ngl > 0 {
      assert_true(copies[1].is_duplicate)
      assert_eq(copies[1].device, "GPU")
    }
  }
}

///|
test "output-only offload still incurs GPU runtime margin and staging" {
  let profile = @model.extract_profile(inventory_fixture("qwen2", false, true)).unwrap()
  let plan = @model.plan(profile, 16, @core.FP16KV, 1, @core.default_policy()).unwrap()
  assert_eq(plan.gpu_block_count, 0)
  assert_eq(plan.ledger.gpu_kv_bytes, 0.0)
  assert_eq(plan.ledger.gpu_runtime_bytes, @core.gib_to_bytes(0.5))
  assert_eq(plan.ledger.gpu_margin_bytes, @core.gib_to_bytes(1.0))
  assert_eq(plan.ledger.gpu_staging_bytes, @core.gib_to_bytes(0.25))
}

///|
test "nonuniform real layer costs change the plan versus uniform block allocation" {
  let profile = @model.extract_profile(inventory_fixture("qwen2", false, true)).unwrap()
  let real = @model.plan(profile, 16, @core.FP16KV, 2, model_zero_policy()).unwrap()
  let uniform = 144.0 + (304.0 + 608.0) / 2.0 + 2048.0
  assert_eq(uniform, 2648.0)
  assert_eq(real.ledger.gpu_steady_bytes, 2800.0)
  let result = @model.search_plan(
    profile,
    16,
    @core.FP16KV,
    @core.bytes_to_gib(2700.0),
    @core.bytes_to_gib(6000.0),
    model_zero_policy(),
  ).unwrap()
  assert_eq(result.status, @core.Offload)
  assert_eq(result.plan.unwrap().n_gpu_layers, 1)
  assert_eq(result.plan.unwrap().gpu_block_count, 0)
}

///|
test "fixed offload preserves every requested output and block count" {
  let profile = @model.extract_profile(inventory_fixture("qwen2", false, true)).unwrap()
  for
    (ngl, status) in [
      (0, @core.CPUOnly),
      (1, @core.Offload),
      (2, @core.Offload),
      (3, @core.FullGPU),
    ] {
    let result = @model.evaluate_plan(
      profile,
      16,
      @core.FP16KV,
      ngl,
      1.0,
      1.0,
      model_zero_policy(),
    ).unwrap()
    assert_eq(result.status, status)
    assert_eq(result.plan.unwrap().n_gpu_layers, ngl)
    assert_true(result.reason.contains("Fixed n_gpu_layers=" + ngl.to_string()))
  }
}

///|
test "fixed offload checks steady and loading budgets without fallback" {
  let profile = @model.extract_profile(inventory_fixture("qwen2", false, true)).unwrap()
  let policy = model_zero_policy()
  let exact = @model.evaluate_plan(
    profile,
    16,
    @core.FP16KV,
    2,
    @core.bytes_to_gib(2800.0),
    @core.bytes_to_gib(3232.0),
    policy,
  ).unwrap()
  assert_eq(exact.plan.unwrap().n_gpu_layers, 2)
  for
    (gpu_bytes, host_bytes, candidate_policy) in [
      (2799.0, 6000.0, policy),
      (2800.0, 3231.0, policy),
      (2800.0, 6000.0, { ..policy, gpu_staging_gib: @core.bytes_to_gib(1.0), }),
    ] {
    let result = @model.evaluate_plan(
      profile,
      16,
      @core.FP16KV,
      2,
      @core.bytes_to_gib(gpu_bytes),
      @core.bytes_to_gib(host_bytes),
      candidate_policy,
    ).unwrap()
    assert_eq(result.status, @core.Insufficient)
    assert_eq(result.plan, None)
    assert_true(result.reason.contains("Fixed n_gpu_layers=2"))
    assert_true(result.reason.contains("no alternative placement was searched"))
  }
  let fallback = @model.search_plan(
    profile,
    16,
    @core.FP16KV,
    @core.bytes_to_gib(2799.0),
    @core.bytes_to_gib(6000.0),
    policy,
  ).unwrap()
  assert_eq(fallback.plan.unwrap().n_gpu_layers, 1)
}

///|
test "fixed offload validates model bound and both capacity inputs" {
  let profile = @model.extract_profile(inventory_fixture("qwen2", false, true)).unwrap()
  for ngl in [-1, 4] {
    assert_true(
      @model.evaluate_plan(
        profile,
        16,
        @core.FP16KV,
        ngl,
        1.0,
        1.0,
        model_zero_policy(),
      )
      is Err(_),
    )
  }
  for invalid in [-1.0, 1000001.0, 0.0 / 0.0, 1.0 / 0.0] {
    assert_true(
      @model.evaluate_plan(
        profile,
        16,
        @core.FP16KV,
        0,
        invalid,
        1.0,
        model_zero_policy(),
      )
      is Err(_),
    )
    assert_true(
      @model.evaluate_plan(
        profile,
        16,
        @core.FP16KV,
        0,
        1.0,
        invalid,
        model_zero_policy(),
      )
      is Err(_),
    )
  }
}

///|
test "artifact planning checks exact capacity loading peaks and zero GPU" {
  let profile = @model.extract_profile(inventory_fixture("qwen2", false, true)).unwrap()
  let policy = model_zero_policy()
  let exact = @model.search_plan(
    profile,
    16,
    @core.FP16KV,
    @core.bytes_to_gib(2800.0),
    @core.bytes_to_gib(6000.0),
    policy,
  ).unwrap()
  assert_eq(exact.plan.unwrap().n_gpu_layers, 2)
  let below = @model.search_plan(
    profile,
    16,
    @core.FP16KV,
    @core.bytes_to_gib(2799.0),
    @core.bytes_to_gib(6000.0),
    policy,
  ).unwrap()
  assert_eq(below.plan.unwrap().n_gpu_layers, 1)
  let host_short = @model.search_plan(
    profile,
    16,
    @core.FP16KV,
    @core.bytes_to_gib(5152.0),
    @core.bytes_to_gib(128.0),
    policy,
  ).unwrap()
  assert_eq(host_short.status, @core.Insufficient)
  let max = @model.search_plan(
    profile,
    16,
    @core.FP16KV,
    @core.bytes_to_gib(5152.0),
    @core.bytes_to_gib(1184.0),
    policy,
  ).unwrap()
  assert_eq(max.status, @core.FullGPU)
  assert_eq(max.plan.unwrap().n_gpu_layers, 3)
  let cpu = @model.search_plan(
    profile,
    16,
    @core.FP16KV,
    0.0,
    @core.bytes_to_gib(5280.0),
    policy,
  ).unwrap()
  assert_eq(cpu.status, @core.CPUOnly)
  assert_eq(cpu.plan.unwrap().n_gpu_layers, 0)
}

///|
test "pinned backend rounds single-sequence context up to 256 tokens" {
  let profile = @model.extract_profile(inventory_fixture("qwen2", false, true)).unwrap()
  for
    (requested, allocated, raw, padded) in [
      (1, 256, 16.0, 4096.0),
      (255, 256, 4080.0, 4096.0),
      (256, 256, 4096.0, 4096.0),
      (257, 512, 4112.0, 8192.0),
    ] {
    let plan = @model.plan(
      profile,
      requested,
      @core.FP16KV,
      2,
      model_zero_policy(),
    ).unwrap()
    assert_eq(plan.kv_bytes, raw)
    assert_eq(plan.allocated_context, allocated)
    assert_eq(plan.allocated_kv_bytes, padded)
    assert_eq(plan.ledger.gpu_kv_bytes, padded / 2.0)
    assert_eq(plan.ledger.host_kv_bytes, padded / 2.0)
    assert_eq(plan.ledger.host_loading_peak_bytes, 1184.0 + padded / 2.0)
  }
  let max = @model.plan(profile, 1048576, @core.FP16KV, 0, model_zero_policy()).unwrap()
  assert_eq(max.allocated_context, 1048576)
}

///|
test "padding boundaries affect fit and model context warnings use allocated capacity" {
  let profile = @model.extract_profile(inventory_fixture("qwen2", false, true)).unwrap()
  let policy = model_zero_policy()
  let padded_fit = @model.search_plan(
    profile,
    256,
    @core.FP16KV,
    @core.bytes_to_gib(2800.0),
    @core.bytes_to_gib(12000.0),
    policy,
  ).unwrap()
  assert_eq(padded_fit.plan.unwrap().n_gpu_layers, 2)
  let next = @model.search_plan(
    profile,
    257,
    @core.FP16KV,
    @core.bytes_to_gib(2800.0),
    @core.bytes_to_gib(12000.0),
    policy,
  ).unwrap()
  assert_eq(next.plan.unwrap().n_gpu_layers, 1)
  let limited = { ..profile, context_length: 255, }
  let plan = @model.plan(limited, 255, @core.FP16KV, 0, policy).unwrap()
  assert_true(
    plan.warnings.any(warning => {
      warning.contains("Allocated context after 256-token padding exceeds")
    }),
  )
  let aligned = { ..profile, context_length: 256, }
  let plan = @model.plan(aligned, 255, @core.FP16KV, 0, policy).unwrap()
  assert_false(
    plan.warnings.any(warning => {
      warning.contains("exceeds GGUF training context")
    }),
  )
}

///|
test "planner validates profile counts budgets policy context and quant alignment" {
  let profile = @model.extract_profile(inventory_fixture("qwen2", false, true)).unwrap()
  let policy = model_zero_policy()
  for ngl in [-1, 4] {
    assert_true(@model.plan(profile, 16, @core.FP16KV, ngl, policy) is Err(_))
  }
  assert_true(@model.plan(profile, 0, @core.FP16KV, 0, policy) is Err(_))
  assert_true(@model.plan(profile, 16, @core.Q8KV, 0, policy) is Err(_))
  assert_true(
    @model.plan({ ..profile, layer_bytes: [1UL], }, 16, @core.FP16KV, 0, policy)
    is Err(_),
  )
  assert_true(
    @model.plan(profile, 16, @core.FP16KV, 0, {
      ..policy,
      gpu_runtime_gib: -1.0,
    })
    is Err(_),
  )
  for invalid in [-1.0, 1000001.0, 0.0 / 0.0, 1.0 / 0.0] {
    assert_true(
      @model.search_plan(profile, 16, @core.FP16KV, invalid, 64.0, policy)
      is Err(_),
    )
    assert_true(
      @model.search_plan(profile, 16, @core.FP16KV, 16.0, invalid, policy)
      is Err(_),
    )
  }
}

///|
test "public profile mutation cannot change structure roles blocks or storage silently" {
  let profile = @model.extract_profile(inventory_fixture("qwen2", false, true)).unwrap()
  let policy = model_zero_policy()
  assert_true(
    @model.plan(
      { ..profile, shape: { ..profile.shape, head_dim: 1, }, },
      16,
      @core.FP16KV,
      1,
      policy,
    )
    is Err(_),
  )
  let moved = profile.tensors.map(record => {
    if record.name == "output_norm.weight" {
      { ..record, role: "block", block: Some(0), }
    } else {
      record
    }
  })
  assert_true(
    @model.plan(
      {
        ..profile,
        tensors: moved,
        layer_bytes: [320UL, 608UL],
        output_bytes: 128UL,
      },
      16,
      @core.FP16KV,
      1,
      policy,
    )
    is Err(_),
  )
  let swapped = profile.tensors.map(record => {
    if record.role == "block" {
      { ..record, block: Some(1 - record.block.unwrap()), }
    } else {
      record
    }
  })
  assert_true(
    @model.plan(
      { ..profile, tensors: swapped, layer_bytes: [608UL, 304UL], },
      16,
      @core.FP16KV,
      2,
      policy,
    )
    is Err(_),
  )
  let forged = profile.tensors.map(record => {
    if record.name == "blk.0.attn_q.weight" {
      { ..record, type_id: 999, }
    } else {
      record
    }
  })
  assert_true(
    @model.plan({ ..profile, tensors: forged, }, 16, @core.FP16KV, 0, policy)
    is Err(_),
  )
}

///|
test "aligned artifact structures support scale-inclusive quantized KV payloads" {
  let original = inventory_fixture("qwen2", false, false)
  let wider = replace_metadata(
    replace_metadata(original, "qwen2.embedding_length", @gguf.UIntValue(64UL)),
    "qwen2.feed_forward_length",
    @gguf.UIntValue(128UL),
  )
  let tensors = original.tensors.map(tensor => {
    let shape = if tensor.name == "token_embd.weight" ||
      tensor.name == "output.weight" {
      [64UL, 8UL]
    } else if tensor.name.has_suffix("attn_q.weight") ||
      tensor.name.has_suffix("attn_output.weight") {
      [64UL, 64UL]
    } else if tensor.name.has_suffix("attn_k.weight") ||
      tensor.name.has_suffix("attn_v.weight") {
      [64UL, 32UL]
    } else if tensor.name.has_suffix("ffn_gate.weight") ||
      tensor.name.has_suffix("ffn_up.weight") {
      [64UL, 128UL]
    } else if tensor.name.has_suffix("ffn_down.weight") {
      [128UL, 64UL]
    } else {
      [64UL]
    }
    fixture_tensor(tensor.name, shape, false)
  })
  let profile = @model.extract_profile({ ..wider, tensors, }).unwrap()
  for
    (precision, expected) in [
      (@core.FP16KV, 2097152.0),
      (@core.Q8KV, 1114112.0),
      (@core.Q4KV, 589824.0),
    ] {
    let plan = @model.plan(profile, 8192, precision, 2, model_zero_policy()).unwrap()
    assert_eq(plan.kv_bytes, expected)
    assert_eq(plan.allocated_context, 8192)
    assert_eq(plan.allocated_kv_bytes, expected)
    assert_eq(plan.ledger.gpu_kv_bytes, expected / 2.0)
    assert_eq(plan.ledger.host_kv_bytes, expected / 2.0)
  }
}

///|
test "payload domain is checked independently of large integer file offsets" {
  let original = inventory_fixture("qwen2", false, false)
  let huge_file = {
    ..original,
    file_size: 9007199254741099UL,
    data_offset: 9007199254740993UL,
  }
  assert_eq(
    @model.extract_profile(huge_file).unwrap().tensor_payload_bytes,
    1488UL,
  )
  let bad_elements = original.tensors.map(tensor => {
    if tensor.name == "token_embd.weight" {
      { ..tensor, element_count: 9007199254740992UL, }
    } else {
      tensor
    }
  })
  assert_true(
    @model.extract_profile({ ..original, tensors: bad_elements, }) is Err(_),
  )
  let bad_bytes = original.tensors.map(tensor => {
    if tensor.name == "token_embd.weight" {
      { ..tensor, byte_size: Some(9007199254740992UL), }
    } else {
      tensor
    }
  })
  assert_true(
    @model.extract_profile({ ..original, tensors: bad_bytes, }) is Err(_),
  )
}
