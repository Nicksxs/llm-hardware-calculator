///|
fn metadata_value(model : @gguf.Model, key : String) -> @gguf.Value? {
  for entry in model.metadata {
    if entry.key == key {
      return Some(entry.value)
    }
  }
  None
}

///|
fn metadata_integer(model : @gguf.Model, key : String) -> Result[Int, String] {
  match metadata_value(model, key) {
    Some(@gguf.UIntValue(value)) =>
      if value <= 2147483647UL {
        Ok(value.to_int())
      } else {
        Err("metadata integer out of range: " + key)
      }
    Some(@gguf.IntValue(value)) =>
      if value >= 0L && value <= 2147483647L {
        Ok(value.to_int())
      } else {
        Err("metadata integer out of range: " + key)
      }
    _ => Err("required scalar integer metadata missing or invalid: " + key)
  }
}

///|
fn optional_integer(
  model : @gguf.Model,
  key : String,
  fallback : Int,
) -> Result[Int, String] {
  match metadata_value(model, key) {
    None => Ok(fallback)
    Some(_) => metadata_integer(model, key)
  }
}

///|
fn supported_type(type_id : Int) -> Bool {
  match type_id {
    0
    | 1
    | 2
    | 3
    | 6
    | 7
    | 8
    | 10
    | 11
    | 12
    | 13
    | 14
    | 16
    | 17
    | 18
    | 19
    | 20
    | 21
    | 22
    | 23
    | 29
    | 30 => true
    _ => false
  }
}

///|
fn layer_leaf(architecture : String, leaf : String) -> Bool {
  match leaf {
    "attn_norm.weight"
    | "attn_q.weight"
    | "attn_k.weight"
    | "attn_v.weight"
    | "attn_output.weight"
    | "ffn_norm.weight"
    | "ffn_gate.weight"
    | "ffn_down.weight"
    | "ffn_up.weight"
    | "attn_q.bias"
    | "attn_k.bias"
    | "attn_v.bias" => true
    "attn_output.bias" | "ffn_gate.bias" | "ffn_down.bias" | "ffn_up.bias" =>
      architecture == "llama"
    _ => false
  }
}

///|
fn check_shape(
  tensor : @gguf.Tensor,
  expected : Array[UInt64],
) -> Result[Unit, String] {
  if tensor.dimensions != expected {
    Err("tensor shape incompatible with architecture metadata: " + tensor.name)
  } else {
    Ok(())
  }
}

///|
fn check_tensor_storage(tensor : @gguf.Tensor) -> Result[Unit, String] {
  if tensor.dimensions.is_empty() || tensor.dimensions.length() > 4 {
    return Err("planner requires nonempty tensor dimensions: " + tensor.name)
  }
  let (block, storage) = match @gguf.tensor_layout(tensor.type_id) {
    Some(layout) => layout
    None => return Err("unknown tensor storage type: " + tensor.name)
  }
  if tensor.dimensions[0] % block != 0UL {
    return Err("quantized tensor row is not a complete block: " + tensor.name)
  }
  let mut elements = 1UL
  for dimension in tensor.dimensions {
    if dimension == 0UL || dimension > 9007199254740991UL / elements {
      return Err(
        "tensor element count exceeds planner exact-integer domain: " +
        tensor.name,
      )
    }
    elements *= dimension
  }
  if elements != tensor.element_count ||
    elements / block > 9007199254740991UL / storage ||
    tensor.byte_size != Some(elements / block * storage) {
    return Err(
      "tensor dimensions, element count and byte size disagree: " + tensor.name,
    )
  }
  Ok(())
}

///|
/// Strict adapter for separate-Q/K/V llama and qwen2, one scalar structure for
/// every full-attention block. Other GGUF files remain inspectable by gguf.
pub fn extract_profile(model : @gguf.Model) -> Result[Profile, String] {
  if !model.complete {
    return Err("planning requires a complete known-type tensor inventory")
  }
  let architecture = match metadata_value(model, "general.architecture") {
    Some(@gguf.StringValue(value)) if value == "llama" || value == "qwen2" =>
      value
    _ =>
      return Err(
        "planner supports only dense full-attention llama and qwen2 architectures",
      )
  }
  let prefix = architecture + "."
  match metadata_value(model, prefix + "attention.causal") {
    Some(@gguf.BoolValue(false)) =>
      return Err("noncausal attention is outside the decoder planner scope")
    None | Some(@gguf.BoolValue(true)) => ()
    _ => return Err("attention.causal must be Boolean")
  }
  for entry in model.metadata {
    if entry.key.has_prefix(prefix) &&
      (
        entry.key.contains("expert") ||
        entry.key.contains("sliding") ||
        entry.key.contains("recurrent") ||
        entry.key.contains("ssm") ||
        entry.key.contains("nextn")
      ) {
      // A declared zero scalar is harmless; nonzero, arrays and nonnumeric
      // declarations are unsupported instead of silently assuming dense/full.
      match entry.value {
        @gguf.UIntValue(0UL) | @gguf.IntValue(0L) | @gguf.BoolValue(false) => ()
        _ =>
          return Err(
            "unsupported non-dense/full-attention metadata: " + entry.key,
          )
      }
    }
  }
  let layers = match metadata_integer(model, prefix + "block_count") {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let embedding = match metadata_integer(model, prefix + "embedding_length") {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let query_heads = match
    metadata_integer(model, prefix + "attention.head_count") {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let kv_heads = match
    optional_integer(model, prefix + "attention.head_count_kv", query_heads) {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let training_context = match
    metadata_integer(model, prefix + "context_length") {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let feed_forward = match
    metadata_integer(model, prefix + "feed_forward_length") {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  if embedding < 1 ||
    embedding > 1048576 ||
    training_context < 1 ||
    feed_forward < 1 ||
    feed_forward > 1048576 ||
    query_heads < 1 {
    return Err(
      "invalid embedding, feed-forward, context or head-count metadata",
    )
  }
  let default_head = if embedding % query_heads == 0 {
    embedding / query_heads
  } else {
    0
  }
  let key_dim = match
    optional_integer(model, prefix + "attention.key_length", default_head) {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let value_dim = match
    optional_integer(model, prefix + "attention.value_length", default_head) {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  if key_dim != value_dim {
    return Err("planner requires equal key and value head dimensions")
  }
  let rotary = match
    optional_integer(model, prefix + "rope.dimension_count", key_dim) {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  if rotary != key_dim {
    return Err(
      "llama/qwen2 planner requires rope.dimension_count equal to key/value head dimension",
    )
  }
  let shape : @core.Architecture = {
    layers,
    query_heads,
    kv_heads,
    head_dim: key_dim,
    source: "GGUF scalar architecture metadata; no parameter-count proxy",
  }
  match @core.validate_architecture(shape) {
    Err(message) => return Err(message)
    Ok(_) => ()
  }
  if architecture == "qwen2" && key_dim * query_heads != embedding {
    return Err(
      "qwen2 adapter requires head dimension times query heads = embedding length",
    )
  }
  let layer_bytes = Array::make(layers, 0UL)
  let records : Array[TensorRecord] = []
  let seen : Map[String, Bool] = Map([])
  let mut payload = 0UL
  let mut parameter_count = 0UL
  let mut input_bytes = 0UL
  let mut output_bytes = 0UL
  let mut output_present = false
  let mut vocabulary = 0UL
  for tensor in model.tensors {
    if seen.get(tensor.name) is Some(_) {
      return Err("duplicate tensor name: " + tensor.name)
    }
    seen[tensor.name] = true
    if !supported_type(tensor.type_id) {
      return Err("tensor type is not supported by planner: " + tensor.name)
    }
    match check_tensor_storage(tensor) {
      Ok(_) => ()
      Err(message) => return Err(message)
    }
    let bytes = match tensor.byte_size {
      Some(value) if value > 0UL => value
      _ => return Err("tensor has unknown or empty byte size: " + tensor.name)
    }
    payload = match exact_add(payload, bytes) {
      Ok(value) => value
      Err(message) => return Err(message)
    }
    parameter_count = match exact_add(parameter_count, tensor.element_count) {
      Ok(value) => value
      Err(message) => return Err(message)
    }
    let (role, block, expected) = match tensor.name {
      "token_embd.weight" => {
        if tensor.dimensions.length() != 2 ||
          tensor.dimensions[0] != embedding.to_uint64() ||
          tensor.dimensions[1] == 0UL {
          return Err(
            "token_embd.weight must have [embedding_length, vocabulary] dimensions",
          )
        }
        vocabulary = tensor.dimensions[1]
        input_bytes = bytes
        ("input", None, tensor.dimensions)
      }
      "output_norm.weight" => {
        output_bytes = match exact_add(output_bytes, bytes) {
          Ok(v) => v
          Err(m) => return Err(m)
        }
        ("output", None, [embedding.to_uint64()])
      }
      "output.weight" => {
        output_present = true
        output_bytes = match exact_add(output_bytes, bytes) {
          Ok(v) => v
          Err(m) => return Err(m)
        }
        // Vocabulary is checked in a second pass, independent of file order.
        ("output", None, tensor.dimensions)
      }
      "output.bias" if architecture == "qwen2" => {
        output_bytes = match exact_add(output_bytes, bytes) {
          Ok(v) => v
          Err(m) => return Err(m)
        }
        ("output", None, tensor.dimensions)
      }
      _ => {
        let rest = match tensor.name.strip_prefix("blk.") {
          Some(rest) => rest
          None => return Err("unsupported non-layer tensor: " + tensor.name)
        }
        let (index_text, leaf_view) = match rest.split_once(".") {
          Some(parts) => parts
          None => return Err("invalid block tensor name: " + tensor.name)
        }
        let index = @string.parse_int(index_text, base=10) catch {
          _ => return Err("invalid block index: " + tensor.name)
        }
        let leaf = leaf_view.to_owned()
        if index < 0 ||
          index >= layers ||
          index.to_string() != index_text.to_owned() ||
          !layer_leaf(architecture, leaf) {
          return Err("unsupported block tensor name or index: " + tensor.name)
        }
        layer_bytes[index] = match exact_add(layer_bytes[index], bytes) {
          Ok(v) => v
          Err(m) => return Err(m)
        }
        let embed = embedding.to_uint64()
        let query_width = (query_heads * key_dim).to_uint64()
        let kv_width = (kv_heads * key_dim).to_uint64()
        let ff = feed_forward.to_uint64()
        let expected = match leaf {
          "attn_norm.weight"
          | "ffn_norm.weight"
          | "attn_output.bias"
          | "ffn_down.bias" => [embed]
          "attn_q.weight" => [embed, query_width]
          "attn_k.weight" | "attn_v.weight" => [embed, kv_width]
          "attn_output.weight" => [query_width, embed]
          "ffn_gate.weight" | "ffn_up.weight" => [embed, ff]
          "ffn_down.weight" => [ff, embed]
          "attn_q.bias" => [query_width]
          "attn_k.bias" | "attn_v.bias" => [kv_width]
          "ffn_gate.bias" | "ffn_up.bias" => [ff]
          _ => return Err("unsupported tensor leaf: " + leaf)
        }
        ("block", Some(index), expected)
      }
    }
    match check_shape(tensor, expected) {
      Ok(_) => ()
      Err(message) => return Err(message)
    }
    records.push({
      name: tensor.name,
      byte_size: bytes,
      role,
      block,
      dimensions: tensor.dimensions.copy(),
      type_id: tensor.type_id,
      element_count: tensor.element_count,
    })
  }
  if input_bytes == 0UL || seen.get("output_norm.weight") is None {
    return Err("required token_embd.weight or output_norm.weight is missing")
  }
  match metadata_value(model, "tokenizer.ggml.tokens") {
    Some(@gguf.ArrayValue(8, count, _, _)) if count == vocabulary => ()
    Some(_) =>
      return Err(
        "tokenizer token array length/type does not match embedding vocabulary",
      )
    None => ()
  }
  for tensor in model.tensors {
    if tensor.name == "output.weight" {
      match check_shape(tensor, [embedding.to_uint64(), vocabulary]) {
        Ok(_) => ()
        Err(message) => return Err(message)
      }
    } else if tensor.name == "output.bias" {
      match check_shape(tensor, [vocabulary]) {
        Ok(_) => ()
        Err(message) => return Err(message)
      }
    }
  }
  for index in 0..<layers {
    for
      leaf in [
        "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_norm.weight", "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight",
      ] {
      let required = "blk." + index.to_string() + "." + leaf
      if seen.get(required) is None {
        return Err("required block tensor missing: " + required)
      }
    }
  }
  let warnings = [
    "Tensor payloads exclude backend buffer alignment, repacking, graph buffers and driver allocations.",
    "Placement assumes single discrete GPU, standard CPU buffers, no repack/host/ACCEL overrides, and supported preferred-device kernels; fallback is not simulated.",
    "Only separate Q/K/V dense full-attention tensors are supported; fused QKV, rope-factor tensors and adapters require separate support.",
    "Stored tensor element count is reported; it is not certified trainable parameter count.",
    "A parsed tensor directory does not certify payload contents, tokenizer validity or successful backend loading.",
  ]
  Ok({
    architecture,
    shape,
    context_length: training_context,
    embedding_length: embedding,
    tensor_payload_bytes: payload,
    parameter_count,
    layer_bytes,
    input_bytes,
    output_bytes,
    tied_output: !output_present,
    tensors: records,
    backend_revision: backend_revision(),
    warnings,
  })
}
