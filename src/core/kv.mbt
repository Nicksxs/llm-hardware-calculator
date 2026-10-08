///|
/// Dense full-attention model shape. K and V use the same head dimension.
/// A parameter-count proxy is a disclosed planning assumption, not detection.
pub(all) struct Architecture {
  layers : Int
  kv_heads : Int
  head_dim : Int
  query_heads : Int
  source : String
} derive(Eq, Debug)

///|
pub extend Architecture with Eq::{equal, not_equal}

///|
pub extend Architecture with @debug.Debug::{to_repr}

///|
/// Storage recipes describe KV tensor payload only. Backend support is not
/// guaranteed and runtime allocation/padding is budgeted separately.
pub(all) enum KvPrecision {
  FP16KV
  Q8KV
  Q4KV
} derive(Eq, Debug)

///|
pub extend KvPrecision with Eq::{equal, not_equal}

///|
pub extend KvPrecision with @debug.Debug::{to_repr}

///|
/// The basic proxy uses an upper-size bucket from published Qwen2.5 configs.
/// Other architectures can have substantially different KV memory. Models
/// above 72B require an explicit architecture instead of silent extrapolation.
pub fn proxy_architecture(params_b : Double) -> Result[Architecture, String] {
  if !(params_b > 0.0 && params_b <= 72.0) {
    return Err(
      "basic architecture proxy requires finite params in (0, 72] B; use explicit architecture above 72B",
    )
  }
  let (layers, kv_heads, query_heads, source) = if params_b <= 7.0 {
    (
      28, 4, 28, "Qwen2.5-7B-Instruct proxy: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/a09a35458c702b33eeacc393d103063234e8bc28/config.json",
    )
  } else if params_b <= 14.0 {
    (
      48, 8, 40, "Qwen2.5-14B-Instruct proxy: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/raw/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/config.json",
    )
  } else if params_b <= 32.0 {
    (
      64, 8, 40, "Qwen2.5-32B-Instruct proxy: https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/raw/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd/config.json",
    )
  } else {
    (
      80, 8, 64, "Qwen2.5-72B-Instruct proxy: https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/raw/495f39366efef23836d0cfae4fbe635880d2be31/config.json",
    )
  }
  Ok({ layers, kv_heads, head_dim: 128, query_heads, source, })
}

///|
pub fn validate_architecture(
  architecture : Architecture,
) -> Result[Architecture, String] {
  if architecture.layers < 1 || architecture.layers > 512 {
    return Err("layers must be in [1, 512]")
  }
  if architecture.kv_heads < 1 || architecture.kv_heads > 512 {
    return Err("kv-heads must be in [1, 512]")
  }
  if architecture.head_dim < 1 || architecture.head_dim > 1024 {
    return Err("head-dim must be in [1, 1024]")
  }
  if architecture.query_heads < architecture.kv_heads ||
    architecture.query_heads > 512 {
    return Err("query-heads must be at least kv-heads and no more than 512")
  }
  Ok(architecture)
}

///|
pub fn kv_precision_name(precision : KvPrecision) -> String {
  match precision {
    FP16KV => "FP16"
    Q8KV => "Q8_0"
    Q4KV => "Q4_0"
  }
}

///|
pub fn parse_kv_precision(value : String) -> Result[KvPrecision, String] {
  match value.to_lower() {
    "fp16" | "f16" => Ok(FP16KV)
    "q8" | "q8_0" => Ok(Q8KV)
    "q4" | "q4_0" => Ok(Q4KV)
    _ => Err("unknown KV precision: " + value)
  }
}

///|
/// Returns bytes for one full-attention sequence with uniform layers:
/// tokens * layers * KV heads * head dimension * (K + V) * bytes/element.
/// FP16 is 2 B; GGML Q8_0 is 34/32 B; Q4_0 is 18/32 B. Conversion to
/// Double happens before multiplication so valid large shapes do not overflow
/// Int. Quantized rows must contain complete blocks of 32 elements.
pub fn estimate_kv(
  context : Int,
  architecture : Architecture,
  precision : KvPrecision,
) -> Result[Double, String] {
  if context < 1 || context > 1048576 {
    return Err("context must be in [1, 1048576] tokens")
  }
  match validate_architecture(architecture) {
    Err(message) => return Err(message)
    Ok(_) => ()
  }
  let bytes_per_element = match precision {
    FP16KV => 2.0
    Q8KV => 34.0 / 32.0
    Q4KV => 18.0 / 32.0
  }
  if precision != FP16KV && architecture.head_dim % 32 != 0 {
    return Err("quantized KV requires head-dim divisible by 32")
  }
  Ok(
    context.to_double() *
    architecture.layers.to_double() *
    architecture.kv_heads.to_double() *
    architecture.head_dim.to_double() *
    2.0 *
    bytes_per_element,
  )
}
