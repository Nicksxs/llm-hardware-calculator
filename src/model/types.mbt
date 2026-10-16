///|
/// Stored tensors only; a tied output alias is represented by the Profile flag.
pub(all) struct TensorRecord {
  name : String
  byte_size : UInt64
  role : String
  block : Int?
  dimensions : Array[UInt64]
  type_id : Int
  element_count : UInt64
} derive(Eq, Debug)

///|
pub extend TensorRecord with Eq::{equal, not_equal}

///|
pub extend TensorRecord with @debug.Debug::{to_repr}

///|
/// A validated dense full-attention inventory. Output bytes include only stored
/// output tensors; absent output.weight reuses/duplicates the input embedding.
pub(all) struct Profile {
  architecture : String
  shape : @core.Architecture
  context_length : Int
  embedding_length : Int
  tensor_payload_bytes : UInt64
  parameter_count : UInt64
  layer_bytes : Array[UInt64]
  input_bytes : UInt64
  output_bytes : UInt64
  tied_output : Bool
  tensors : Array[TensorRecord]
  backend_revision : String
  warnings : Array[String]
} derive(Eq, Debug)

///|
pub extend Profile with Eq::{equal, not_equal}

///|
pub extend Profile with @debug.Debug::{to_repr}

///|
/// A physical payload assignment; tied output on GPU creates a second copy.
pub(all) struct TensorPlacement {
  name : String
  role : String
  device : String
  byte_size : UInt64
  is_duplicate : Bool
} derive(Eq, Debug)

///|
pub extend TensorPlacement with Eq::{equal, not_equal}

///|
pub extend TensorPlacement with @debug.Debug::{to_repr}

///|
pub(all) struct Plan {
  n_gpu_layers : Int
  gpu_block_count : Int
  output_on_gpu : Bool
  gpu_weight_bytes : UInt64
  cpu_weight_bytes : UInt64
  duplicated_bytes : UInt64
  /// Raw payload for the requested context, before backend token padding.
  kv_bytes : Double
  allocated_context : Int
  allocated_kv_bytes : Double
  ledger : @core.MemoryLedger
  placements : Array[TensorPlacement]
  backend_policy : String
  warnings : Array[String]
} derive(Eq, Debug)

///|
pub extend Plan with Eq::{equal, not_equal}

///|
pub extend Plan with @debug.Debug::{to_repr}

///|
pub(all) struct PlanSearch {
  status : @core.FitStatus
  plan : Plan?
  reason : String
} derive(Eq, Debug)

///|
pub extend PlanSearch with Eq::{equal, not_equal}

///|
pub extend PlanSearch with @debug.Debug::{to_repr}

///|
pub fn backend_revision() -> String {
  "88aaf325a7026ac1127d0897394b0a3d6413dc2a"
}

///|
/// Public byte totals must remain exactly representable when passed to the
/// existing Double-based capacity core. File offsets remain unrestricted UInt64.
fn exact_add(left : UInt64, right : UInt64) -> Result[UInt64, String] {
  let limit = 9007199254740991UL
  if left > limit || right > limit - left {
    Err("planner tensor totals must not exceed 2^53-1 bytes/elements")
  } else {
    Ok(left + right)
  }
}
