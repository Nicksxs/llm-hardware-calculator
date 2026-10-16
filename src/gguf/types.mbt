///|
/// Offsets and file sizes remain integer-valued even above 2^53. A source must
/// return exactly the requested bytes or an error, without loading all payload.
pub(all) struct ReaderAt {
  size : UInt64
  read : (UInt64, Int) -> Result[Bytes, String]
}

///|
pub(all) struct Limits {
  max_header_bytes : UInt64
  max_metadata_entries : Int
  max_tensors : Int
  max_string_bytes : Int
  max_array_elements : UInt64
  max_retained_array_values : Int
} derive(Eq, Debug)

///|
pub extend Limits with Eq::{equal, not_equal}

///|
pub extend Limits with @debug.Debug::{to_repr}

///|
pub fn default_limits() -> Limits {
  {
    max_header_bytes: 67108864UL,
    max_metadata_entries: 100000,
    max_tensors: 200000,
    max_string_bytes: 1048576,
    max_array_elements: 2000000UL,
    max_retained_array_values: 256,
  }
}

///|
/// ArrayValue(type ID, original count, retained values, fully retained).
/// Large arrays are structurally scanned but not materialized or UTF-8 decoded.
pub(all) enum Value {
  UIntValue(UInt64)
  IntValue(Int64)
  FloatValue(Double)
  BoolValue(Bool)
  StringValue(String)
  ArrayValue(Int, UInt64, Array[Value], Bool)
} derive(Eq, Debug)

///|
pub extend Value with Eq::{equal, not_equal}

///|
pub extend Value with @debug.Debug::{to_repr}

///|
pub(all) struct Metadata {
  key : String
  value : Value
} derive(Eq, Debug)

///|
pub extend Metadata with Eq::{equal, not_equal}

///|
pub extend Metadata with @debug.Debug::{to_repr}

///|
pub(all) struct Tensor {
  name : String
  dimensions : Array[UInt64]
  type_id : Int
  offset : UInt64
  absolute_offset : UInt64
  element_count : UInt64
  byte_size : UInt64?
} derive(Eq, Debug)

///|
pub extend Tensor with Eq::{equal, not_equal}

///|
pub extend Tensor with @debug.Debug::{to_repr}

///|
pub(all) struct Model {
  version : Int
  file_size : UInt64
  data_offset : UInt64
  metadata : Array[Metadata]
  tensors : Array[Tensor]
  complete : Bool
  warnings : Array[String]
} derive(Eq, Debug)

///|
pub extend Model with Eq::{equal, not_equal}

///|
pub extend Model with @debug.Debug::{to_repr}

///|
pub fn metadata_uint(model : Model, key : String) -> UInt64? {
  for item in model.metadata {
    if item.key == key {
      return match item.value {
        UIntValue(value) => Some(value)
        IntValue(value) =>
          if value >= 0L {
            Some(value.reinterpret_as_uint64())
          } else {
            None
          }
        _ => None
      }
    }
  }
  None
}

///|
pub fn metadata_string(model : Model, key : String) -> String? {
  for item in model.metadata {
    if item.key == key {
      return match item.value {
        StringValue(value) => Some(value)
        _ => None
      }
    }
  }
  None
}
