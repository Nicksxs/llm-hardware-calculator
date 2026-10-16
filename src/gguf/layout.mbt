///|
/// Block element count and serialized bytes, pinned to upstream GGML layouts.
/// Reserved/unknown IDs are deliberately not inferred from adjacent offsets.
pub fn tensor_layout(type_id : Int) -> (UInt64, UInt64)? {
  match type_id {
    0 => Some((1UL, 4UL))
    1 | 25 | 30 => Some((1UL, 2UL))
    2 | 20 => Some((32UL, 18UL))
    3 => Some((32UL, 20UL))
    6 => Some((32UL, 22UL))
    7 => Some((32UL, 24UL))
    8 => Some((32UL, 34UL))
    9 => Some((32UL, 36UL))
    10 => Some((256UL, 84UL))
    11 | 21 => Some((256UL, 110UL))
    12 => Some((256UL, 144UL))
    13 => Some((256UL, 176UL))
    14 => Some((256UL, 210UL))
    15 => Some((256UL, 292UL))
    16 => Some((256UL, 66UL))
    17 => Some((256UL, 74UL))
    18 => Some((256UL, 98UL))
    19 => Some((256UL, 50UL))
    22 => Some((256UL, 82UL))
    23 => Some((256UL, 136UL))
    24 => Some((1UL, 1UL))
    26 => Some((1UL, 4UL))
    27 | 28 => Some((1UL, 8UL))
    29 => Some((256UL, 56UL))
    34 => Some((256UL, 54UL))
    35 => Some((256UL, 66UL))
    39 => Some((32UL, 17UL))
    40 => Some((64UL, 36UL))
    41 => Some((128UL, 18UL))
    42 => Some((64UL, 18UL))
    _ => None
  }
}

///|
pub fn tensor_type_name(type_id : Int) -> String {
  match type_id {
    0 => "F32"
    1 => "F16"
    2 => "Q4_0"
    3 => "Q4_1"
    6 => "Q5_0"
    7 => "Q5_1"
    8 => "Q8_0"
    9 => "Q8_1"
    10 => "Q2_K"
    11 => "Q3_K"
    12 => "Q4_K"
    13 => "Q5_K"
    14 => "Q6_K"
    15 => "Q8_K"
    16 => "IQ2_XXS"
    17 => "IQ2_XS"
    18 => "IQ3_XXS"
    19 => "IQ1_S"
    20 => "IQ4_NL"
    21 => "IQ3_S"
    22 => "IQ2_S"
    23 => "IQ4_XS"
    24 => "I8"
    25 => "I16"
    26 => "I32"
    27 => "I64"
    28 => "F64"
    29 => "IQ1_M"
    30 => "BF16"
    34 => "TQ1_0"
    35 => "TQ2_0"
    39 => "MXFP4"
    40 => "NVFP4"
    41 => "Q1_0"
    42 => "Q2_0"
    _ => "UNKNOWN(\{type_id})"
  }
}
