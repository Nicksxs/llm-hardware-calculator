///|
test "raw integer JSON preserves values above 2^53 and UInt64 maximum" {
  for
    (value, expected) in [
      (9007199254740993UL, "9007199254740993"),
      (18446744073709551615UL, "18446744073709551615"),
    ] {
    assert_eq(@json.parse(exact(value).stringify()), Json(expected))
  }
  assert_eq(@json.parse(optional_exact(None).stringify()), Json::null())
}

///|
test "metadata array summary preserves exact total count and omission status" {
  let metadata : @gguf.Metadata = {
    key: "tokenizer.test",
    value: @gguf.ArrayValue(8, 9007199254740993UL, [], false),
  }
  guard metadata_json(metadata) is Object(fields) else {
    fail("metadata must be an object")
  }
  guard fields.get("value") is Some(Object(value)) else {
    fail("metadata value must be tagged")
  }
  assert_eq(value.get("kind"), Some(Json("array")))
  assert_eq(value.get("total_count"), Some(Json("9007199254740993")))
  assert_eq(value.get("retention_complete"), Some(Json(false)))
  assert_eq(value.get("omitted"), Some(Json(true)))
  assert_eq(value.get("value"), Some(Json::null()))
}

///|
test "metadata integers are tagged decimal strings rather than rounded numbers" {
  let samples : Array[(@gguf.Metadata, String)] = [
    (
      { key: "unsigned", value: @gguf.UIntValue(9007199254740993UL), },
      "9007199254740993",
    ),
    ({ key: "signed", value: @gguf.IntValue(-1L), }, "-1"),
  ]
  for (metadata, expected) in samples {
    guard metadata_json(metadata) is Object(fields) else {
      fail("metadata must be an object")
    }
    guard fields.get("value") is Some(Object(value)) else {
      fail("metadata value must be tagged")
    }
    assert_eq(value.get("value"), Some(Json(expected)))
  }
}

///|
test "unknown tensor JSON retains exact offsets and exposes null byte size" {
  let tensor : @gguf.Tensor = {
    name: "unknown.weight",
    dimensions: [9007199254740993UL],
    type_id: 999,
    offset: 9007199254740993UL,
    absolute_offset: 9007199254741025UL,
    element_count: 9007199254740993UL,
    byte_size: None,
  }
  guard tensor_json(tensor) is Object(fields) else {
    fail("tensor must be an object")
  }
  assert_eq(fields.get("offset"), Some(Json("9007199254740993")))
  assert_eq(fields.get("absolute_offset"), Some(Json("9007199254741025")))
  assert_eq(fields.get("element_count"), Some(Json("9007199254740993")))
  assert_eq(fields.get("byte_size"), Some(Json::null()))
  assert_eq(
    fields.get("dimensions"),
    Some(Json::array([Json("9007199254740993")])),
  )
}

///|
test "artifact maximum offload does not claim full GPU residency" {
  assert_eq(artifact_status(@core.FullGPU), "MAX_OFFLOAD")
  assert_eq(artifact_status(@core.Offload), "GPU+CPU OFFLOAD")
  assert_eq(artifact_status(@core.CPUOnly), "CPU ONLY")
  assert_eq(artifact_status(@core.Insufficient), "INSUFFICIENT")
  let record : @model.TensorRecord = {
    name: "token_embd.weight",
    byte_size: 9007199254740993UL,
    role: "input",
    block: None,
    dimensions: [9007199254740993UL],
    type_id: 999,
    element_count: 9007199254740993UL,
  }
  guard tensor_record_json(record) is Object(fields) else {
    fail("tensor record must be an object")
  }
  assert_eq(fields.get("byte_size"), Some(Json("9007199254740993")))
  assert_eq(fields.get("block"), Some(Json::null()))
}
