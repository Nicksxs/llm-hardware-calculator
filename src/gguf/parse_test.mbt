///|
fn append_uint(out : Array[Byte], value : UInt64, width : Int) -> Unit {
  for i = 0; i < width; i = i + 1 {
    out.push(((value >> (8 * i)) & 255UL).to_int().to_byte())
  }
}

///|
fn append_string(out : Array[Byte], value : String) -> Unit {
  let bytes = @utf8.encode(value[:])
  append_uint(out, bytes.length().to_uint64(), 8)
  for byte in bytes {
    out.push(byte)
  }
}

///|
/// This independent wire-format writer creates only a directory. Any parser
/// attempt to read tensor payload fails, even though the virtual file is huge.
fn fixture(
  offset : UInt64,
  dimensions : Array[UInt64],
  type_id : Int,
  size : UInt64,
  alignment : UInt64,
  version : Int,
) -> @gguf.ReaderAt {
  let out : Array[Byte] = [b'G', b'G', b'U', b'F']
  append_uint(out, version.to_uint64(), 4)
  append_uint(out, 1UL, 8)
  append_uint(out, 1UL, 8)
  append_string(out, "general.alignment")
  append_uint(out, 4UL, 4)
  append_uint(out, alignment, 4)
  append_string(out, "test.weight")
  append_uint(out, dimensions.length().to_uint64(), 4)
  for dimension in dimensions {
    append_uint(out, dimension, 8)
  }
  append_uint(out, type_id.to_uint64(), 4)
  append_uint(out, offset, 8)
  let bytes = Bytes::makei(out.length(), i => out[i])
  {
    size,
    read: fn(position, length) {
      if position > bytes.length().to_uint64() ||
        length.to_uint64() > bytes.length().to_uint64() - position {
        return Err(
          "unit ReaderAt forbids reads beyond metadata and descriptors",
        )
      }
      Ok(Bytes::makei(length, i => bytes[position.to_int() + i]))
    },
  }
}

///|
test "GGUF v2 and v3 parse tensor directories without reading payload" {
  for version in [2, 3] {
    let model = @gguf.parse(
      fixture(0UL, [8UL, 2UL], 0, 4096UL, 32UL, version),
      @gguf.default_limits(),
    ).unwrap()
    assert_eq(model.version, version)
    assert_true(model.complete)
    assert_eq(model.tensors[0].element_count, 16UL)
    assert_eq(model.tensors[0].byte_size, Some(64UL))
    assert_eq(model.tensors[0].absolute_offset, model.data_offset)
  }
}

///|
test "GGUF offsets above 2^53 retain the low bit exactly" {
  let offset = 9007199254740993UL
  let model = @gguf.parse(
    fixture(offset, [1UL], 0, 18014398509481984UL, 1UL, 3),
    @gguf.default_limits(),
  ).unwrap()
  assert_eq(model.tensors[0].offset, offset)
  assert_eq(model.tensors[0].absolute_offset, model.data_offset + offset)
  assert_true(model.tensors[0].absolute_offset % 2UL != model.data_offset % 2UL)
}

///|
test "GGUF offset and tensor dimension products use checked UInt64 arithmetic" {
  assert_true(
    @gguf.parse(
      fixture(18446744073709551615UL, [1UL], 0, 18446744073709551615UL, 1UL, 3),
      @gguf.default_limits(),
    )
    is Err(_),
  )
  assert_true(
    @gguf.parse(
      fixture(
        0UL,
        [18446744073709551615UL, 2UL],
        0,
        18446744073709551615UL,
        1UL,
        3,
      ),
      @gguf.default_limits(),
    )
    is Err(_),
  )
  assert_true(
    @gguf.parse(
      fixture(0UL, [18446744073709551615UL], 0, 18446744073709551615UL, 1UL, 3),
      @gguf.default_limits(),
    )
    is Err(_),
  )
}

///|
test "unknown tensor types remain explicitly incomplete" {
  let model = @gguf.parse(
    fixture(0UL, [32UL], 777, 4096UL, 32UL, 3),
    @gguf.default_limits(),
  ).unwrap()
  assert_false(model.complete)
  assert_eq(model.tensors[0].byte_size, None)
  assert_eq(model.warnings.length(), 1)
}

///|
test "quantized rows include their scales and must be block aligned" {
  let fixtures = [
    (2, 32UL, 18UL),
    (8, 32UL, 34UL),
    (9, 32UL, 36UL),
    (12, 256UL, 144UL),
    (13, 256UL, 176UL),
    (14, 256UL, 210UL),
    (23, 256UL, 136UL),
    (30, 1UL, 2UL),
  ]
  for (type_id, block, bytes) in fixtures {
    let model = @gguf.parse(
      fixture(0UL, [block, 3UL], type_id, 4096UL, 32UL, 3),
      @gguf.default_limits(),
    ).unwrap()
    assert_eq(model.tensors[0].byte_size, Some(bytes * 3UL))
  }
  assert_true(
    @gguf.parse(
      fixture(0UL, [33UL], 8, 4096UL, 32UL, 3),
      @gguf.default_limits(),
    )
    is Err(_),
  )
}

///|
test "GGUF validates payload end against the physical file length" {
  assert_true(
    @gguf.parse(
      fixture(4096UL, [8UL], 0, 4096UL, 32UL, 3),
      @gguf.default_limits(),
    )
    is Err(_),
  )
  assert_true(
    @gguf.parse(
      fixture(0UL, [1024UL], 0, 4096UL, 32UL, 3),
      @gguf.default_limits(),
    )
    is Err(_),
  )
  assert_true(
    @gguf.parse(fixture(0UL, [0UL], 0, 4096UL, 32UL, 3), @gguf.default_limits())
    is Err(_),
  )
}

///|
test "GGUF validates alignment version and resource bounds" {
  assert_true(
    @gguf.parse(fixture(1UL, [1UL], 0, 4096UL, 32UL, 3), @gguf.default_limits())
    is Err(_),
  )
  assert_true(
    @gguf.parse(fixture(0UL, [1UL], 0, 4096UL, 3UL, 3), @gguf.default_limits())
    is Err(_),
  )
  assert_true(
    @gguf.parse(fixture(0UL, [1UL], 0, 4096UL, 32UL, 1), @gguf.default_limits())
    is Err(_),
  )
  let limited = { ..@gguf.default_limits(), max_tensors: 0, }
  assert_true(
    @gguf.parse(fixture(0UL, [1UL], 0, 4096UL, 32UL, 3), limited) is Err(_),
  )
  let invalid = { ..@gguf.default_limits(), max_retained_array_values: -1, }
  assert_true(
    @gguf.parse(fixture(0UL, [1UL], 0, 4096UL, 32UL, 3), invalid) is Err(_),
  )
}

///|
test "short ReaderAt results propagate as errors" {
  let reader : @gguf.ReaderAt = {
    size: 4096UL,
    read: fn(_offset, _length) { Ok(b"") },
  }
  assert_true(@gguf.parse(reader, @gguf.default_limits()) is Err(_))
}

///|
test "empty GGUF directory does not require unused trailing data padding" {
  let out : Array[Byte] = [b'G', b'G', b'U', b'F']
  append_uint(out, 3UL, 4)
  append_uint(out, 0UL, 8)
  append_uint(out, 0UL, 8)
  let reader : @gguf.ReaderAt = {
    size: 24UL,
    read: fn(position, length) {
      Ok(Bytes::makei(length, i => out[position.to_int() + i]))
    },
  }
  let model = @gguf.parse(reader, @gguf.default_limits()).unwrap()
  assert_eq(model.tensors.length(), 0)
  assert_eq(model.data_offset, 32UL)
}

///|
test "general alignment requires the UINT32 wire type" {
  let base = fixture(0UL, [1UL], 0, 4096UL, 32UL, 3)
  let reader : @gguf.ReaderAt = {
    size: base.size,
    read: fn(position, length) {
      if position == 49UL && length == 4 {
        Ok(b"\x00\x00\x00\x00")
      } else {
        (base.read)(position, length)
      }
    },
  }
  match @gguf.parse(reader, @gguf.default_limits()) {
    Err(message) => assert_true(message.contains("UINT32"))
    Ok(_) => fail("alignment with a non-UINT32 wire type must fail")
  }
}
