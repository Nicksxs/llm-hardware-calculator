///|
priv suberror FormatError {
  FormatError(String)
}

///|
priv struct Cursor {
  reader : ReaderAt
  limits : Limits
  mut position : UInt64
}

///|
fn checked_add(a : UInt64, b : UInt64) -> UInt64 raise FormatError {
  if b > 18446744073709551615UL - a {
    raise FormatError("UInt64 offset overflow")
  }
  a + b
}

///|
fn checked_mul(a : UInt64, b : UInt64) -> UInt64 raise FormatError {
  if a != 0UL && b > 18446744073709551615UL / a {
    raise FormatError("UInt64 size overflow")
  }
  a * b
}

///|
fn Cursor::skip(self : Cursor, length : UInt64) -> Unit raise FormatError {
  let end = checked_add(self.position, length)
  if end > self.reader.size {
    raise FormatError("truncated GGUF header")
  }
  if end > self.limits.max_header_bytes {
    raise FormatError("GGUF header exceeds configured byte limit")
  }
  self.position = end
}

///|
fn Cursor::take(self : Cursor, length : Int) -> Bytes raise FormatError {
  if length < 0 || length > 16777216 {
    raise FormatError("invalid bounded read length")
  }
  let start = self.position
  self.skip(length.to_uint64())
  match (self.reader.read)(start, length) {
    Err(message) => raise FormatError("read at \{start}: " + message)
    Ok(bytes) => {
      if bytes.length() != length {
        raise FormatError("ReaderAt returned a short read")
      }
      bytes
    }
  }
}

///|
fn Cursor::unsigned(self : Cursor, width : Int) -> UInt64 raise FormatError {
  let bytes = self.take(width)
  let mut value = 0UL
  for i = 0; i < width; i = i + 1 {
    value = value | (bytes[i].to_int().to_uint64() << (8 * i))
  }
  value
}

///|
fn Cursor::signed(self : Cursor, width : Int) -> Int64 raise FormatError {
  let raw = self.unsigned(width)
  if width == 8 {
    return raw.reinterpret_as_int64()
  }
  let sign = 1UL << (width * 8 - 1)
  if (raw & sign) != 0UL {
    (raw | (18446744073709551615UL << (width * 8))).reinterpret_as_int64()
  } else {
    raw.reinterpret_as_int64()
  }
}

///|
fn Cursor::string(self : Cursor, retain : Bool) -> String raise FormatError {
  let length = self.unsigned(8)
  if length > self.limits.max_string_bytes.to_uint64() {
    raise FormatError("GGUF string exceeds configured limit")
  }
  if !retain {
    self.skip(length)
    return ""
  }
  let bytes = self.take(length.to_int())
  @utf8.decode(bytes[:], ignore_bom=false) catch {
    _ => raise FormatError("invalid UTF-8 in retained GGUF string")
  }
}

///|
fn Cursor::scalar(
  self : Cursor,
  type_id : Int,
  retain : Bool,
) -> Value raise FormatError {
  match type_id {
    0 => UIntValue(self.unsigned(1))
    1 => IntValue(self.signed(1))
    2 => UIntValue(self.unsigned(2))
    3 => IntValue(self.signed(2))
    4 => UIntValue(self.unsigned(4))
    5 => IntValue(self.signed(4))
    6 =>
      FloatValue(
        Float::reinterpret_from_int(self.unsigned(4).to_int()).to_double(),
      )
    7 => {
      let boolean = self.unsigned(1)
      if boolean > 1UL {
        raise FormatError("invalid GGUF boolean")
      }
      BoolValue(boolean == 1UL)
    }
    8 => StringValue(self.string(retain))
    10 => UIntValue(self.unsigned(8))
    11 => IntValue(self.signed(8))
    12 => FloatValue(self.unsigned(8).reinterpret_as_double())
    9 => raise FormatError("nested metadata arrays are unsupported")
    _ => raise FormatError("unknown metadata value type \{type_id}")
  }
}

///|
fn metadata_width(type_id : Int) -> Int {
  match type_id {
    0 | 1 | 7 => 1
    2 | 3 => 2
    4 | 5 | 6 => 4
    10 | 11 | 12 => 8
    _ => 0
  }
}

///|
fn Cursor::value(self : Cursor, type_id : Int) -> Value raise FormatError {
  if type_id != 9 {
    return self.scalar(type_id, true)
  }
  let element_type = self.unsigned(4)
  if element_type > 12UL || element_type == 9UL {
    raise FormatError("unknown or nested metadata array element type")
  }
  let count = self.unsigned(8)
  if count > self.limits.max_array_elements {
    raise FormatError("metadata array exceeds configured element limit")
  }
  let retain = count <= self.limits.max_retained_array_values.to_uint64()
  let values : Array[Value] = []
  if !retain && element_type != 8UL && element_type != 7UL {
    self.skip(
      checked_mul(count, metadata_width(element_type.to_int()).to_uint64()),
    )
  } else {
    for i = 0UL; i < count; i = i + 1UL {
      let value = self.scalar(element_type.to_int(), retain)
      if retain {
        values.push(value)
      }
    }
  }
  ArrayValue(element_type.to_int(), count, values, retain)
}

///|
fn parse_internal(
  reader : ReaderAt,
  limits : Limits,
) -> Model raise FormatError {
  if limits.max_header_bytes < 24UL ||
    limits.max_header_bytes > 1073741824UL ||
    limits.max_metadata_entries < 0 ||
    limits.max_metadata_entries > 1000000 ||
    limits.max_tensors < 0 ||
    limits.max_tensors > 1000000 ||
    limits.max_string_bytes < 1 ||
    limits.max_string_bytes > 16777216 ||
    limits.max_array_elements > 10000000UL ||
    limits.max_retained_array_values < 0 ||
    limits.max_retained_array_values > 4096 {
    raise FormatError("invalid parser resource limits")
  }
  let c = { reader, limits, position: 0UL, }
  if c.take(4) != b"GGUF" {
    raise FormatError("not a little-endian GGUF file")
  }
  let version = c.unsigned(4)
  if version != 2UL && version != 3UL {
    raise FormatError("only little-endian GGUF v2/v3 are supported")
  }
  let tensor_count = c.unsigned(8)
  let metadata_count = c.unsigned(8)
  if tensor_count > limits.max_tensors.to_uint64() ||
    metadata_count > limits.max_metadata_entries.to_uint64() {
    raise FormatError("GGUF entry count exceeds configured limits")
  }
  let metadata : Array[Metadata] = []
  let keys : Map[String, Bool] = Map([])
  let warnings : Array[String] = []
  let mut alignment = 32UL
  for i = 0UL; i < metadata_count; i = i + 1UL {
    let key = c.string(true)
    if key.is_empty() || @utf8.encode(key[:]).length() > 1024 {
      raise FormatError("invalid GGUF metadata key byte length")
    }
    if keys.contains(key) {
      raise FormatError("duplicate metadata key: " + key)
    }
    keys[key] = true
    let type_id = c.unsigned(4)
    if type_id > 12UL {
      raise FormatError("unknown metadata value type \{type_id}")
    }
    if key == "general.alignment" && type_id != 4UL {
      raise FormatError("general.alignment must have UINT32 metadata type")
    }
    let value = c.value(type_id.to_int())
    if key == "general.alignment" {
      alignment = match value {
        UIntValue(n) => n
        _ => raise FormatError("general.alignment must be unsigned")
      }
      if alignment == 0UL ||
        alignment > 65536UL ||
        (alignment & (alignment - 1UL)) != 0UL {
        raise FormatError("alignment must be a power of two in [1, 65536]")
      }
    }
    if value is ArrayValue(_, _, _, false) {
      warnings.push("array values omitted by retention limit: " + key)
    }
    metadata.push({ key, value, })
  }
  let tensors : Array[Tensor] = []
  let names : Map[String, Bool] = Map([])
  let mut complete = true
  for i = 0UL; i < tensor_count; i = i + 1UL {
    let name = c.string(true)
    if name.is_empty() || @utf8.encode(name[:]).length() > 64 {
      raise FormatError("tensor name must contain 1..64 UTF-8 bytes")
    }
    if names.contains(name) {
      raise FormatError("duplicate tensor name: " + name)
    }
    names[name] = true
    let ndims = c.unsigned(4)
    if ndims < 1UL || ndims > 4UL {
      raise FormatError("tensor dimension count must be in [1, 4]")
    }
    let dimensions : Array[UInt64] = []
    let mut elements = 1UL
    for j = 0UL; j < ndims; j = j + 1UL {
      let dimension = c.unsigned(8)
      if dimension == 0UL {
        raise FormatError("zero tensor dimension")
      }
      elements = checked_mul(elements, dimension)
      dimensions.push(dimension)
    }
    let type_id = c.unsigned(4)
    if type_id > 2147483647UL {
      raise FormatError("tensor type ID exceeds supported integer range")
    }
    let offset = c.unsigned(8)
    if offset % alignment != 0UL {
      raise FormatError("unaligned tensor offset: " + name)
    }
    let byte_size = match tensor_layout(type_id.to_int()) {
      Some((block, bytes)) => {
        if dimensions[0] % block != 0UL {
          raise FormatError(
            "tensor row is not aligned to quantization block: " + name,
          )
        }
        Some(checked_mul(elements / block, bytes))
      }
      None => {
        complete = false
        warnings.push("unknown tensor type \{type_id}: " + name)
        None
      }
    }
    tensors.push({
      name,
      dimensions,
      type_id: type_id.to_int(),
      offset,
      absolute_offset: 0UL,
      element_count: elements,
      byte_size,
    })
  }
  let padding = (alignment - c.position % alignment) % alignment
  let data_offset = checked_add(c.position, padding)
  if tensor_count > 0UL && data_offset > reader.size {
    raise FormatError("truncated GGUF data alignment padding")
  }
  for i = 0; i < tensors.length(); i = i + 1 {
    let tensor = tensors[i]
    let absolute_offset = checked_add(data_offset, tensor.offset)
    if absolute_offset >= reader.size {
      raise FormatError("tensor offset outside file: " + tensor.name)
    }
    if tensor.byte_size is Some(bytes) {
      if checked_add(absolute_offset, bytes) > reader.size {
        raise FormatError("tensor payload extends beyond file: " + tensor.name)
      }
    }
    tensors[i] = { ..tensor, absolute_offset, }
  }
  let order : Array[(UInt64, Int)] = []
  for i = 0; i < tensors.length(); i = i + 1 {
    order.push((tensors[i].absolute_offset, i))
  }
  order.sort_by(fn(a, b) {
    if a.0 < b.0 {
      -1
    } else if a.0 > b.0 {
      1
    } else {
      0
    }
  })
  let mut last_end = data_offset
  let mut last_start : UInt64? = None
  for (start, index) in order {
    if start < last_end || last_start == Some(start) {
      raise FormatError("overlapping tensor data ranges")
    }
    last_start = Some(start)
    if tensors[index].byte_size is Some(bytes) {
      last_end = checked_add(start, bytes)
    }
  }
  {
    version: version.to_int(),
    file_size: reader.size,
    data_offset,
    metadata,
    tensors,
    complete,
    warnings,
  }
}

///|
pub fn parse(reader : ReaderAt, limits : Limits) -> Result[Model, String] {
  Ok(parse_internal(reader, limits)) catch {
    FormatError(message) => Err(message)
  }
}
