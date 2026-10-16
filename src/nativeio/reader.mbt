///|
#borrow(path)
extern "C" fn open_ffi(path : Bytes) -> Int = "llmcalc_open_readonly"

///|
extern "C" fn size_ffi(fd : Int) -> Int64 = "llmcalc_file_size"

///|
#borrow(buffer)
extern "C" fn read_ffi(
  fd : Int,
  position : Int64,
  buffer : FixedArray[Byte],
  length : Int,
) -> Int = "llmcalc_pread_exact"

///|
extern "C" fn close_ffi(fd : Int) -> Int = "llmcalc_close_file"

///|
pub(all) struct IoStats {
  read_calls : Int
  bytes_read : UInt64
  max_read_end : UInt64
  max_request : Int
} derive(Eq, Debug)

///|
pub extend IoStats with Eq::{equal, not_equal}

///|
pub extend IoStats with @debug.Debug::{to_repr}

///|
pub(all) struct Scan {
  model : @gguf.Model
  io : IoStats
} derive(Debug)

///|
pub extend Scan with @debug.Debug::{to_repr}

///|
/// Opens a regular file read-only. The captured descriptor cannot escape the
/// synchronous parse call and is closed on both success and error paths.
pub fn inspect_file(
  path : String,
  limits : @gguf.Limits,
) -> Result[Scan, String] {
  if path.contains("\u0000") {
    return Err("file path contains NUL")
  }
  let encoded = @utf8.encode((path + "\u0000")[:])
  let fd = open_ffi(encoded)
  if fd < 0 {
    return Err("cannot open model file (OS error \{-fd})")
  }
  let size = size_ffi(fd)
  if size < 0L {
    let _ = close_ffi(fd)
    return Err("cannot stat regular model file (OS error \{-size})")
  }
  let file_size = size.reinterpret_as_uint64()
  let mut calls = 0
  let mut bytes_read = 0UL
  let mut max_read_end = 0UL
  let mut max_request = 0
  let reader : @gguf.ReaderAt = {
    size: file_size,
    read: fn(position, length) {
      if length < 0 || length > 16777216 {
        return Err("read request exceeds the 16 MiB adapter limit")
      }
      if position > file_size || length.to_uint64() > file_size - position {
        return Err("read range is outside the file")
      }
      if position > 9223372036854775807UL {
        return Err("file offset is outside signed 64-bit OS range")
      }
      let buffer = FixedArray::make(length, b'\x00')
      let actual = read_ffi(fd, position.reinterpret_as_int64(), buffer, length)
      calls += 1
      if actual < 0 {
        return Err("model read failed (OS error \{-actual})")
      }
      bytes_read += actual.to_uint64()
      let end = position + actual.to_uint64()
      if end > max_read_end {
        max_read_end = end
      }
      if length > max_request {
        max_request = length
      }
      if actual != length {
        return Err("short read: model file may have changed during inspection")
      }
      Ok(Bytes::makei(length, i => buffer[i]))
    },
  }
  let result = @gguf.parse(reader, limits)
  let final_size = size_ffi(fd)
  let closed = close_ffi(fd)
  if final_size != size {
    return Err("model file size changed during inspection")
  }
  if closed < 0 {
    return Err("cannot close model file (OS error \{-closed})")
  }
  match result {
    Err(message) => Err(message)
    Ok(model) =>
      Ok({
        model,
        io: { read_calls: calls, bytes_read, max_read_end, max_request, },
      })
  }
}
