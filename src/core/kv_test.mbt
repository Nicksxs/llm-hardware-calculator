///|
fn kv_shape() -> @core.Architecture {
  {
    layers: 64,
    kv_heads: 8,
    head_dim: 128,
    query_heads: 40,
    source: "test shape",
  }
}

///|
test "KV known 32B GQA shape includes quantization block scales" {
  assert_eq(@core.estimate_kv(8192, kv_shape(), @core.FP16KV), Ok(2147483648.0))
  assert_eq(@core.estimate_kv(8192, kv_shape(), @core.Q8KV), Ok(1140850688.0))
  assert_eq(@core.estimate_kv(8192, kv_shape(), @core.Q4KV), Ok(603979776.0))
}

///|
test "context and MHA sensitivity scale KV proportionally" {
  assert_eq(
    @core.estimate_kv(16384, kv_shape(), @core.FP16KV),
    Ok(4294967296.0),
  )
  assert_eq(
    @core.estimate_kv(8192, { ..kv_shape(), kv_heads: 40, }, @core.FP16KV),
    Ok(10737418240.0),
  )
}

///|
test "proxy bucket boundaries stay transparent" {
  let low = @core.proxy_architecture(7.0).unwrap()
  let middle = @core.proxy_architecture(14.0).unwrap()
  let large = @core.proxy_architecture(32.0).unwrap()
  let biggest = @core.proxy_architecture(72.0).unwrap()
  assert_eq((low.layers, low.kv_heads, low.query_heads), (28, 4, 28))
  assert_eq((middle.layers, middle.kv_heads, middle.query_heads), (48, 8, 40))
  assert_eq((large.layers, large.kv_heads, large.query_heads), (64, 8, 40))
  assert_eq(
    (biggest.layers, biggest.kv_heads, biggest.query_heads),
    (80, 8, 64),
  )
  assert_eq(@core.proxy_architecture(7.001).unwrap().layers, 48)
  assert_eq(@core.proxy_architecture(14.001).unwrap().layers, 64)
  assert_eq(@core.proxy_architecture(32.001).unwrap().layers, 80)
  assert_eq(@core.proxy_architecture(27.0).unwrap().layers, 64)
}

///|
test "proxy rejects invalid and unsupported extrapolation" {
  for value in [0.0, -1.0, 72.001, 0.0 / 0.0, 1.0 / 0.0] {
    assert_true(@core.proxy_architecture(value) is Err(_))
  }
}

///|
test "architecture rejects invalid shape bounds" {
  for
    architecture in [
      { ..kv_shape(), layers: 0, },
      { ..kv_shape(), layers: 513, },
      { ..kv_shape(), kv_heads: 0, },
      { ..kv_shape(), kv_heads: 513, },
      { ..kv_shape(), head_dim: 0, },
      { ..kv_shape(), head_dim: 1025, },
      { ..kv_shape(), query_heads: 7, },
      { ..kv_shape(), query_heads: 513, },
    ] {
    assert_true(@core.validate_architecture(architecture) is Err(_))
    assert_true(@core.estimate_kv(8192, architecture, @core.FP16KV) is Err(_))
  }
}

///|
test "context and quantized row boundaries are enforced" {
  assert_true(@core.estimate_kv(0, kv_shape(), @core.FP16KV) is Err(_))
  assert_true(@core.estimate_kv(1048577, kv_shape(), @core.FP16KV) is Err(_))
  let shape = { ..kv_shape(), head_dim: 33, }
  assert_true(@core.estimate_kv(8192, shape, @core.FP16KV) is Ok(_))
  assert_true(@core.estimate_kv(8192, shape, @core.Q8KV) is Err(_))
  assert_true(@core.estimate_kv(8192, shape, @core.Q4KV) is Err(_))
}

///|
test "valid large shape avoids integer overflow" {
  let shape : @core.Architecture = {
    layers: 512,
    kv_heads: 512,
    head_dim: 1024,
    query_heads: 512,
    source: "test maximum",
  }
  assert_eq(
    @core.estimate_kv(1048576, shape, @core.FP16KV),
    Ok(1125899906842624.0),
  )
}

///|
test "KV recipe parser distinguishes weight recipe names" {
  assert_eq(@core.parse_kv_precision("FP16"), Ok(@core.FP16KV))
  assert_eq(@core.parse_kv_precision("q8_0"), Ok(@core.Q8KV))
  assert_eq(@core.parse_kv_precision("Q4"), Ok(@core.Q4KV))
  assert_true(@core.parse_kv_precision("Q4_K_M") is Err(_))
  assert_eq(@core.kv_precision_name(@core.Q4KV), "Q4_0")
}

///|
test "GQA query heads must form complete KV groups" {
  let shape : @core.Architecture = {
    layers: 4,
    kv_heads: 3,
    query_heads: 5,
    head_dim: 128,
    source: "invalid GQA test fixture",
  }
  assert_true(@core.validate_architecture(shape) is Err(_))
  assert_true(@core.estimate_kv(4096, shape, @core.FP16KV) is Err(_))
  assert_true(
    @core.validate_architecture({ ..shape, query_heads: 6, }) is Ok(_),
  )
}
