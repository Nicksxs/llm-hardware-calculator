///|
fn sample() -> @core.BasicInput {
  {
    params_b: 27.0,
    quant: @core.Q4_K_M,
    vram_gib: 16.0,
    ram_gib: 64.0,
    context: 8192,
  }
}

///|
test "input accepts zero GPU and RAM for later capacity classification" {
  let input = { ..sample(), vram_gib: 0.0, ram_gib: 0.0, }
  assert_eq(@core.validate_input(input), Ok(input))
}

///|
test "input rejects invalid parameters and context" {
  assert_true(@core.validate_input({ ..sample(), params_b: 0.0, }) is Err(_))
  assert_true(@core.validate_input({ ..sample(), params_b: -1.0, }) is Err(_))
  assert_true(@core.validate_input({ ..sample(), params_b: 1001.0, }) is Err(_))
  assert_true(@core.validate_input({ ..sample(), context: 0, }) is Err(_))
  assert_true(@core.validate_input({ ..sample(), context: 1048577, }) is Err(_))
}

///|
test "input rejects negative capacity and non-finite floating values" {
  assert_true(@core.validate_input({ ..sample(), vram_gib: -1.0, }) is Err(_))
  assert_true(@core.validate_input({ ..sample(), ram_gib: -1.0, }) is Err(_))
  assert_true(
    @core.validate_input({ ..sample(), params_b: 0.0 / 0.0, }) is Err(_),
  )
  assert_true(
    @core.validate_input({ ..sample(), vram_gib: 1.0 / 0.0, }) is Err(_),
  )
  assert_true(
    @core.validate_input({ ..sample(), ram_gib: 1.0 / 0.0, }) is Err(_),
  )
}

///|
test "GiB conversion distinguishes binary capacity from decimal bytes" {
  assert_eq(@core.gib_to_bytes(1.0), 1073741824.0)
  assert_eq(@core.bytes_to_gib(1073741824.0), 1.0)
  assert_true(@core.bytes_to_gib(1000000000.0) < 1.0)
}
