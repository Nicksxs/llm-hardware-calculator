///|
fn zero_policy() -> @core.BudgetPolicy {
  {
    gpu_runtime_gib: 0.0,
    gpu_margin_gib: 0.0,
    host_runtime_gib: 0.0,
    os_reserve_gib: 0.0,
    loading_workspace_gib: 0.0,
    gpu_staging_gib: 0.0,
    id: "synthetic zero-overhead fixture",
  }
}

///|
test "ledger conserves weights and KV at every integer split" {
  let weights = @core.gib_to_bytes(8.0)
  let kv = @core.gib_to_bytes(4.0)
  for gpu_layers = 0; gpu_layers <= 4; gpu_layers = gpu_layers + 1 {
    let ledger = @core.build_ledger(weights, kv, 4, gpu_layers, zero_policy()).unwrap()
    assert_eq(ledger.gpu_weights_bytes + ledger.host_weights_bytes, weights)
    assert_eq(ledger.gpu_kv_bytes + ledger.host_kv_bytes, kv)
    assert_eq(ledger.gpu_layers + ledger.cpu_layers, 4)
    assert_eq(ledger.host_loading_peak_bytes, weights + ledger.host_kv_bytes)
  }
}

///|
test "CPU only has no GPU runtime margin or staging" {
  let ledger = @core.build_ledger(
    @core.gib_to_bytes(8.0),
    @core.gib_to_bytes(4.0),
    4,
    0,
    @core.default_policy(),
  ).unwrap()
  assert_eq(ledger.gpu_steady_bytes, 0.0)
  assert_eq(ledger.gpu_loading_peak_bytes, 0.0)
  assert_eq(ledger.gpu_runtime_bytes, 0.0)
  assert_eq(ledger.gpu_margin_bytes, 0.0)
  assert_eq(ledger.gpu_staging_bytes, 0.0)
  assert_eq(@core.bytes_to_gib(ledger.host_steady_bytes), 17.0)
  assert_eq(@core.bytes_to_gib(ledger.host_loading_peak_bytes), 18.0)
}

///|
test "full GPU still retains a conservative host loading requirement" {
  let ledger = @core.build_ledger(
    @core.gib_to_bytes(8.0),
    @core.gib_to_bytes(4.0),
    4,
    4,
    @core.default_policy(),
  ).unwrap()
  assert_eq(@core.bytes_to_gib(ledger.gpu_steady_bytes), 13.5)
  assert_eq(@core.bytes_to_gib(ledger.gpu_loading_peak_bytes), 13.75)
  assert_eq(@core.bytes_to_gib(ledger.host_steady_bytes), 5.0)
  assert_eq(@core.bytes_to_gib(ledger.host_loading_peak_bytes), 14.0)
}

///|
test "host loading accounting does not double count CPU weights" {
  let ledger = @core.build_ledger(
    @core.gib_to_bytes(8.0),
    @core.gib_to_bytes(4.0),
    4,
    3,
    @core.default_policy(),
  ).unwrap()
  assert_eq(@core.bytes_to_gib(ledger.host_steady_bytes), 8.0)
  assert_eq(@core.bytes_to_gib(ledger.host_loading_peak_bytes), 15.0)
  assert_eq(@core.bytes_to_gib(ledger.gpu_loading_peak_bytes), 10.75)
}

///|
test "ledger rejects invalid components splits and allowances" {
  let policy = zero_policy()
  assert_true(@core.build_ledger(0.0, 1.0, 4, 0, policy) is Err(_))
  assert_true(@core.build_ledger(1.0, -1.0, 4, 0, policy) is Err(_))
  assert_true(@core.build_ledger(1.0, 1.0, 0, 0, policy) is Err(_))
  assert_true(@core.build_ledger(1.0, 1.0, 4, 5, policy) is Err(_))
  assert_true(@core.build_ledger(1.0 / 0.0, 1.0, 4, 0, policy) is Err(_))
  assert_true(
    @core.validate_policy({ ..policy, gpu_margin_gib: -1.0, }) is Err(_),
  )
  assert_true(
    @core.validate_policy({ ..policy, os_reserve_gib: 0.0 / 0.0, }) is Err(_),
  )
}
