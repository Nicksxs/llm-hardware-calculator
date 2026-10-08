///|
fn fit_fixture_policy() -> @core.BudgetPolicy {
  {
    gpu_runtime_gib: 0.0,
    gpu_margin_gib: 0.0,
    host_runtime_gib: 0.0,
    os_reserve_gib: 0.0,
    loading_workspace_gib: 0.0,
    gpu_staging_gib: 0.0,
    id: "synthetic zero-overhead fit fixture",
  }
}

///|
fn fit_fixture(vram_gib : Double, ram_gib : Double) -> @core.FitResult {
  @core.search_fit(
    @core.gib_to_bytes(8.0),
    @core.gib_to_bytes(4.0),
    4,
    vram_gib,
    ram_gib,
    fit_fixture_policy(),
  ).unwrap()
}

///|
test "fit classifies full GPU offload CPU and insufficient separately" {
  let full = fit_fixture(12.0, 8.0)
  assert_eq(full.status, @core.FullGPU)
  assert_eq(full.ledger.unwrap().gpu_layers, 4)
  let offload = fit_fixture(9.0, 9.0)
  assert_eq(offload.status, @core.Offload)
  assert_eq(offload.ledger.unwrap().gpu_layers, 3)
  let cpu = fit_fixture(2.0, 12.0)
  assert_eq(cpu.status, @core.CPUOnly)
  assert_eq(cpu.ledger.unwrap().gpu_layers, 0)
  let insufficient = fit_fixture(9.0, 8.0)
  assert_eq(insufficient.status, @core.Insufficient)
  assert_eq(insufficient.ledger, None)
  assert_true(insufficient.reason.contains("full-host-weights-loading-policy"))
  assert_true(insufficient.reason.contains("does not establish"))
}

///|
test "fit accepts exact budgets and rejects a one byte shortfall" {
  let one_byte_gib = @core.bytes_to_gib(1.0)
  assert_eq(fit_fixture(12.0, 8.0).status, @core.FullGPU)
  assert_eq(fit_fixture(12.0 - one_byte_gib, 9.0).status, @core.Offload)
  assert_eq(fit_fixture(12.0, 8.0 - one_byte_gib).status, @core.Insufficient)
  assert_eq(fit_fixture(0.0, 12.0).status, @core.CPUOnly)
  assert_eq(fit_fixture(0.0, 12.0 - one_byte_gib).status, @core.Insufficient)
}

///|
test "fit checks host loading even if both steady budgets fit" {
  let result = fit_fixture(12.0, 0.0)
  let full = @core.build_ledger(
    @core.gib_to_bytes(8.0),
    @core.gib_to_bytes(4.0),
    4,
    4,
    fit_fixture_policy(),
  ).unwrap()
  assert_eq(full.host_steady_bytes, 0.0)
  assert_eq(full.gpu_steady_bytes, @core.gib_to_bytes(12.0))
  assert_eq(result.status, @core.Insufficient)
}

///|
test "GPU staging can require fewer GPU layers than steady state suggests" {
  let policy = { ..fit_fixture_policy(), gpu_staging_gib: 1.0, }
  let result = @core.search_fit(
    @core.gib_to_bytes(8.0),
    @core.gib_to_bytes(4.0),
    4,
    12.0,
    9.0,
    policy,
  ).unwrap()
  // Four layers have a 12 GiB steady budget but a 13 GiB loading peak.
  assert_eq(result.status, @core.Offload)
  let ledger = result.ledger.unwrap()
  assert_eq(ledger.gpu_layers, 3)
  assert_eq(ledger.gpu_loading_peak_bytes, @core.gib_to_bytes(10.0))
}

///|
test "zero VRAM uses CPU only without charging GPU allowances" {
  let policy = @core.default_policy()
  let result = @core.search_fit(
    @core.gib_to_bytes(8.0),
    @core.gib_to_bytes(4.0),
    4,
    0.0,
    18.0,
    policy,
  ).unwrap()
  assert_eq(result.status, @core.CPUOnly)
  assert_eq(result.ledger.unwrap().gpu_loading_peak_bytes, 0.0)
}

///|
test "increasing VRAM preserves feasibility and does not reduce GPU layers" {
  let mut previous_layers = 0
  for vram = 0; vram <= 12; vram = vram + 1 {
    let result = fit_fixture(vram.to_double(), 12.0)
    let ledger = result.ledger.unwrap()
    assert_true(ledger.gpu_layers >= previous_layers)
    assert_true(
      ledger.gpu_loading_peak_bytes <= @core.gib_to_bytes(vram.to_double()),
    )
    assert_true(ledger.host_loading_peak_bytes <= @core.gib_to_bytes(12.0))
    previous_layers = ledger.gpu_layers
  }
}

///|
test "more RAM cannot discard a feasible GPU layer placement" {
  let mut previous_layers = 0
  let mut was_feasible = false
  for ram = 0; ram <= 12; ram = ram + 1 {
    let result = fit_fixture(9.0, ram.to_double())
    match result.ledger {
      Some(ledger) => {
        assert_true(ledger.gpu_layers >= previous_layers)
        previous_layers = ledger.gpu_layers
        was_feasible = true
      }
      None => assert_false(was_feasible)
    }
  }
}

///|
test "search validates capacities components and policies before classification" {
  let policy = fit_fixture_policy()
  for invalid in [-1.0, 1000001.0, 0.0 / 0.0, 1.0 / 0.0, -1.0 / 0.0] {
    assert_true(@core.search_fit(8.0, 4.0, 4, invalid, 12.0, policy) is Err(_))
    assert_true(@core.search_fit(8.0, 4.0, 4, 12.0, invalid, policy) is Err(_))
  }
  assert_true(@core.search_fit(8.0, 4.0, 0, 12.0, 12.0, policy) is Err(_))
  assert_true(@core.search_fit(8.0, 4.0, -1, 12.0, 12.0, policy) is Err(_))
  assert_true(@core.search_fit(8.0, 4.0, 513, 12.0, 12.0, policy) is Err(_))
  assert_true(@core.search_fit(0.0, 4.0, 4, 12.0, 12.0, policy) is Err(_))
  assert_true(@core.search_fit(1.0 / 0.0, 4.0, 4, 12.0, 12.0, policy) is Err(_))
  assert_true(@core.search_fit(8.0, -1.0, 4, 12.0, 12.0, policy) is Err(_))
  assert_true(@core.search_fit(8.0, 0.0 / 0.0, 4, 12.0, 12.0, policy) is Err(_))
  assert_true(
    @core.search_fit(8.0, 4.0, 4, 12.0, 12.0, {
      ..policy,
      gpu_runtime_gib: -1.0,
    })
    is Err(_),
  )
}

///|
test "status names are stable public report values" {
  assert_eq(@core.fit_status_name(@core.FullGPU), "FULL_GPU")
  assert_eq(@core.fit_status_name(@core.Offload), "GPU+CPU OFFLOAD")
  assert_eq(@core.fit_status_name(@core.CPUOnly), "CPU ONLY")
  assert_eq(@core.fit_status_name(@core.Insufficient), "INSUFFICIENT")
}
