///|
fn plan_args() -> Array[String] {
  ["plan", "model.gguf", "--vram", "16GiB", "--ram", "64GiB", "--context", "8K"]
}

///|
test "inspect path and JSON flag are distinct from Basic inputs" {
  let command = @artifact_cli.parse_args([
    "inspect", "models/a model.gguf", "--json",
  ])
  guard command is Ok(@artifact_cli.Inspect(options)) else {
    fail("expected inspect options")
  }
  assert_eq(options.path, "models/a model.gguf")
  assert_eq(options.json, true)
  assert_true(
    @artifact_cli.parse_args(["inspect", "file.gguf", "--vram=16GiB"]) is Err(_),
  )
}

///|
test "plan parses memory inputs and defaults without exposing proxy fields" {
  guard @artifact_cli.parse_args(plan_args()) is Ok(@artifact_cli.Plan(options)) else {
    fail("expected plan options")
  }
  assert_eq(options.path, "model.gguf")
  assert_eq(options.vram_gib, 16.0)
  assert_eq(options.ram_gib, 64.0)
  assert_eq(options.context, 8192)
  assert_eq(options.kv_precision, @core.FP16KV)
  assert_eq(options.policy, @core.default_policy())
  assert_eq(options.json, false)
}

///|
test "plan shares decimal GB conversion KV recipes and custom policy" {
  let command = @artifact_cli.parse_args([
    "plan", "model.gguf", "--vram=16GB", "--ram=64GB", "--context=16K", "--kv=Q8_0",
    "--os-reserve=0", "--gpu-staging=0.5", "--json",
  ])
  guard command is Ok(@artifact_cli.Plan(options)) else {
    fail("expected plan options")
  }
  assert_eq(options.vram_gib, 16000000000.0 / 1073741824.0)
  assert_eq(options.ram_gib, 64000000000.0 / 1073741824.0)
  assert_eq(options.context, 16384)
  assert_eq(options.kv_precision, @core.Q8KV)
  assert_eq(options.policy.os_reserve_gib, 0.0)
  assert_eq(options.policy.gpu_staging_gib, 0.5)
  assert_true(options.policy.id.contains("custom CLI overrides"))
  assert_eq(options.json, true)
}

///|
test "artifact help requires no file and still rejects invalid arguments" {
  assert_eq(
    @artifact_cli.parse_args(["inspect", "--help"]),
    Ok(@artifact_cli.Help),
  )
  assert_eq(
    @artifact_cli.parse_args(["plan", "--help"]),
    Ok(@artifact_cli.Help),
  )
  assert_true(
    @artifact_cli.parse_args(["plan", "--help", "--vram=NaNGiB"]) is Err(_),
  )
  assert_true(
    @artifact_cli.parse_args(["inspect", "--help", "--unknown"]) is Err(_),
  )
}

///|
test "artifact parser rejects missing files extra positionals and invalid commands" {
  for
    args in [
      [],
      ["other"],
      ["inspect"],
      ["inspect", ""],
      ["inspect", "a.gguf", "b.gguf"],
      ["inspect", "a\u{0}.gguf"],
      ["plan", "--vram=16GiB", "--ram=64GiB", "--context=8K"],
    ] {
    assert_true(@artifact_cli.parse_args(args) is Err(_))
  }
}

///|
test "artifact parser rejects override of GGUF parameters and architecture" {
  for
    option in [
      "--params=27B", "--quant=Q4_K_M", "--layers=64", "--head-dim=128",
    ] {
    let args = plan_args()
    args.push(option)
    assert_true(@artifact_cli.parse_args(args) is Err(_))
  }
}

///|
test "artifact parser rejects unknown duplicate missing and malformed flags" {
  for
    extra in [
      ["--unknown"],
      ["--json", "--json"],
      ["--vram=16GiB"],
      ["--kv"],
      ["--kv", "--json"],
      ["--kv="],
      ["--json=true"],
      ["--kv=BF16"],
      ["--os-reserve=-1"],
      ["--gpu-margin=NaN"],
    ] {
    let args = plan_args()
    args.append(extra[:])
    assert_true(@artifact_cli.parse_args(args) is Err(_))
  }
  assert_true(
    @artifact_cli.parse_args(["inspect", "file.gguf", "--json", "--json"])
    is Err(_),
  )
}

///|
test "plan requires all three capacity and context flags" {
  for
    args in [
      ["plan", "file.gguf"],
      ["plan", "file.gguf", "--ram=64GiB", "--context=8K"],
      ["plan", "file.gguf", "--vram=16GiB", "--context=8K"],
      ["plan", "file.gguf", "--vram=16GiB", "--ram=64GiB"],
    ] {
    assert_true(@artifact_cli.parse_args(args) is Err(_))
  }
}
