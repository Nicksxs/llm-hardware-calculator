///|
/// Arguments exclude the executable name. Memory capacities are normalized to
/// GiB and context K means 1024 tokens. Help carries an unused valid input.
pub(all) struct CliOptions {
  input : @core.BasicInput
  architecture : @core.Architecture?
  kv_precision : @core.KvPrecision
  policy : @core.BudgetPolicy
  json : Bool
  help : Bool
} derive(Eq, Debug)

///|
pub extend CliOptions with Eq::{equal, not_equal}

///|
pub extend CliOptions with @debug.Debug::{to_repr}

///|
fn number(value : StringView, option : String) -> Result[Double, String] {
  Ok(@string.parse_double(value)) catch {
    _ => Err(option + " requires a number")
  }
}

///|
fn integer(value : StringView, option : String) -> Result[Int, String] {
  Ok(@string.parse_int(value, base=10)) catch {
    _ => Err(option + " requires a decimal integer")
  }
}

///|
fn params(value : String) -> Result[Double, String] {
  let upper = value.to_upper()
  let numeric = match upper.strip_suffix("B") {
    Some(value) => value
    None => upper[:]
  }
  number(numeric, "--params")
}

///|
fn capacity(value : String, option : String) -> Result[Double, String] {
  let upper = value.to_upper()
  let (numeric, factor) = match upper.strip_suffix("GIB") {
    Some(value) => (value, 1.0)
    None =>
      match upper.strip_suffix("GB") {
        Some(value) => (value, 1000000000.0 / 1073741824.0)
        None => return Err(option + " requires GiB or GB (GB is decimal)")
      }
  }
  match number(numeric, option) {
    Ok(value) => Ok(value * factor)
    Err(error) => Err(error)
  }
}

///|
fn context(value : String) -> Result[Int, String] {
  let upper = value.to_upper()
  let (numeric, factor) = match upper.strip_suffix("K") {
    Some(value) => (value, 1024)
    None => (upper[:], 1)
  }
  match integer(numeric, "--context") {
    Ok(value) =>
      if value < 1 || value > 1048576 / factor {
        Err("--context must be in [1, 1048576] tokens; K means 1024")
      } else {
        Ok(value * factor)
      }
    Err(error) => Err(error)
  }
}

///|
fn policy_capacity(value : String, option : String) -> Result[Double, String] {
  let upper = value.to_upper()
  if upper.has_suffix("GIB") || upper.has_suffix("GB") {
    capacity(value, option)
  } else {
    number(value[:], option)
  }
}

///|
fn parse_policy(
  values : Map[String, String],
) -> Result[@core.BudgetPolicy, String] {
  let mut policy = @core.default_policy()
  let overrides : Array[String] = []
  for
    option in [
      "--gpu-runtime", "--gpu-margin", "--host-runtime", "--os-reserve", "--loading-workspace",
      "--gpu-staging",
    ] {
    match values.get(option) {
      None => ()
      Some(value) => {
        let amount = match policy_capacity(value, option) {
          Ok(amount) => amount
          Err(error) => return Err(error)
        }
        policy = match option {
          "--gpu-runtime" => { ..policy, gpu_runtime_gib: amount, }
          "--gpu-margin" => { ..policy, gpu_margin_gib: amount, }
          "--host-runtime" => { ..policy, host_runtime_gib: amount, }
          "--os-reserve" => { ..policy, os_reserve_gib: amount, }
          "--loading-workspace" => { ..policy, loading_workspace_gib: amount, }
          "--gpu-staging" => { ..policy, gpu_staging_gib: amount, }
          _ => policy
        }
        overrides.push(option)
      }
    }
  }
  if !overrides.is_empty() {
    policy = {
      ..policy,
      id: "custom CLI overrides (" + overrides.join(", ") + ") of " + policy.id,
    }
  }
  @core.validate_policy(policy)
}

///|
fn required(
  values : Map[String, String],
  option : String,
  help : Bool,
  help_default : String,
) -> Result[String, String] {
  match values.get(option) {
    Some(value) => Ok(value)
    None =>
      if help {
        Ok(help_default)
      } else {
        Err("missing required option: " + option)
      }
  }
}

///|
/// Parse both `--flag value` and `--flag=value`. Unknown and duplicate flags,
/// unexpected positional arguments, malformed values, and partial architecture
/// overrides return errors without running the calculator.
pub fn parse_args(args : Array[String]) -> Result[CliOptions, String] {
  let values : Map[String, String] = Map([])
  let seen : Map[String, Bool] = Map([])
  let mut help = false
  let mut json = false
  let mut index = 0
  while index < args.length() {
    let token = args[index]
    let (option, inline) = match token.split_once("=") {
      Some((name, value)) => (name.to_owned(), Some(value.to_owned()))
      None => (token, None)
    }
    let needs_value = match option {
      "--help" | "--json" => false
      "--params"
      | "--quant"
      | "--vram"
      | "--ram"
      | "--context"
      | "--layers"
      | "--kv-heads"
      | "--head-dim"
      | "--query-heads"
      | "--kv"
      | "--gpu-runtime"
      | "--gpu-margin"
      | "--host-runtime"
      | "--os-reserve"
      | "--loading-workspace"
      | "--gpu-staging" => true
      _ => return Err("unknown option or unexpected argument: " + option)
    }
    if seen.get(option) is Some(_) {
      return Err("duplicate option: " + option)
    }
    seen[option] = true
    if needs_value {
      let value = match inline {
        Some(value) => value
        None => {
          if index + 1 >= args.length() || args[index + 1].has_prefix("--") {
            return Err("missing value for " + option)
          }
          index += 1
          args[index]
        }
      }
      if value.is_empty() {
        return Err("missing value for " + option)
      }
      values[option] = value
    } else {
      if inline is Some(_) {
        return Err(option + " does not accept a value")
      }
      match option {
        "--help" => help = true
        "--json" => json = true
        _ => ()
      }
    }
    index += 1
  }
  let params_text = match required(values, "--params", help, "27B") {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let params_b = match params(params_text) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let quant_text = match required(values, "--quant", help, "Q4_K_M") {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let quant = match @core.parse_weight_quant(quant_text) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let vram_text = match required(values, "--vram", help, "16GiB") {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let vram_gib = match capacity(vram_text, "--vram") {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let ram_text = match required(values, "--ram", help, "64GiB") {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let ram_gib = match capacity(ram_text, "--ram") {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let context_text = match required(values, "--context", help, "8K") {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let context = match context(context_text) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let input : @core.BasicInput = {
    params_b,
    quant,
    vram_gib,
    ram_gib,
    context,
  }
  match @core.validate_input(input) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  let kv_text = values.get("--kv").unwrap_or("FP16")
  let kv_precision = match @core.parse_kv_precision(kv_text) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let architecture = match
    (values.get("--layers"), values.get("--kv-heads"), values.get("--head-dim")) {
    (None, None, None) => {
      if values.get("--query-heads") is Some(_) {
        return Err("--query-heads requires --layers, --kv-heads and --head-dim")
      }
      None
    }
    (Some(layers), Some(kv_heads), Some(head_dim)) => {
      let layers = match integer(layers[:], "--layers") {
        Ok(value) => value
        Err(error) => return Err(error)
      }
      let kv_heads = match integer(kv_heads[:], "--kv-heads") {
        Ok(value) => value
        Err(error) => return Err(error)
      }
      let head_dim = match integer(head_dim[:], "--head-dim") {
        Ok(value) => value
        Err(error) => return Err(error)
      }
      let query_heads = match values.get("--query-heads") {
        Some(value) =>
          match integer(value[:], "--query-heads") {
            Ok(value) => value
            Err(error) => return Err(error)
          }
        None => kv_heads
      }
      let architecture : @core.Architecture = {
        layers,
        query_heads,
        kv_heads,
        head_dim,
        source: "explicit CLI architecture override",
      }
      match @core.validate_architecture(architecture) {
        Ok(_) => ()
        Err(error) => return Err(error)
      }
      Some(architecture)
    }
    _ =>
      return Err(
        "--layers, --kv-heads and --head-dim must be supplied together",
      )
  }
  let policy = match parse_policy(values) {
    Ok(policy) => policy
    Err(error) => return Err(error)
  }
  Ok({ input, architecture, kv_precision, policy, json, help, })
}
