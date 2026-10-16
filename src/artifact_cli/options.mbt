///|
pub(all) struct InspectOptions {
  path : String
  json : Bool
} derive(Eq, Debug)

///|
pub extend InspectOptions with Eq::{equal, not_equal}

///|
pub extend InspectOptions with @debug.Debug::{to_repr}

///|
/// Artifact weights and architecture come from GGUF rather than CLI proxies.
pub(all) struct PlanOptions {
  path : String
  vram_gib : Double
  ram_gib : Double
  context : Int
  kv_precision : @core.KvPrecision
  policy : @core.BudgetPolicy
  json : Bool
} derive(Eq, Debug)

///|
pub extend PlanOptions with Eq::{equal, not_equal}

///|
pub extend PlanOptions with @debug.Debug::{to_repr}

///|
pub(all) enum Command {
  Inspect(InspectOptions)
  Plan(PlanOptions)
  Help
} derive(Eq, Debug)

///|
pub extend Command with Eq::{equal, not_equal}

///|
pub extend Command with @debug.Debug::{to_repr}

///|
/// Args include the subcommand and exclude the executable. Numeric memory/KV
/// validation is shared with the Basic CLI; its parameter/quantization
/// placeholders are discarded and never used for artifact computation.
pub fn parse_args(args : Array[String]) -> Result[Command, String] {
  if args.is_empty() {
    return Err("expected inspect or plan subcommand")
  }
  let is_plan = match args[0] {
    "inspect" => false
    "plan" => true
    _ => return Err("unknown artifact subcommand: " + args[0])
  }
  let forwarded = ["--params=1", "--quant=FP16"]
  let seen : Map[String, Bool] = Map([])
  let mut path : String? = None
  let mut json = false
  let mut help = false
  let mut index = 1
  while index < args.length() {
    let token = args[index]
    if !token.has_prefix("--") {
      if path is Some(_) {
        return Err("unexpected positional argument: " + token)
      }
      if token.is_empty() || token.contains_char('\u{0}') {
        return Err("GGUF path must be nonempty and contain no NUL")
      }
      path = Some(token)
      index += 1
      continue
    }
    let (option, inline) = match token.split_once("=") {
      Some((name, value)) => (name.to_owned(), Some(value.to_owned()))
      None => (token, None)
    }
    let needs_value = match option {
      "--json" | "--help" => false
      "--vram"
      | "--ram"
      | "--context"
      | "--kv"
      | "--gpu-runtime"
      | "--gpu-margin"
      | "--host-runtime"
      | "--os-reserve"
      | "--loading-workspace"
      | "--gpu-staging" => {
        if !is_plan {
          return Err("inspect does not accept option: " + option)
        }
        true
      }
      _ => return Err("unknown artifact option: " + option)
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
      forwarded.push(option + "=" + value)
    } else {
      if inline is Some(_) {
        return Err(option + " does not accept a value")
      }
      forwarded.push(option)
      match option {
        "--help" => help = true
        "--json" => json = true
        _ => ()
      }
    }
    index += 1
  }
  if is_plan {
    let shared = match @cli.parse_args(forwarded) {
      Ok(value) => value
      Err(error) => return Err(error)
    }
    if help {
      return Ok(Help)
    }
    let path = match path {
      Some(path) => path
      None => return Err("plan requires a GGUF file path")
    }
    Ok(
      Plan({
        path,
        vram_gib: shared.input.vram_gib,
        ram_gib: shared.input.ram_gib,
        context: shared.input.context,
        kv_precision: shared.kv_precision,
        policy: shared.policy,
        json,
      }),
    )
  } else {
    if help {
      return Ok(Help)
    }
    let path = match path {
      Some(path) => path
      None => return Err("inspect requires a GGUF file path")
    }
    Ok(Inspect({ path, json, }))
  }
}

///|
pub fn help_text() -> String {
  "GGUF artifact commands\n\nUsage:\n  llmcalc inspect MODEL.gguf [--json]\n  llmcalc plan MODEL.gguf --vram 16GiB --ram 64GiB --context 8K [options]\n\ninspect reads GGUF metadata and tensor descriptors without loading tensor payloads.\nUnknown tensor types produce an incomplete inspection rather than a weight guess.\nplan requires complete supported tensor sizes and a supported model architecture.\n\nPlan options:\n  --vram NUMBERGiB --ram NUMBERGiB  Decimal GB also accepted\n  --context INTEGER[K]            K means 1024 tokens\n  --kv FP16|Q8_0|Q4_0              Default FP16; KV memory scenarios only\n  --gpu-runtime NUMBER            Default 0.5 GiB\n  --gpu-margin NUMBER             Default 1 GiB\n  --host-runtime NUMBER           Default 1 GiB\n  --os-reserve NUMBER             Default 4 GiB; 0 for already-free RAM\n  --loading-workspace NUMBER      Default 1 GiB\n  --gpu-staging NUMBER            Default 0.25 GiB\nPolicy numbers mean GiB unless GiB/GB is specified.\n\nCommon: --json, --help. Flags accept --flag=value; duplicates are errors.\nGGUF byte counts, offsets and element counts use decimal strings in JSON.\nGiB values are presentation conversions. Tensor payload size is not runtime memory."
}
