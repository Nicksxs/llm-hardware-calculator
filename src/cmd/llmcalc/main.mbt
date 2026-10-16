///|
/// Native process exit is supplied by libc; errors are printed on stderr.
extern "C" fn process_exit(code : Int) -> Unit = "exit"

///|
fn artifact_main(args : Array[String]) -> Unit {
  let command = match @artifact_cli.parse_args(args) {
    Ok(command) => command
    Err(message) => {
      eprintln("llmcalc: " + message)
      process_exit(2)
      return
    }
  }
  match command {
    Help => println(@artifact_cli.help_text())
    Inspect(options) => {
      let scan = match
        @nativeio.inspect_file(options.path, @gguf.default_limits()) {
        Ok(scan) => scan
        Err(message) => {
          eprintln("llmcalc: " + message)
          process_exit(1)
          return
        }
      }
      println(
        if options.json {
          @artifact_cli.render_inspect_json(scan)
        } else {
          @artifact_cli.render_inspect_text(scan)
        },
      )
    }
    Plan(options) => {
      let scan = match
        @nativeio.inspect_file(options.path, @gguf.default_limits()) {
        Ok(scan) => scan
        Err(message) => {
          eprintln("llmcalc: " + message)
          process_exit(1)
          return
        }
      }
      let profile = match @model.extract_profile(scan.model) {
        Ok(profile) => profile
        Err(message) => {
          eprintln("llmcalc: " + message)
          process_exit(1)
          return
        }
      }
      let result = match options.n_gpu_layers {
        Some(requested) =>
          @model.evaluate_plan(
            profile,
            options.context,
            options.kv_precision,
            requested,
            options.vram_gib,
            options.ram_gib,
            options.policy,
          )
        None =>
          @model.search_plan(
            profile,
            options.context,
            options.kv_precision,
            options.vram_gib,
            options.ram_gib,
            options.policy,
          )
      }
      let search = match result {
        Ok(search) => search
        Err(message) => {
          eprintln("llmcalc: " + message)
          process_exit(1)
          return
        }
      }
      println(
        if options.json {
          @artifact_cli.render_plan_json(scan, profile, search, options)
        } else {
          @artifact_cli.render_plan_text(scan, profile, search, options)
        },
      )
    }
  }
}

///|
fn main {
  let args = @env.args()
  if args.length() > 1 && (args[1] == "inspect" || args[1] == "plan") {
    artifact_main(args[1:].to_owned())
    return
  }
  if args.length() == 2 && args[1] == "--help" {
    println(@artifact_cli.help_text() + "\n\n" + @cli.help_text())
    return
  }
  let options = match @cli.parse_args(args[1:].to_owned()) {
    Ok(options) => options
    Err(error) => {
      eprintln("llmcalc: " + error + "\nUse --help for usage.")
      process_exit(2)
      return
    }
  }
  if options.help {
    println(@cli.help_text())
    return
  }
  let report = match
    @core.calculate(
      options.input,
      options.architecture,
      options.kv_precision,
      options.policy,
    ) {
    Ok(report) => report
    Err(error) => {
      eprintln("llmcalc: " + error)
      process_exit(1)
      return
    }
  }
  println(
    if options.json {
      @cli.render_json(report)
    } else {
      @cli.render_text(report)
    },
  )
}
