///|
/// Native process exit is supplied by libc; errors are printed on stderr.
extern "C" fn process_exit(code : Int) -> Unit = "exit"

///|
fn main {
  let args = @env.args()
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
