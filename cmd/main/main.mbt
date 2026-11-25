// CRF 命令行：无参数运行一个 BIO 约束的命名实体识别演示；
// 或用 --model / --emit 传入 JSON 权重与发射分做解码。
//
// 用法：
//   moon run cmd/main
//   moon run cmd/main -- --model "<crf-json>" --emit "[[1.0,0.0],[0.0,2.0]]"
//   moon run cmd/main -- --help

///|
suberror CliError {
  Msg(String)
}

///|
fn CliError::message(self : Self) -> String {
  match self {
    Msg(s) => s
  }
}

///|
fn main {
  run(cli_args()) catch {
    e => {
      eprintln("error: \{e.message()}")
      panic()
    }
  }
}

///|
#cfg(target="js")
fn cli_args() -> Array[String] {
  @env.args()[1:].to_owned()
}

///|
#cfg(not(target="js"))
fn cli_args() -> Array[String] {
  @env.args()
}

///|
/// 把库的 CrfError 转成 CLI 错误，避免在包外构造只读的错误构造器。
fn[A] lift(f : () -> A raise @lib.CrfError) -> A raise CliError {
  f() catch {
    e => raise Msg(e.message())
  }
}

///|
fn run(args : Array[String]) -> Unit raise CliError {
  let mut model : String? = None
  let mut emit : String? = None
  let mut help = false
  let mut i = 1
  while i < args.length() {
    match args[i] {
      "--help" => {
        help = true
        i += 1
      }
      "--model" => {
        i += 1
        if i < args.length() {
          model = Some(args[i])
          i += 1
        } else {
          raise Msg("--model requires a JSON value")
        }
      }
      "--emit" => {
        i += 1
        if i < args.length() {
          emit = Some(args[i])
          i += 1
        } else {
          raise Msg("--emit requires a JSON value")
        }
      }
      other => raise Msg("unknown option: " + other)
    }
  }
  if help {
    println(
      "usage: moon run cmd/main [-- --model <crf-json> --emit <emissions-json>]",
    )
    println("       (no arguments runs a built-in BIO NER demo)")
    return
  }
  match (model, emit) {
    (Some(m), Some(e)) => {
      let crf = lift(() => @lib.Crf::from_json(m))
      let emissions = parse_emissions(e)
      report(crf, emissions, None, None)
    }
    (None, None) => demo()
    _ => raise Msg("--model and --emit must be given together")
  }
}

///|
fn parse_emissions(s : String) -> Array[Array[Double]] raise CliError {
  let v = @json.parse(s[:]) catch {
    _ => raise Msg("emissions is not valid JSON")
  }
  @json.from_json(v) catch {
    _ => raise Msg("emissions must be a 2-D array of numbers")
  }
}

///|
fn demo() -> Unit raise CliError {
  let tags = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
  let crf = lift(() => @lib.Crf::uniform(tags.length()))
  let ca = lift(() => @lib.bio_mask(tags))
  let emit = [
    [0.2, 2.0, 0.1, 0.0, 0.0],
    [0.1, 0.0, 1.8, 0.0, 0.0],
    [2.5, 0.0, 0.0, 0.1, 0.0],
    [0.0, 0.0, 0.0, 2.2, 0.1],
  ]
  println("== BIO-constrained NER demo ==")
  report(crf, emit, Some(ca), Some(tags))
}

///|
fn report(
  crf : @lib.Crf,
  emit : Array[Array[Double]],
  ca : @lib.Constraints?,
  tags : Array[String]?,
) -> Unit raise CliError {
  let path = lift(() => crf.decode(emit, None, ca))
  let vp = lift(() => crf.viterbi(emit, None, ca))
  let lz = lift(() => crf.log_partition(emit, None, ca))
  let mg = lift(() => crf.marginals(emit, None, ca))
  let nb = lift(() => crf.nbest(emit, None, ca, 3))
  println("decoded path : \{fmt_ints(path)}")
  match tags {
    Some(ts) => println("tag names    : \{fmt_names(ts, path)}")
    None => ()
  }
  println("path score   : \{vp.score}")
  println("log Z        : \{lz}")
  println("log L        : \{vp.score - lz}")
  println("marginals (per position):")
  for t in 0..<mg.length() {
    println("  t=\{t}: \{fmt_doubles(mg[t])}")
  }
  println("top-3 paths:")
  for p in nb {
    println("  [\{fmt_ints(p.tags)}]  score=\{p.score}")
  }
}

///|
fn fmt_ints(a : Array[Int32]) -> String {
  let mut s = ""
  for i in 0..<a.length() {
    if i > 0 {
      s = s + " "
    }
    s = s + "\{a[i]}"
  }
  s
}

///|
fn fmt_doubles(a : Array[Double]) -> String {
  let mut s = ""
  for i in 0..<a.length() {
    if i > 0 {
      s = s + " "
    }
    s = s + "\{a[i]}"
  }
  s
}

///|
fn fmt_names(tags : Array[String], path : Array[Int32]) -> String {
  let mut s = ""
  for i in 0..<path.length() {
    if i > 0 {
      s = s + " "
    }
    s = s + tags[path[i]]
  }
  s
}
