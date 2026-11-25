// CRF 命令行：无参数运行一个 BIO 约束的命名实体识别演示；
// 或用 --model / --emit 传入 JSON 权重与发射分做解码。
//
// 用法：
//   moon run cmd/main
//   moon run cmd/main -- --model "<crf-json>" --emit "[[1.0,0.0],[0.0,2.0]]"
//   moon run cmd/main -- --model "<crf-json>" --emit "<json>" --bio "O,B-PER,I-PER" --nbest 5
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
  let mut bio : String? = None
  let mut bioes : String? = None
  let mut nbest = 3
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
      "--bio" => {
        i += 1
        if i < args.length() {
          bio = Some(args[i])
          i += 1
        } else {
          raise Msg("--bio requires a comma-separated tag list")
        }
      }
      "--bioes" => {
        i += 1
        if i < args.length() {
          bioes = Some(args[i])
          i += 1
        } else {
          raise Msg("--bioes requires a comma-separated tag list")
        }
      }
      "--nbest" => {
        i += 1
        if i < args.length() {
          nbest = parse_positive(args[i])
          i += 1
        } else {
          raise Msg("--nbest requires a positive integer")
        }
      }
      other => raise Msg("unknown option: " + other)
    }
  }
  if help {
    println(
      "usage: moon run cmd/main [-- --model <crf-json> --emit <emissions-json>]",
    )
    println(
      "       [--bio O,B-PER,I-PER,... | --bioes O,B-PER,I-PER,E-PER,S-PER,...] [--nbest <n>]",
    )
    println("       (no arguments runs a built-in BIO NER demo)")
    return
  }
  if bio is Some(_) && bioes is Some(_) {
    raise Msg("--bio and --bioes are mutually exclusive")
  }
  match (model, emit) {
    (Some(m), Some(e)) => {
      let crf = lift(() => @lib.Crf::from_json(m))
      let emissions = parse_emissions(e)
      let (ca, tags) : (@lib.Constraints?, Array[String]?) = match
        (bio, bioes) {
        (Some(b), _) => {
          let ts = split_comma(b)
          check_tag_count(ts, crf)
          (Some(lift(() => @lib.bio_mask(ts))), Some(ts))
        }
        (_, Some(b)) => {
          let ts = split_comma(b)
          check_tag_count(ts, crf)
          (Some(lift(() => @lib.bioes_mask(ts))), Some(ts))
        }
        _ => (None, None)
      }
      report(crf, emissions, ca, tags, nbest)
    }
    (None, None) => demo(nbest)
    _ => raise Msg("--model and --emit must be given together")
  }
}

///|
fn check_tag_count(tags : Array[String], crf : @lib.Crf) -> Unit raise CliError {
  if tags.length() != crf.tag_count() {
    raise Msg("tag list length must equal model num_tags")
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
fn split_comma(s : String) -> Array[String] {
  let out : Array[String] = []
  let mut cur = ""
  for c in s.to_array() {
    if c == ',' {
      if cur.length() > 0 {
        out.push(cur)
      }
      cur = ""
    } else {
      cur = cur + c.to_string()
    }
  }
  if cur.length() > 0 {
    out.push(cur)
  }
  out
}

///|
fn parse_positive(s : String) -> Int32 raise CliError {
  if s.length() == 0 {
    raise Msg("--nbest requires a positive integer")
  }
  let mut v = 0
  for c in s.to_array() {
    if !c.is_ascii_digit() {
      raise Msg("--nbest requires a positive integer")
    }
    v = v * 10 + (c.to_int() - '0'.to_int())
  }
  if v <= 0 {
    raise Msg("--nbest requires a positive integer")
  }
  v
}

///|
fn demo(nbest : Int32) -> Unit raise CliError {
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
  report(crf, emit, Some(ca), Some(tags), nbest)
}

///|
fn report(
  crf : @lib.Crf,
  emit : Array[Array[Double]],
  ca : @lib.Constraints?,
  tags : Array[String]?,
  n : Int32,
) -> Unit raise CliError {
  let path = lift(() => crf.decode(emit, None, ca))
  let vp = lift(() => crf.viterbi(emit, None, ca))
  let lz = lift(() => crf.log_partition(emit, None, ca))
  let mg = lift(() => crf.marginals(emit, None, ca))
  let conf = lift(() => crf.confidence(emit, None, ca))
  let nb = lift(() => crf.nbest(emit, None, ca, n))
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
  println("confidence (max_prob / entropy):")
  for t in 0..<conf.max_prob.length() {
    println("  t=\{t}: \{conf.max_prob[t]} / \{conf.entropy[t]}")
  }
  println("top-\{n} paths:")
  for p in nb {
    match tags {
      Some(ts) =>
        println(
          "  [\{fmt_ints(p.tags)}] \{fmt_names(ts, p.tags)}  score=\{p.score}",
        )
      None => println("  [\{fmt_ints(p.tags)}]  score=\{p.score}")
    }
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
