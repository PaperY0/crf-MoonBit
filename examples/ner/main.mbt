// 最小端到端示例：JSON 权重加载 → BIO 约束 → 标签名解码 → 置信度 → n-best。
// 运行：moon run examples/ner

///|
fn main {
  run() catch {
    e => eprintln("error: \{e.message()}")
  }
}

///|
fn run() -> Unit raise @lib.CrfError {
  let tags = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
  // torchcrf 兼容权重 JSON。真实项目中由 tools/export_torchcrf.py
  // 从训练好的 state_dict 导出；这里手写一组演示权重。
  let weights_json =
    #|{
    #|  "num_tags": 5,
    #|  "start_transitions": [-0.5, 0.8, -2.0, 0.8, -2.0],
    #|  "end_transitions": [0.5, -0.5, 0.3, -0.5, 0.3],
    #|  "transitions": [
    #|    [0.5, 0.6, -3.0, 0.6, -3.0],
    #|    [-0.2, -1.0, 1.2, -1.0, -2.0],
    #|    [0.6, 0.2, 0.8, 0.2, -2.0],
    #|    [-0.2, -1.0, -2.0, -1.0, 1.2],
    #|    [0.6, 0.2, -2.0, 0.2, 0.8]
    #|  ]
    #|}
  let crf = @lib.Crf::from_json(weights_json)
  let ca = @lib.bio_mask(tags)
  // 上游编码器对 "Sherry lives in Qingdao" 逐 token 输出的发射分。
  let tokens = ["Sherry", "lives", "in", "Qingdao"]
  let emissions = [
    [0.1, 2.2, 0.0, 0.2, 0.0],
    [1.8, 0.0, 0.1, 0.0, 0.0],
    [1.5, 0.0, 0.0, 0.1, 0.0],
    [0.2, 0.1, 0.0, 2.4, 0.0],
  ]
  println("== tokens ==")
  let names = crf.decode_labels(tags, emissions, None, Some(ca))
  for i in 0..<tokens.length() {
    println("\{tokens[i]} -> \{names[i]}")
  }
  println("== confidence ==")
  let conf = crf.confidence(emissions, None, Some(ca))
  for i in 0..<conf.max_prob.length() {
    println("t=\{i} max_prob=\{conf.max_prob[i]} entropy=\{conf.entropy[i]}")
  }
  println("== top-3 candidates ==")
  let nb = crf.nbest(emissions, None, Some(ca), 3)
  for p in nb {
    let ns = p.labels(tags)
    let mut line = ""
    for j in 0..<ns.length() {
      if j > 0 {
        line = line + " "
      }
      line = line + ns[j]
    }
    println("[\{line}] score=\{p.score}")
  }
}
