// 权重 JSON  interchange：与 torchcrf state_dict 导出的 JSON 互通。
//
//  schema:
//   { "num_tags": K,
//     "start_transitions": [K],
//     "end_transitions":   [K],
//     "transitions":       [[K] x K] }
//
// tools/export_torchcrf.py 可把训练好的 torchcrf 参数导出为此格式。

///|
fn[T : @json.FromJson] decode_field(
  m : Map[String, @json.Json],
  key : String,
) -> T raise CrfError {
  let j = match m.get(key) {
    Some(v) => v
    None => raise CrfError::JsonError("missing field: " + key)
  }
  @json.from_json(j) catch {
    e => raise CrfError::JsonError(key + ": " + e.to_string())
  }
}

///|
/// 从 JSON 加载模型（会校验维度）。
pub fn Crf::from_json(s : String) -> Crf raise CrfError {
  let j = @json.parse(s[:]) catch {
    e => raise CrfError::JsonError(e.to_string())
  }
  let m = match j {
    Object(m) => m
    _ => raise CrfError::JsonError("model JSON must be an object")
  }
  let nt : Int32 = decode_field(m, "num_tags")
  let start : Array[Double] = decode_field(m, "start_transitions")
  let end : Array[Double] = decode_field(m, "end_transitions")
  let trans : Array[Array[Double]] = decode_field(m, "transitions")
  Crf::new(nt, start, end, trans)
}

///|
/// 导出模型为 JSON。
pub fn Crf::to_json(self : Crf) -> String {
  let m = Map::from_array([
    ("num_tags", @json.to_json(self.num_tags)),
    ("start_transitions", @json.to_json(self.start_transitions)),
    ("end_transitions", @json.to_json(self.end_transitions)),
    ("transitions", @json.to_json(self.transitions)),
  ])
  @json.Json::object(m).stringify()
}
