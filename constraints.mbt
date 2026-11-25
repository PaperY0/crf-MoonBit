// 转移约束：把 BIO / BIOES 标注体系的合法性编码成布尔掩码，
// 在 Viterbi / forward / backward / n-best 中屏蔽非法转移。
//
// 这是命名实体识别（NER）里防止 "I-PER 跟在 O 后面" 这类非法路径的关键。
// torchcrf 本体不带转移约束，本库将其作为一等公民提供。

///|
/// 约束掩码。start[k]/end[k] 表示首/尾位置是否允许标签 k；
/// transitions[i][j] 表示 i->j 是否合法。
#derive(Eq, Debug)
pub struct Constraints {
  start : Array[Bool]
  end : Array[Bool]
  transitions : Array[Array[Bool]]
}

///|
/// 全允许约束。
pub fn Constraints::allow_all(num_tags : Int32) -> Constraints {
  Constraints::{
    start: Array::make(num_tags, true),
    end: Array::make(num_tags, true),
    transitions: Array::makei(num_tags, _ => Array::make(num_tags, true)),
  }
}

///|
/// 解析标签前缀与实体类型。"B-PER" -> ("B","PER")，"O" -> ("O","")。
fn parse_tag(tag : String) -> (String, String) {
  match tag.find("-") {
    Some(idx) => (tag[:idx].to_owned(), tag[idx + 1:].to_owned())
    None => (tag, "")
  }
}

///|
fn validate_prefixes(
  tags : Array[String],
  allowed : Array[String],
) -> Unit raise CrfError {
  for i in 0..<tags.length() {
    let (p, _) = parse_tag(tags[i])
    let mut ok = false
    for j in 0..<allowed.length() {
      if p == allowed[j] {
        ok = true
      }
    }
    if !ok {
      raise CrfError::InvalidConstraint("tag \{tags[i]} has illegal prefix")
    }
  }
}

///|
/// BIO 约束：标签形如 "O" / "B-X" / "I-X"。
/// I-X 只能跟在 B-X / I-X 之后；首位不允许 I-X。
pub fn bio_mask(tags : Array[String]) -> Constraints raise CrfError {
  let k = tags.length()
  if k == 0 {
    raise CrfError::InvalidConstraint("empty tag set")
  }
  validate_prefixes(tags, ["B", "I", "O"])
  let prefixes = Array::makei(k, i => {
    let (p, _) = parse_tag(tags[i])
    p
  })
  let types = Array::makei(k, i => {
    let (_, ty) = parse_tag(tags[i])
    ty
  })
  let start = Array::makei(k, i => prefixes[i] != "I")
  let end = Array::make(k, true)
  let trans = Array::makei(k, i => {
    Array::makei(k, j => {
      if prefixes[j] == "I" {
        (prefixes[i] == "B" || prefixes[i] == "I") && types[i] == types[j]
      } else {
        true
      }
    })
  })
  Constraints::{ start, end, transitions: trans, }
}

///|
/// BIOES 约束：标签形如 "O" / "B-X" / "I-X" / "E-X" / "S-X"。
/// B-X/I-X 只能转入同类型 I-X/E-X；E-X/S-X/O 之后只能开新实体(B/S)或 O；
/// 首位允许 B/S/O，尾位允许 E/S/O。
pub fn bioes_mask(tags : Array[String]) -> Constraints raise CrfError {
  let k = tags.length()
  if k == 0 {
    raise CrfError::InvalidConstraint("empty tag set")
  }
  validate_prefixes(tags, ["B", "I", "E", "S", "O"])
  let prefixes = Array::makei(k, i => {
    let (p, _) = parse_tag(tags[i])
    p
  })
  let types = Array::makei(k, i => {
    let (_, ty) = parse_tag(tags[i])
    ty
  })
  let complete = Array::makei(k, i => {
    prefixes[i] == "E" || prefixes[i] == "S" || prefixes[i] == "O"
  })
  let start = Array::makei(k, i => {
    prefixes[i] == "B" || prefixes[i] == "S" || prefixes[i] == "O"
  })
  let end = Array::makei(k, i => complete[i])
  let trans = Array::makei(k, i => {
    Array::makei(k, j => {
      if prefixes[j] == "I" || prefixes[j] == "E" {
        (prefixes[i] == "B" || prefixes[i] == "I") && types[i] == types[j]
      } else {
        complete[i]
      }
    })
  })
  Constraints::{ start, end, transitions: trans, }
}

///|
/// 取约束下 i->j 是否允许（None 视为全允许）。
fn trans_allowed(c : Constraints?, i : Int32, j : Int32) -> Bool {
  match c {
    None => true
    Some(cc) => cc.transitions[i][j]
  }
}

///|
fn start_allowed(c : Constraints?, k : Int32) -> Bool {
  match c {
    None => true
    Some(cc) => cc.start[k]
  }
}

///|
fn end_allowed(c : Constraints?, k : Int32) -> Bool {
  match c {
    None => true
    Some(cc) => cc.end[k]
  }
}
