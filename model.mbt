// 模型定义与输入校验。
//
// 权重布局与 torchcrf (pytorch-crf) 完全一致，便于直接导入训练好的参数：
//   start_transitions[k]      : START -> k 的转移分
//   end_transitions[k]        : k -> STOP 的转移分
//   transitions[i][j]         : i -> j 的转移分
// 发射分 emissions[t][k] 由上游编码器（BiLSTM / BERT 等）给出，本库只负责
// 给定发射分与转移分后的**结构化推理**，不做神经网络前向，也不做训练。

///|
/// 线性链 CRF 模型（推理用）。
#derive(Eq, Debug)
pub struct Crf {
  num_tags : Int32
  start_transitions : Array[Double]
  end_transitions : Array[Double]
  transitions : Array[Array[Double]]
}

///|
/// 解码结果：一条标签路径及其未归一化 log 分数。
#derive(Eq, Debug)
pub struct DecodedPath {
  tags : Array[Int32]
  score : Double
}

///|
/// 错误类型。
#derive(Eq, Debug)
pub suberror CrfError {
  InvalidShape(String)
  EmptySequence
  InvalidTags(String)
  InvalidConstraint(String)
  JsonError(String)
}

///|
/// 人类可读错误信息。
pub fn CrfError::message(self : CrfError) -> String {
  match self {
    InvalidShape(s) => "invalid shape: " + s
    EmptySequence => "empty sequence"
    InvalidTags(s) => "invalid tags: " + s
    InvalidConstraint(s) => "invalid constraint: " + s
    JsonError(s) => "json error: " + s
  }
}

///|
/// 构造模型并校验维度。
pub fn Crf::new(
  num_tags : Int32,
  start_transitions : Array[Double],
  end_transitions : Array[Double],
  transitions : Array[Array[Double]],
) -> Crf raise CrfError {
  if num_tags <= 0 {
    raise CrfError::InvalidShape("num_tags must be positive")
  }
  if start_transitions.length() != num_tags {
    raise CrfError::InvalidShape("start_transitions length != num_tags")
  }
  if end_transitions.length() != num_tags {
    raise CrfError::InvalidShape("end_transitions length != num_tags")
  }
  if transitions.length() != num_tags {
    raise CrfError::InvalidShape("transitions rows != num_tags")
  }
  for i in 0..<num_tags {
    if transitions[i].length() != num_tags {
      raise CrfError::InvalidShape("transitions row \{i} length != num_tags")
    }
  }
  check_finite(start_transitions, "start_transitions")
  check_finite(end_transitions, "end_transitions")
  for i in 0..<num_tags {
    check_finite(transitions[i], "transitions row \{i}")
  }
  Crf::{ num_tags, start_transitions, end_transitions, transitions, }
}

///|
/// 全零权重模型（均匀转移）。常用于演示与测试基线。
pub fn Crf::uniform(num_tags : Int32) -> Crf raise CrfError {
  let z = Array::make(num_tags, 0.0)
  let t = Array::makei(num_tags, _ => Array::make(num_tags, 0.0))
  Crf::new(num_tags, z, z.copy(), t)
}

///|
/// 校验数组全部为有限数（非 NaN / ±Inf），否则报 InvalidShape。
fn check_finite(arr : Array[Double], what : String) -> Unit raise CrfError {
  for i in 0..<arr.length() {
    if !arr[i].is_finite() {
      raise CrfError::InvalidShape("\{what}[\{i}] is not finite")
    }
  }
}

///|
/// 校验发射分矩阵形状 (T x K) 与有限性，返回序列长度 T。
fn check_emissions(
  emissions : Array[Array[Double]],
  num_tags : Int32,
) -> Int32 raise CrfError {
  let t = emissions.length()
  for i in 0..<t {
    if emissions[i].length() != num_tags {
      raise CrfError::InvalidShape("emissions row \{i} length != num_tags")
    }
    for j in 0..<num_tags {
      if !emissions[i][j].is_finite() {
        raise CrfError::InvalidShape("emissions[\{i}][\{j}] is not finite")
      }
    }
  }
  t
}

///|
/// 按 mask 过滤出有效位置（支持任意位置的 padding，不要求前缀连续）。
/// 返回有效发射分序列；mask 为 None 时原样返回（已校验形状）。
fn effective_emissions(
  emissions : Array[Array[Double]],
  mask : Array[Bool]?,
  num_tags : Int32,
) -> Array[Array[Double]] raise CrfError {
  let t = check_emissions(emissions, num_tags)
  match mask {
    None => emissions
    Some(m) => {
      if m.length() != t {
        raise CrfError::InvalidShape("mask length != emissions length")
      }
      let out : Array[Array[Double]] = []
      for i in 0..<t {
        if m[i] {
          out.push(emissions[i])
        }
      }
      out
    }
  }
}

///|
/// 校验标签路径长度与取值范围。
fn check_tags(
  tags : Array[Int32],
  len : Int32,
  num_tags : Int32,
) -> Unit raise CrfError {
  if tags.length() != len {
    raise CrfError::InvalidTags("tags length != valid sequence length")
  }
  for i in 0..<tags.length() {
    if tags[i] < 0 || tags[i] >= num_tags {
      raise CrfError::InvalidTags("tag \{tags[i]} out of range")
    }
  }
}
