// 后验解码与路径合法性校验。
//
// Viterbi 求"全局最优路径"；posterior_decode 求"逐位置期望正确数最大"的标签
// （边际最大后验，MBR 的逐位置特例）。两种准则在概率上不等价：后验路径可能
// 违反结构约束，用 validate_path 检查，或改用 decode 保证合法。

///|
/// 逐位置取边际概率最大的标签。返回路径长度 = 有效位置数。
/// 注意：结果可能不满足 constraints 的转移合法性（这是该准则的固有性质），
/// 需要合法路径时请用 `Crf::decode`，或用 `Crf::validate_path` 校验。
pub fn Crf::posterior_decode(
  self : Crf,
  emissions : Array[Array[Double]],
  mask : Array[Bool]?,
  constraints : Constraints?,
) -> Array[Int32] raise CrfError {
  let mg = self.marginals(emissions, mask, constraints)
  Array::makei(mg.length(), t => {
    let mut best = 0
    for j in 1..<self.num_tags {
      if mg[t][j] > mg[t][best] {
        best = j
      }
    }
    best
  })
}

///|
/// 校验路径在给定约束下是否合法：标签范围、起始/结束标签、相邻转移。
/// 空路径 raise EmptySequence；越界标签 raise InvalidTags。
/// constraints 为 None 时只校验标签范围。
pub fn Crf::validate_path(
  self : Crf,
  tags : Array[Int32],
  constraints : Constraints?,
) -> Bool raise CrfError {
  let len = tags.length()
  if len == 0 {
    raise CrfError::EmptySequence
  }
  check_tags(tags, len, self.num_tags)
  if !start_allowed(constraints, tags[0]) {
    return false
  }
  if !end_allowed(constraints, tags[len - 1]) {
    return false
  }
  for t in 1..<len {
    if !trans_allowed(constraints, tags[t - 1], tags[t]) {
      return false
    }
  }
  true
}
