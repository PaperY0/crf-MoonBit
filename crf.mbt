// 门面：批量推理与模型访问器。

///|
/// 标签数量。
pub fn Crf::tag_count(self : Crf) -> Int32 {
  self.num_tags
}

///|
/// 校验批掩码长度与批大小一致。
fn check_batch_masks(
  masks : Array[Array[Bool]]?,
  n : Int32,
) -> Unit raise CrfError {
  match masks {
    None => ()
    Some(mm) =>
      if mm.length() != n {
        raise CrfError::InvalidShape("masks length != batch size")
      }
  }
}

///|
/// 取第 b 条序列的掩码。
fn batch_mask_at(masks : Array[Array[Bool]]?, b : Int32) -> Array[Bool]? {
  match masks {
    None => None
    Some(mm) => Some(mm[b])
  }
}

///|
/// 批量 Viterbi 解码。masks 为 None 时各序列均无 padding；
/// 为 Some 时长度须与批大小一致。
pub fn Crf::decode_batch(
  self : Crf,
  emissions_batch : Array[Array[Array[Double]]],
  masks : Array[Array[Bool]]?,
  constraints : Constraints?,
) -> Array[Array[Int32]] raise CrfError {
  let n = emissions_batch.length()
  check_batch_masks(masks, n)
  let out : Array[Array[Int32]] = []
  for b in 0..<n {
    out.push(
      self.decode(emissions_batch[b], batch_mask_at(masks, b), constraints),
    )
  }
  out
}

///|
/// 批量前向–后向边际概率，语义与逐条 `marginals` 一致。
/// masks 为 None 时各序列均无 padding；为 Some 时长度须与批大小一致。
pub fn Crf::marginals_batch(
  self : Crf,
  emissions_batch : Array[Array[Array[Double]]],
  masks : Array[Array[Bool]]?,
  constraints : Constraints?,
) -> Array[Array[Array[Double]]] raise CrfError {
  let n = emissions_batch.length()
  check_batch_masks(masks, n)
  let out : Array[Array[Array[Double]]] = []
  for b in 0..<n {
    out.push(
      self.marginals(emissions_batch[b], batch_mask_at(masks, b), constraints),
    )
  }
  out
}

///|
/// 批量置信度（最大边际概率 + Shannon 熵），语义与逐条 `confidence` 一致。
/// masks 为 None 时各序列均无 padding；为 Some 时长度须与批大小一致。
pub fn Crf::confidence_batch(
  self : Crf,
  emissions_batch : Array[Array[Array[Double]]],
  masks : Array[Array[Bool]]?,
  constraints : Constraints?,
) -> Array[Confidence] raise CrfError {
  let n = emissions_batch.length()
  check_batch_masks(masks, n)
  let out : Array[Confidence] = []
  for b in 0..<n {
    out.push(
      self.confidence(emissions_batch[b], batch_mask_at(masks, b), constraints),
    )
  }
  out
}
