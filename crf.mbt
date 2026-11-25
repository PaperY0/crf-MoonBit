// 门面：批量解码与模型访问器。

///|
/// 标签数量。
pub fn Crf::tag_count(self : Crf) -> Int32 {
  self.num_tags
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
  match masks {
    None => ()
    Some(mm) =>
      if mm.length() != n {
        raise CrfError::InvalidShape("masks length != batch size")
      }
  }
  let out : Array[Array[Int32]] = []
  for b in 0..<n {
    let m = match masks {
      None => None
      Some(mm) => Some(mm[b])
    }
    out.push(self.decode(emissions_batch[b], m, constraints))
  }
  out
}
