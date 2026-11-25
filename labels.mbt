// 标签名便捷接口：把索引路径映射为人类可读标签。

///|
/// 校验索引在标签名表范围内并逐一映射。
fn map_labels(
  tags : Array[String],
  path : Array[Int32],
) -> Array[String] raise CrfError {
  check_tags(path, path.length(), tags.length())
  Array::makei(path.length(), i => tags[path[i]])
}

///|
/// Viterbi 解码并直接返回标签名。tags 为标签名表，
/// 下标即标签编号，长度须等于模型标签数。
pub fn Crf::decode_labels(
  self : Crf,
  tags : Array[String],
  emissions : Array[Array[Double]],
  mask : Array[Bool]?,
  constraints : Constraints?,
) -> Array[String] raise CrfError {
  if tags.length() != self.num_tags {
    raise CrfError::InvalidTags("tag names length != num_tags")
  }
  let path = self.decode(emissions, mask, constraints)
  map_labels(tags, path)
}

///|
/// 把解码结果（Viterbi / n-best）的索引路径映射为标签名。
/// 路径索引超出名表范围时报 InvalidTags。
pub fn DecodedPath::labels(
  self : DecodedPath,
  tags : Array[String],
) -> Array[String] raise CrfError {
  map_labels(tags, self.tags)
}
