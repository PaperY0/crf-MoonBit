// 逐位置置信度：把后验边际分布压缩成两个可操作的标量。
//
// max_prob[t] = 位置 t 的最大边际概率（top-1 后验置信度）；
// entropy[t]  = 位置 t 边际分布的 Shannon 熵（nats）。
// 低 max_prob / 高熵 = 模型在该位置不确定，可用于序列标注的幻觉筛查、
// 人工复核分流与主动学习采样。

///|
/// 逐位置置信度报告，两个数组长度均 = 有效位置数。
#derive(Eq, Debug)
pub struct Confidence {
  /// 每个位置的最大边际概率，范围 [1/K, 1]。
  max_prob : Array[Double]
  /// 每个位置边际分布的 Shannon 熵（自然对数，nats），范围 [0, ln K]。
  entropy : Array[Double]
}

///|
/// 计算逐位置置信度。约束与 mask 语义同 `Crf::marginals`。
pub fn Crf::confidence(
  self : Crf,
  emissions : Array[Array[Double]],
  mask : Array[Bool]?,
  constraints : Constraints?,
) -> Confidence raise CrfError {
  let mg = self.marginals(emissions, mask, constraints)
  let max_prob = Array::makei(mg.length(), t => {
    let mut m = 0.0
    for j in 0..<self.num_tags {
      if mg[t][j] > m {
        m = mg[t][j]
      }
    }
    m
  })
  let entropy = Array::makei(mg.length(), t => {
    let mut h = 0.0
    for j in 0..<self.num_tags {
      let p = mg[t][j]
      if p > 0.0 {
        h = h - p * @math.ln(p)
      }
    }
    h
  })
  Confidence::{ max_prob, entropy, }
}
