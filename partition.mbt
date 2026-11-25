// 配分函数（forward）、后验边际（forward-backward）、路径打分与对数似然。
//
// forward 递推参照 torchcrf._compute_normalizer：
//   alpha_0[j]   = start[j] + emit_0[j]
//   alpha_t[j]   = logsumexp_i(alpha_{t-1}[i] + trans[i][j]) + emit_t[j]
//   log Z        = logsumexp_j(alpha_{L-1}[j] + end[j])
// 边际概率 marginal_t[j] = exp(alpha_t[j] + beta_t[j] - log Z)，每行和为 1。

///|
fn forward_logz(
  crf : Crf,
  e : Array[Array[Double]],
  constraints : Constraints?,
) -> Double raise CrfError {
  let len = e.length()
  if len == 0 {
    raise CrfError::EmptySequence
  }
  let k = crf.num_tags
  let mut score = Array::makei(k, j => {
    if start_allowed(constraints, j) {
      crf.start_transitions[j] + e[0][j]
    } else {
      neg_inf
    }
  })
  for t in 1..<len {
    let prev = score
    let et = e[t]
    let next = Array::makei(k, j => {
      let terms = Array::makei(k, i => {
        if trans_allowed(constraints, i, j) {
          prev[i] + crf.transitions[i][j]
        } else {
          neg_inf
        }
      })
      logsumexp(terms) + et[j]
    })
    score = next
  }
  let terms = Array::makei(k, j => {
    if end_allowed(constraints, j) {
      score[j] + crf.end_transitions[j]
    } else {
      neg_inf
    }
  })
  logsumexp(terms)
}

///|
/// 前向算法求 log 配分函数 log Z（归一化常数的对数）。
pub fn Crf::log_partition(
  self : Crf,
  emissions : Array[Array[Double]],
  mask : Array[Bool]?,
  constraints : Constraints?,
) -> Double raise CrfError {
  let e = effective_emissions(emissions, mask, self.num_tags)
  forward_logz(self, e, constraints)
}

///|
/// 给定标签路径的未归一化 log 分数（分子）。
pub fn Crf::score_path(
  self : Crf,
  emissions : Array[Array[Double]],
  tags : Array[Int32],
  mask : Array[Bool]?,
) -> Double raise CrfError {
  let e = effective_emissions(emissions, mask, self.num_tags)
  let len = e.length()
  if len == 0 {
    raise CrfError::EmptySequence
  }
  check_tags(tags, len, self.num_tags)
  let mut s = self.start_transitions[tags[0]] + e[0][tags[0]]
  for t in 1..<len {
    s = s + self.transitions[tags[t - 1]][tags[t]] + e[t][tags[t]]
  }
  s + self.end_transitions[tags[len - 1]]
}

///|
/// 对数似然 log P(tags | emissions) = score_path - log Z。
/// 训练时最小化其相反数即为 CRF 损失（本库专注推理，训练在下游框架完成）。
pub fn Crf::log_likelihood(
  self : Crf,
  emissions : Array[Array[Double]],
  tags : Array[Int32],
  mask : Array[Bool]?,
  constraints : Constraints?,
) -> Double raise CrfError {
  let e = effective_emissions(emissions, mask, self.num_tags)
  let sp = self.score_path(emissions, tags, mask)
  let lz = forward_logz(self, e, constraints)
  sp - lz
}

///|
/// forward-backward 后验边际概率，形状 (L x K)，每行和为 1。
/// 可用作逐 token 置信度（例如检索增强生成里的序列标注幻觉筛查）。
pub fn Crf::marginals(
  self : Crf,
  emissions : Array[Array[Double]],
  mask : Array[Bool]?,
  constraints : Constraints?,
) -> Array[Array[Double]] raise CrfError {
  let e = effective_emissions(emissions, mask, self.num_tags)
  let len = e.length()
  if len == 0 {
    raise CrfError::EmptySequence
  }
  let k = self.num_tags
  // forward: alpha[t][j]
  let alpha : Array[Array[Double]] = []
  alpha.push(
    Array::makei(k, j => {
      if start_allowed(constraints, j) {
        self.start_transitions[j] + e[0][j]
      } else {
        neg_inf
      }
    }),
  )
  for t in 1..<len {
    let prev = alpha[t - 1]
    let et = e[t]
    alpha.push(
      Array::makei(k, j => {
        let terms = Array::makei(k, i => {
          if trans_allowed(constraints, i, j) {
            prev[i] + self.transitions[i][j]
          } else {
            neg_inf
          }
        })
        logsumexp(terms) + et[j]
      }),
    )
  }
  let last = alpha[len - 1]
  let zterms = Array::makei(k, j => {
    if end_allowed(constraints, j) {
      last[j] + self.end_transitions[j]
    } else {
      neg_inf
    }
  })
  let logz = logsumexp(zterms)
  if logz <= -1.0e300 {
    raise CrfError::InvalidConstraint("no valid path under constraints")
  }
  // backward: beta[t][i]
  let beta : Array[Array[Double]] = Array::makei(len, _ => {
    Array::make(k, neg_inf)
  })
  beta[len - 1] = Array::makei(k, j => {
    if end_allowed(constraints, j) {
      self.end_transitions[j]
    } else {
      neg_inf
    }
  })
  let mut t = len - 2
  while t >= 0 {
    let nxt = beta[t + 1]
    let enext = e[t + 1]
    beta[t] = Array::makei(k, i => {
      let terms = Array::makei(k, j => {
        if trans_allowed(constraints, i, j) {
          self.transitions[i][j] + enext[j] + nxt[j]
        } else {
          neg_inf
        }
      })
      logsumexp(terms)
    })
    t -= 1
  }
  Array::makei(len, t => {
    Array::makei(k, j => @math.exp(alpha[t][j] + beta[t][j] - logz))
  })
}
