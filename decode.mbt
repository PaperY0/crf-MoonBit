// Viterbi 最优路径解码 + 精确 n-best。
// 全程 log 域，转移约束以 neg_inf 屏蔽。

///|
fn viterbi_core(
  crf : Crf,
  e : Array[Array[Double]],
  constraints : Constraints?,
) -> DecodedPath raise CrfError {
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
  let back : Array[Array[Int32]] = []
  for t in 1..<len {
    let bt = Array::make(k, 0)
    let next = Array::make(k, neg_inf)
    for j in 0..<k {
      let mut best = neg_inf
      let mut bi = 0
      for i in 0..<k {
        if trans_allowed(constraints, i, j) {
          let cand = score[i] + crf.transitions[i][j]
          if cand > best {
            best = cand
            bi = i
          }
        }
      }
      next[j] = best + e[t][j]
      bt[j] = bi
    }
    back.push(bt)
    score = next
  }
  let mut best = neg_inf
  let mut bk = 0
  for j in 0..<k {
    if end_allowed(constraints, j) {
      let cand = score[j] + crf.end_transitions[j]
      if cand > best {
        best = cand
        bk = j
      }
    }
  }
  if best <= -1.0e300 {
    raise CrfError::InvalidConstraint("no valid path under constraints")
  }
  let path = Array::make(len, 0)
  path[len - 1] = bk
  let mut cur = bk
  let mut t = len - 1
  while t > 0 {
    cur = back[t - 1][cur]
    path[t - 1] = cur
    t -= 1
  }
  DecodedPath::{ tags: path, score: best, }
}

///|
/// Viterbi 解码，返回最优标签路径（长度 = 有效位置数）。
pub fn Crf::decode(
  self : Crf,
  emissions : Array[Array[Double]],
  mask : Array[Bool]?,
  constraints : Constraints?,
) -> Array[Int32] raise CrfError {
  let e = effective_emissions(emissions, mask, self.num_tags)
  viterbi_core(self, e, constraints).tags
}

///|
/// Viterbi 解码，返回最优路径及其未归一化 log 分数。
pub fn Crf::viterbi(
  self : Crf,
  emissions : Array[Array[Double]],
  mask : Array[Bool]?,
  constraints : Constraints?,
) -> DecodedPath raise CrfError {
  let e = effective_emissions(emissions, mask, self.num_tags)
  viterbi_core(self, e, constraints)
}

///|
/// 按分数降序取前 n 个（插入排序，n 与候选数都很小）。
fn top_n(cands : Array[DecodedPath], n : Int32) -> Array[DecodedPath] {
  let a = cands.copy()
  for i in 1..<a.length() {
    let key = a[i]
    let mut j = i - 1
    while j >= 0 && a[j].score < key.score {
      a[j + 1] = a[j]
      j -= 1
    }
    a[j + 1] = key
  }
  let m = if n < a.length() { n } else { a.length() }
  let out : Array[DecodedPath] = []
  for i in 0..<m {
    out.push(a[i])
  }
  out
}

///|
/// 精确 n-best 解码：每个状态保留至多 n 条最优部分路径，逐位置扩展。
/// 返回按分数降序排列的前 n 条完整路径。
pub fn Crf::nbest(
  self : Crf,
  emissions : Array[Array[Double]],
  mask : Array[Bool]?,
  constraints : Constraints?,
  n : Int32,
) -> Array[DecodedPath] raise CrfError {
  if n <= 0 {
    raise CrfError::InvalidShape("n must be positive")
  }
  let e = effective_emissions(emissions, mask, self.num_tags)
  let len = e.length()
  if len == 0 {
    raise CrfError::EmptySequence
  }
  let k = self.num_tags
  let mut beams : Array[Array[DecodedPath]] = Array::makei(k, j => {
    if start_allowed(constraints, j) {
      [DecodedPath::{ tags: [j], score: self.start_transitions[j] + e[0][j], }]
    } else {
      []
    }
  })
  for t in 1..<len {
    let new_beams : Array[Array[DecodedPath]] = Array::makei(k, j => {
      let cands : Array[DecodedPath] = []
      for i in 0..<k {
        if trans_allowed(constraints, i, j) {
          for item in beams[i] {
            let np = item.tags.copy()
            np.push(j)
            cands.push(DecodedPath::{
              tags: np,
              score: item.score + self.transitions[i][j] + e[t][j],
            })
          }
        }
      }
      top_n(cands, n)
    })
    beams = new_beams
  }
  let cands : Array[DecodedPath] = []
  for j in 0..<k {
    if end_allowed(constraints, j) {
      for item in beams[j] {
        cands.push(DecodedPath::{
          tags: item.tags,
          score: item.score + self.end_transitions[j],
        })
      }
    }
  }
  if cands.length() == 0 {
    raise CrfError::InvalidConstraint("no valid path under constraints")
  }
  top_n(cands, n)
}
