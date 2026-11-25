// 数学底座：稳定的 log-sum-exp。CRF 的 forward / backward / 边际概率
// 全部在 log 域递推，避免概率连乘下溢。
//
// 算法参照 Rabiner (1989) HMM tutorial 与 torchcrf 的 _compute_normalizer，
// 用 max-shift 技巧保证 exp 的参数 <= 0。

///|
/// 负无穷哨兵。被约束屏蔽的转移/发射置为此值；
/// exp(neg_inf - max) 下溢为 0，等价于概率 0。
let neg_inf : Double = -1.0e308

///|
fn max_d(a : Double, b : Double) -> Double {
  if a > b {
    a
  } else {
    b
  }
}

///|
/// 稳定 log-sum-exp：`log(sum_i exp(xs[i]))`。
/// 空数组或全为负无穷时返回 `neg_inf`。
pub fn logsumexp(xs : Array[Double]) -> Double {
  let n = xs.length()
  if n == 0 {
    return neg_inf
  }
  let mut m = neg_inf
  for i in 0..<n {
    m = max_d(m, xs[i])
  }
  if m <= -1.0e300 {
    return neg_inf
  }
  let mut s = 0.0
  for i in 0..<n {
    s = s + @math.exp(xs[i] - m)
  }
  m + @math.ln(s)
}
