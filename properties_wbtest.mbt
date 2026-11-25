// 性质测试：对确定性伪随机模型（sin 权重）断言跨模型不变量。
// 与参考向量互补：向量验证"数值精确"，性质验证"关系恒成立"。

///|
fn sin_w(a : Int32, b : Int32, scale : Double) -> Double {
  @math.sin((a * 31 + b * 17 + 7).to_double()) * scale
}

///|
test "properties: score <= logZ, marginals normalized, nbest consistent" {
  for seed in 0..<12 {
    let k = 2 + seed % 3
    let len = 1 + seed % 6
    let start = Array::makei(k, j => sin_w(seed, j, 0.7))
    let end = Array::makei(k, j => sin_w(seed + 100, j, 0.7))
    let trans = Array::makei(k, i => {
      Array::makei(k, j => sin_w(seed + i, j, 0.9))
    })
    let emit = Array::makei(len, t => {
      Array::makei(k, j => sin_w(seed * 3 + t, j, 1.5))
    })
    let crf = Crf::new(k, start, end, trans) catch { e => fail(e.message()) }
    let vp = crf.viterbi(emit, None, None) catch { e => fail(e.message()) }
    let lz = crf.log_partition(emit, None, None) catch {
      e => fail(e.message())
    }
    assert_true(vp.score <= lz + 1.0e-9, msg="score must not exceed logZ")
    let sp = crf.score_path(emit, vp.tags, None) catch {
      e => fail(e.message())
    }
    close(sp, vp.score, "score_path == viterbi")
    let mg = crf.marginals(emit, None, None) catch { e => fail(e.message()) }
    for t in 0..<len {
      let mut s = 0.0
      for j in 0..<k {
        assert_true(
          mg[t][j] >= -1.0e-12 && mg[t][j] <= 1.0 + 1.0e-12,
          msg="marginal in [0,1]",
        )
        s = s + mg[t][j]
      }
      close(s, 1.0, "row sum")
    }
    let nb = crf.nbest(emit, None, None, 3) catch { e => fail(e.message()) }
    assert_eq(nb[0].tags, vp.tags)
    close(nb[0].score, vp.score, "nbest[0]")
    for i in 1..<nb.length() {
      assert_true(nb[i - 1].score >= nb[i].score, msg="nbest sorted")
    }
    let ll = crf.log_likelihood(emit, vp.tags, None, None) catch {
      e => fail(e.message())
    }
    assert_true(ll <= 1.0e-9, msg="ll <= 0")
    let pd = crf.posterior_decode(emit, None, None) catch {
      e => fail(e.message())
    }
    assert_eq(pd.length(), len)
  }
}

///|
test "properties: constrained decode and nbest always legal" {
  let bio_tags = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
  let bioes_tags = ["O", "B-PER", "I-PER", "E-PER", "S-PER"]
  for seed in 0..<8 {
    let len = 2 + seed % 5
    let emit = Array::makei(len, t => {
      Array::makei(5, j => sin_w(seed * 5 + t, j, 2.0))
    })
    let crf = Crf::uniform(5) catch { e => fail(e.message()) }
    let cb = bio_mask(bio_tags) catch { e => fail(e.message()) }
    let ce = bioes_mask(bioes_tags) catch { e => fail(e.message()) }
    let pb = crf.decode(emit, None, Some(cb)) catch { e => fail(e.message()) }
    let okb = crf.validate_path(pb, Some(cb)) catch { e => fail(e.message()) }
    assert_true(okb, msg="BIO decode path must be legal")
    let pe = crf.decode(emit, None, Some(ce)) catch { e => fail(e.message()) }
    let oke = crf.validate_path(pe, Some(ce)) catch { e => fail(e.message()) }
    assert_true(oke, msg="BIOES decode path must be legal")
    let nbe = crf.nbest(emit, None, Some(ce), 4) catch {
      e => fail(e.message())
    }
    for p in nbe {
      let ok = crf.validate_path(p.tags, Some(ce)) catch {
        e => fail(e.message())
      }
      assert_true(ok, msg="BIOES nbest path must be legal")
    }
  }
}
