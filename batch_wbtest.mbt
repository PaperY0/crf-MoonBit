// 批量边际 / 置信度 API 的白盒测试：与逐条调用等价、错误分支。

///|
test "batch marginals and confidence match single-sequence calls" {
  let crf = Crf::uniform(3) catch { e => fail(e.message()) }
  let batch = [
    [[1.0, 0.0, 0.5], [0.0, 2.0, 0.0], [0.5, 0.0, 1.0]],
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
  ]
  let mm : Array[Array[Bool]] = [[true, true, true], [true, false]]
  let mb = crf.marginals_batch(batch, Some(mm), None) catch {
    e => fail(e.message())
  }
  let cb = crf.confidence_batch(batch, Some(mm), None) catch {
    e => fail(e.message())
  }
  assert_eq(mb.length(), 2)
  assert_eq(cb.length(), 2)
  for b in 0..<2 {
    let m1 = crf.marginals(batch[b], Some(mm[b]), None) catch {
      e => fail(e.message())
    }
    let c1 = crf.confidence(batch[b], Some(mm[b]), None) catch {
      e => fail(e.message())
    }
    assert_eq(mb[b].length(), m1.length())
    assert_eq(cb[b].max_prob.length(), c1.max_prob.length())
    for t in 0..<m1.length() {
      for k in 0..<3 {
        close(mb[b][t][k], m1[t][k], "batch marginal")
      }
      close(cb[b].max_prob[t], c1.max_prob[t], "batch max_prob")
      close(cb[b].entropy[t], c1.entropy[t], "batch entropy")
    }
  }
}

///|
test "batch APIs accept empty batch and no masks" {
  let crf = Crf::uniform(2) catch { e => fail(e.message()) }
  let empty : Array[Array[Array[Double]]] = []
  let me = crf.marginals_batch(empty, None, None) catch {
    e => fail(e.message())
  }
  let ce = crf.confidence_batch(empty, None, None) catch {
    e => fail(e.message())
  }
  assert_eq(me.length(), 0)
  assert_eq(ce.length(), 0)
  let one = [[[0.0, 1.0]]]
  let m1 = crf.marginals_batch(one, None, None) catch { e => fail(e.message()) }
  let m0 = crf.marginals(one[0], None, None) catch { e => fail(e.message()) }
  close(m1[0][0][1], m0[0][1], "no-mask batch marginal")
}

///|
test "batch APIs reject mask length mismatch" {
  let crf = Crf::uniform(2) catch { e => fail(e.message()) }
  let batch = [[[0.0, 1.0]], [[1.0, 0.0]]]
  let bad : Array[Array[Bool]] = [[true]]
  assert_true(raises(() => crf.marginals_batch(batch, Some(bad), None)))
  assert_true(raises(() => crf.confidence_batch(batch, Some(bad), None)))
}
