// confidence 的白盒测试：与手算 softmax 熵对照 + 取值范围不变量。

///|
test "confidence matches hand-computed softmax entropy" {
  let crf = Crf::uniform(3) catch { e => fail(e.message()) }
  let emit = [[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]]
  let conf = crf.confidence(emit, None, None) catch { e => fail(e.message()) }
  for t in 0..<2 {
    let z = logsumexp(emit[t])
    let mut h = 0.0
    let mut mx = 0.0
    for j in 0..<3 {
      let p = @math.exp(emit[t][j] - z)
      h = h - p * @math.ln(p)
      if p > mx {
        mx = p
      }
    }
    close(conf.max_prob[t], mx, "max_prob")
    close(conf.entropy[t], h, "entropy")
  }
}

///|
test "confidence bounds: uniform vs peaked" {
  let crf = Crf::uniform(2) catch { e => fail(e.message()) }
  // 完全不确定：p = 0.5/0.5，熵 = ln 2。
  let flat = crf.confidence([[0.0, 0.0]], None, None) catch {
    e => fail(e.message())
  }
  close(flat.max_prob[0], 0.5, "flat max_prob")
  close(flat.entropy[0], @math.ln(2.0), "flat entropy")
  // 高度确定：熵趋近 0，max_prob 趋近 1。
  let peak = crf.confidence([[30.0, 0.0]], None, None) catch {
    e => fail(e.message())
  }
  assert_true(peak.max_prob[0] > 0.999, msg="peaked max_prob")
  assert_true(peak.entropy[0] < 1.0e-6, msg="peaked entropy")
}

///|
test "confidence under constraints stays in range" {
  let tags = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
  let ca = bio_mask(tags) catch { e => fail(e.message()) }
  let crf = Crf::uniform(5) catch { e => fail(e.message()) }
  let emit = [
    [0.2, 2.0, 0.1, 0.0, 0.0],
    [0.1, 0.0, 1.8, 0.0, 0.0],
    [2.5, 0.0, 0.0, 0.1, 0.0],
  ]
  let conf = crf.confidence(emit, None, Some(ca)) catch {
    e => fail(e.message())
  }
  let lnk = @math.ln(5.0)
  for t in 0..<3 {
    assert_true(conf.max_prob[t] >= 0.2 - 1.0e-9, msg="max_prob >= 1/K")
    assert_true(conf.max_prob[t] <= 1.0 + 1.0e-9, msg="max_prob <= 1")
    assert_true(conf.entropy[t] >= -1.0e-9, msg="entropy >= 0")
    assert_true(conf.entropy[t] <= lnk + 1.0e-9, msg="entropy <= ln K")
  }
}
