// 白盒测试：手算不变量 + numpy 参考向量交叉验证 + 约束/n-best/mask/IO/错误。

///|
fn close(a : Double, b : Double, msg : String) -> Unit raise {
  let d = if a > b { a - b } else { b - a }
  assert_true(d < 1.0e-6, msg="\{msg}: \{a} vs \{b} (diff \{d})")
}

///|
test "cross-validate against independent numpy reference" {
  for c in ref_cases {
    let crf = Crf::new(c.num_tags, c.start, c.end, c.trans) catch {
      e => fail(e.message())
    }
    let ca : Constraints? = match c.scheme {
      "bio" => Some(bio_mask(c.tags) catch { e => fail(e.message()) })
      "bioes" => Some(bioes_mask(c.tags) catch { e => fail(e.message()) })
      _ => None
    }
    let path = crf.decode(c.emissions, None, ca) catch {
      e => fail(e.message())
    }
    assert_eq(path, c.expected_path)
    let vp = crf.viterbi(c.emissions, None, ca) catch { e => fail(e.message()) }
    close(vp.score, c.expected_score, c.name + " score")
    let lz = crf.log_partition(c.emissions, None, ca) catch {
      e => fail(e.message())
    }
    close(lz, c.expected_logz, c.name + " logz")
    let mg = crf.marginals(c.emissions, None, ca) catch {
      e => fail(e.message())
    }
    assert_eq(mg.length(), c.expected_marginals.length())
    for t in 0..<mg.length() {
      let mut s = 0.0
      for j in 0..<c.num_tags {
        close(mg[t][j], c.expected_marginals[t][j], c.name + " marginal")
        s = s + mg[t][j]
      }
      close(s, 1.0, c.name + " row sum")
    }
    let ll = crf.log_likelihood(c.emissions, path, None, ca) catch {
      e => fail(e.message())
    }
    close(ll, c.expected_score - c.expected_logz, c.name + " ll")
    assert_true(ll <= 1.0e-9, msg="log-likelihood must be <= 0")
  }
}

///|
test "uniform model factorizes to per-position softmax" {
  let crf = Crf::uniform(3) catch { e => fail(e.message()) }
  let emit = [[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]]
  let mg = crf.marginals(emit, None, None) catch { e => fail(e.message()) }
  let path = crf.decode(emit, None, None) catch { e => fail(e.message()) }
  assert_eq(path, [1, 2])
  for t in 0..<2 {
    let z = logsumexp(emit[t])
    for j in 0..<3 {
      close(mg[t][j], @math.exp(emit[t][j] - z), "softmax marginal")
    }
  }
}

///|
test "BIO constraint forbids illegal I-after-O" {
  let tags = ["O", "B-PER", "I-PER"]
  let ca = bio_mask(tags) catch { e => fail(e.message()) }
  let crf = Crf::uniform(3) catch { e => fail(e.message()) }
  let emit = [[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]]
  let free = crf.decode(emit, None, None) catch { e => fail(e.message()) }
  assert_eq(free, [0, 2])
  let con = crf.decode(emit, None, Some(ca)) catch { e => fail(e.message()) }
  assert_true(ca.transitions[con[0]][con[1]], msg="constrained path is illegal")
  assert_true(ca.start[con[0]], msg="illegal start tag")
}

///|
test "BIOES mask rejects dangling B and illegal start" {
  let tags = ["O", "B-PER", "I-PER", "E-PER", "S-PER"]
  let ca = bioes_mask(tags) catch { e => fail(e.message()) }
  // B-PER cannot end a sequence; I-PER cannot start.
  assert_true(!ca.end[1], msg="B should not be a valid end")
  assert_true(!ca.start[2], msg="I should not be a valid start")
  assert_true(ca.transitions[1][3], msg="B->E same type allowed")
  assert_true(!ca.transitions[1][0], msg="B->O forbidden")
  assert_true(ca.transitions[3][0], msg="E->O allowed")
}

///|
test "n-best top matches viterbi and scores descend" {
  let crf = Crf::uniform(3) catch { e => fail(e.message()) }
  let emit = [[1.0, 2.0, 0.5], [0.5, 0.0, 3.0], [2.0, 1.0, 0.0]]
  let vp = crf.viterbi(emit, None, None) catch { e => fail(e.message()) }
  let nb = crf.nbest(emit, None, None, 5) catch { e => fail(e.message()) }
  assert_true(nb.length() > 0, msg="nbest empty")
  assert_eq(nb[0].tags, vp.tags)
  close(nb[0].score, vp.score, "nbest[0] == viterbi")
  for i in 1..<nb.length() {
    assert_true(nb[i - 1].score >= nb[i].score, msg="nbest not sorted")
  }
}

///|
test "mask padding equivalence" {
  let crf = Crf::uniform(2) catch { e => fail(e.message()) }
  let full = [[1.0, 0.0], [0.0, 2.0]]
  let padded = [[1.0, 0.0], [0.0, 2.0], [9.0, 9.0]]
  let mask = [true, true, false]
  let a = crf.decode(full, None, None) catch { e => fail(e.message()) }
  let b = crf.decode(padded, Some(mask), None) catch { e => fail(e.message()) }
  assert_eq(a, b)
  let la = crf.log_partition(full, None, None) catch { e => fail(e.message()) }
  let lb = crf.log_partition(padded, Some(mask), None) catch {
    e => fail(e.message())
  }
  close(la, lb, "masked logz")
}

///|
test "json roundtrip preserves model and decoding" {
  let crf = Crf::new(2, [0.1, -0.2], [0.3, 0.4], [[0.5, -0.6], [0.7, 0.8]]) catch {
    e => fail(e.message())
  }
  let back = Crf::from_json(crf.to_json()) catch { e => fail(e.message()) }
  assert_eq(crf, back)
  let emit = [[1.0, 2.0], [3.0, 0.0]]
  let d1 = crf.decode(emit, None, None) catch { e => fail(e.message()) }
  let d2 = back.decode(emit, None, None) catch { e => fail(e.message()) }
  assert_eq(d1, d2)
}

///|
test "score_path equals viterbi score on decoded path" {
  let crf = Crf::new(3, [0.2, -0.1, 0.0], [0.0, 0.1, -0.2], [
    [0.1, 0.2, -0.3],
    [0.0, -0.1, 0.4],
    [0.5, 0.0, 0.1],
  ]) catch {
    e => fail(e.message())
  }
  let emit = [[0.5, 1.0, 0.0], [0.0, 0.2, 1.5], [1.0, 0.0, 0.3]]
  let vp = crf.viterbi(emit, None, None) catch { e => fail(e.message()) }
  let sp = crf.score_path(emit, vp.tags, None) catch { e => fail(e.message()) }
  close(sp, vp.score, "score_path == viterbi")
}

///|
fn bad_start_raises() -> Bool {
  let _ = Crf::new(2, [0.0], [0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]) catch {
    _ => return true
  }
  false
}

///|
fn bad_trans_raises() -> Bool {
  let _ = Crf::new(2, [0.0, 0.0], [0.0, 0.0], [[0.0, 0.0]]) catch {
    _ => return true
  }
  false
}

///|
fn bad_tag_raises() -> Bool {
  let crf = Crf::uniform(2) catch { _ => return false }
  let _ = crf.score_path([[0.0, 0.0]], [5], None) catch { _ => return true }
  false
}

///|
test "invalid inputs raise" {
  assert_true(bad_start_raises(), msg="bad start should raise")
  assert_true(bad_trans_raises(), msg="bad trans should raise")
  assert_true(bad_tag_raises(), msg="out-of-range tag should raise")
}
