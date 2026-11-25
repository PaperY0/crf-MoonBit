// posterior_decode / validate_path 的白盒测试。

///|
test "posterior_decode picks per-position marginal argmax" {
  let crf = Crf::uniform(3) catch { e => fail(e.message()) }
  let emit = [[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]]
  let p = crf.posterior_decode(emit, None, None) catch {
    e => fail(e.message())
  }
  assert_eq(p, [1, 2])
}

///|
test "posterior path can be illegal; validate_path detects it" {
  let ca = bio_mask(["O", "B-PER", "I-PER"]) catch { e => fail(e.message()) }
  let crf = Crf::uniform(3) catch { e => fail(e.message()) }
  let emit = [[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]]
  // 无约束后验：O 后接 I-PER（非法）。
  let p = crf.posterior_decode(emit, None, None) catch {
    e => fail(e.message())
  }
  assert_eq(p, [0, 2])
  let legal = crf.validate_path(p, Some(ca)) catch { e => fail(e.message()) }
  assert_true(!legal, msg="O -> I-PER should be illegal")
  // Viterbi 在约束下保证合法。
  let v = crf.decode(emit, None, Some(ca)) catch { e => fail(e.message()) }
  let ok = crf.validate_path(v, Some(ca)) catch { e => fail(e.message()) }
  assert_true(ok, msg="constrained viterbi path must be legal")
}

///|
test "validate_path checks start/end and range" {
  let ca = bioes_mask(["O", "B-PER", "I-PER", "E-PER", "S-PER"]) catch {
    e => fail(e.message())
  }
  let crf = Crf::uniform(5) catch { e => fail(e.message()) }
  // B-PER 不能收尾。
  let dangling = crf.validate_path([0, 1], Some(ca)) catch {
    e => fail(e.message())
  }
  assert_true(!dangling, msg="B at end should be illegal")
  // S-PER 单标签合法。
  let single = crf.validate_path([4], Some(ca)) catch { e => fail(e.message()) }
  assert_true(single, msg="S alone should be legal")
  // 无约束时任意范围内路径合法。
  let free = crf.validate_path([2, 2, 2], None) catch { e => fail(e.message()) }
  assert_true(free)
}

///|
fn validate_empty_raises() -> Bool {
  let crf = Crf::uniform(2) catch { _ => return false }
  let _ = crf.validate_path([], None) catch { _ => return true }
  false
}

///|
fn validate_oob_raises() -> Bool {
  let crf = Crf::uniform(2) catch { _ => return false }
  let _ = crf.validate_path([7], None) catch { _ => return true }
  false
}

///|
test "validate_path rejects empty and out-of-range" {
  assert_true(validate_empty_raises(), msg="empty path should raise")
  assert_true(validate_oob_raises(), msg="out-of-range tag should raise")
}
