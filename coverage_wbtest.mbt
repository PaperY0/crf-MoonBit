// 覆盖率补测：由 `moon coverage analyze` 驱动，补齐错误分支与少用 API。
// CLI（cmd/main）不在单测覆盖内，由 CI 的三后端冒烟测试覆盖。

///|
fn[A] raises(f : () -> A raise CrfError) -> Bool {
  let _ = f() catch { _ => return true }
  false
}

///|
test "coverage: empty sequence raises across all APIs" {
  let crf = Crf::uniform(2) catch { e => fail(e.message()) }
  let empty : Array[Array[Double]] = []
  assert_true(raises(() => crf.decode(empty, None, None)))
  assert_true(raises(() => crf.viterbi(empty, None, None)))
  assert_true(raises(() => crf.nbest(empty, None, None, 2)))
  assert_true(raises(() => crf.log_partition(empty, None, None)))
  assert_true(raises(() => crf.score_path(empty, [], None)))
  assert_true(raises(() => crf.log_likelihood(empty, [], None, None)))
  assert_true(raises(() => crf.marginals(empty, None, None)))
  assert_true(raises(() => crf.posterior_decode(empty, None, None)))
  assert_true(raises(() => crf.confidence(empty, None, None)))
}

///|
test "coverage: shape validation errors" {
  assert_true(raises(() => Crf::new(0, [], [], [])))
  assert_true(raises(() => Crf::uniform(0)))
  assert_true(
    raises(() => Crf::new(2, [0.0, 0.0], [0.0], [[0.0, 0.0], [0.0, 0.0]])),
  )
  assert_true(
    raises(() => Crf::new(2, [0.0, 0.0], [0.0, 0.0], [[0.0, 0.0], [0.0]])),
  )
  let crf = Crf::uniform(2) catch { e => fail(e.message()) }
  assert_true(raises(() => crf.decode([[0.0, 0.0, 0.0]], None, None)))
  assert_true(raises(() => crf.decode([[0.0, 0.0]], Some([true, false]), None)))
  assert_true(raises(() => crf.score_path([[0.0, 0.0]], [0, 1], None)))
  assert_true(raises(() => crf.nbest([[0.0, 0.0]], None, None, 0)))
}

///|
test "coverage: impossible constraints raise" {
  let crf = Crf::uniform(2) catch { e => fail(e.message()) }
  let ca = Constraints::{
    start: [false, false],
    end: [true, true],
    transitions: [[true, true], [true, true]],
  }
  assert_true(raises(() => crf.decode([[0.0, 0.0]], None, Some(ca))))
  assert_true(raises(() => crf.nbest([[0.0, 0.0]], None, Some(ca), 2)))
  assert_true(raises(() => crf.marginals([[0.0, 0.0]], None, Some(ca))))
}

///|
test "coverage: json error branches" {
  assert_true(raises(() => Crf::from_json("not json")))
  assert_true(raises(() => Crf::from_json("[1,2]")))
  assert_true(raises(() => Crf::from_json("{}")))
  assert_true(
    raises(() => {
      Crf::from_json(
        "{\"num_tags\":\"x\",\"start_transitions\":[],\"end_transitions\":[],\"transitions\":[]}",
      )
    }),
  )
}

///|
test "coverage: tag-set validation errors" {
  assert_true(raises(() => bio_mask([])))
  assert_true(raises(() => bioes_mask([])))
  assert_true(raises(() => bio_mask(["X-PER"])))
  assert_true(raises(() => bioes_mask(["Q-PER"])))
}

///|
test "coverage: validate_path rejects illegal middle transition" {
  let ca = bio_mask(["O", "B-PER", "I-PER"]) catch { e => fail(e.message()) }
  let crf = Crf::uniform(3) catch { e => fail(e.message()) }
  // B-PER -> O 合法，O -> I-PER 非法；首尾本身都允许。
  let bad = crf.validate_path([1, 0, 2], Some(ca)) catch {
    e => fail(e.message())
  }
  assert_true(!bad, msg="middle illegal transition must be rejected")
  // I-PER 不能开头。
  let bad_start = crf.validate_path([2, 2], Some(ca)) catch {
    e => fail(e.message())
  }
  assert_true(!bad_start, msg="illegal start must be rejected")
}

///|
test "coverage: batch decode, allow_all, tag_count, message, logsumexp edges" {
  let crf = Crf::uniform(3) catch { e => fail(e.message()) }
  assert_eq(crf.tag_count(), 3)
  let ca = Constraints::allow_all(3)
  let batch = [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 2.0]]]
  let paths = crf.decode_batch(batch, None, Some(ca)) catch {
    e => fail(e.message())
  }
  assert_eq(paths, [[1, 0], [2]])
  let masks = [[true, false], [true]]
  let p2 = crf.decode_batch(batch, Some(masks), None) catch {
    e => fail(e.message())
  }
  assert_eq(p2, [[1], [2]])
  assert_true(raises(() => crf.decode_batch(batch, Some([[true]]), None)))
  let msg = CrfError::EmptySequence.message()
  assert_true(msg.length() > 0, msg="error message renders")
  assert_true(CrfError::InvalidShape("x").message().length() > 0)
  assert_true(CrfError::InvalidTags("y").message().length() > 0)
  assert_true(CrfError::InvalidConstraint("z").message().length() > 0)
  assert_true(CrfError::JsonError("j").message().length() > 0)
  close(logsumexp([]), neg_inf, "empty logsumexp")
  close(logsumexp([neg_inf, neg_inf]), neg_inf, "all -inf logsumexp")
}

///|
test "coverage: non-finite weights and emissions raise" {
  let nan = @math.ln(-1.0)
  let inf = @math.exp(1.0e308)
  assert_true(
    raises(() => Crf::new(2, [nan, 0.0], [0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]])),
  )
  assert_true(
    raises(() => Crf::new(2, [0.0, 0.0], [inf, 0.0], [[0.0, 0.0], [0.0, 0.0]])),
  )
  assert_true(
    raises(() => Crf::new(2, [0.0, 0.0], [0.0, 0.0], [[nan, 0.0], [0.0, 0.0]])),
  )
  let crf = Crf::uniform(2) catch { e => fail(e.message()) }
  assert_true(raises(() => crf.decode([[0.0, nan]], None, None)))
  assert_true(raises(() => crf.decode([[inf, 0.0]], None, None)))
}
