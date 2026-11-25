// 黑盒测试：仅通过公开 API（@crf 限定）冒烟验证。

///|
test "blackbox: uniform model decodes per-position argmax" {
  let crf = @crf.Crf::uniform(2) catch { e => fail(e.message()) }
  let path = crf.decode([[0.0, 1.0], [2.0, 0.0]], None, None) catch {
    e => fail(e.message())
  }
  assert_eq(path, [1, 0])
}

///|
test "blackbox: bio_mask constrains decoding" {
  let ca = @crf.bio_mask(["O", "B-X", "I-X"]) catch { e => fail(e.message()) }
  let crf = @crf.Crf::uniform(3) catch { e => fail(e.message()) }
  let path = crf.decode([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]], None, Some(ca)) catch {
    e => fail(e.message())
  }
  assert_true(ca.transitions[path[0]][path[1]], msg="illegal transition")
}

///|
test "blackbox: json roundtrip via public API" {
  let crf = @crf.Crf::new(2, [0.0, 0.0], [0.0, 0.0], [[0.0, 1.0], [1.0, 0.0]]) catch {
    e => fail(e.message())
  }
  let back = @crf.Crf::from_json(crf.to_json()) catch { e => fail(e.message()) }
  assert_eq(crf, back)
}
