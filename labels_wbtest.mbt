// 标签名接口的白盒测试。

///|
test "decode_labels returns names consistent with decode" {
  let tags = ["O", "B-PER", "I-PER"]
  let crf = Crf::uniform(3) catch { e => fail(e.message()) }
  let ca = bio_mask(tags) catch { e => fail(e.message()) }
  let emit = [[0.0, 2.0, 0.0], [0.0, 0.0, 1.5], [2.0, 0.0, 0.0]]
  let names = crf.decode_labels(tags, emit, None, Some(ca)) catch {
    e => fail(e.message())
  }
  let path = crf.decode(emit, None, Some(ca)) catch { e => fail(e.message()) }
  assert_eq(names.length(), path.length())
  for i in 0..<path.length() {
    assert_eq(names[i], tags[path[i]])
  }
  assert_eq(names, ["B-PER", "I-PER", "O"])
}

///|
test "DecodedPath::labels maps n-best paths" {
  let tags = ["O", "B-PER", "I-PER"]
  let crf = Crf::uniform(3) catch { e => fail(e.message()) }
  let emit = [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0]]
  let nb = crf.nbest(emit, None, None, 3) catch { e => fail(e.message()) }
  for p in nb {
    let names = p.labels(tags) catch { e => fail(e.message()) }
    assert_eq(names.length(), p.tags.length())
    for i in 0..<p.tags.length() {
      assert_eq(names[i], tags[p.tags[i]])
    }
  }
}

///|
test "label APIs reject wrong table size and out-of-range index" {
  let crf = Crf::uniform(3) catch { e => fail(e.message()) }
  let emit = [[1.0, 0.0, 0.0]]
  assert_true(raises(() => crf.decode_labels(["O"], emit, None, None)))
  let bad : DecodedPath = { tags: [5], score: 0.0, }
  assert_true(raises(() => bad.labels(["O", "B-PER", "I-PER"])))
}
