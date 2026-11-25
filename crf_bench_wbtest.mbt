// 性能基准：moon bench --release 运行。
// 发射分由确定性 LCG 生成，规模覆盖短序列到长序列、小标签集到中等标签集。

///|
fn bench_emissions(k : Int32, t : Int32) -> Array[Array[Double]] {
  let mut seed : Int64 = 20261125
  Array::makei(t, _ => {
    Array::makei(k, _ => {
      seed = seed * 48271 % 2147483647
      (seed % 1000).to_double() / 500.0 - 1.0
    })
  })
}

///|
test (b : @bench.T) {
  let crf = Crf::uniform(5) catch { e => fail(e.message()) }
  let emit = bench_emissions(5, 100)
  b.bench(name="decode K=5 T=100", fn() {
    b.keep(crf.decode(emit, None, None) catch { _ => [] })
  })
}

///|
test (b : @bench.T) {
  let crf = Crf::uniform(20) catch { e => fail(e.message()) }
  let emit = bench_emissions(20, 200)
  b.bench(name="decode K=20 T=200", fn() {
    b.keep(crf.decode(emit, None, None) catch { _ => [] })
  })
}

///|
test (b : @bench.T) {
  let crf = Crf::uniform(10) catch { e => fail(e.message()) }
  let emit = bench_emissions(10, 200)
  b.bench(name="log_partition K=10 T=200", fn() {
    b.keep(crf.log_partition(emit, None, None) catch { _ => 0.0 })
  })
  b.bench(name="marginals K=10 T=200", fn() {
    b.keep(crf.marginals(emit, None, None) catch { _ => [] })
  })
}

///|
test (b : @bench.T) {
  let crf = Crf::uniform(5) catch { e => fail(e.message()) }
  let emit = bench_emissions(5, 50)
  b.bench(name="nbest n=5 K=5 T=50", fn() {
    b.keep(crf.nbest(emit, None, None, 5) catch { _ => [] })
  })
}
