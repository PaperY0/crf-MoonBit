# 性能基准

运行方式（release 模式，wasm 后端）：

```bash
moon bench --release --target wasm
```

发射分由确定性 LCG（种子 20261125）生成，权重为均匀模型（全零），
无约束、无掩码。基准定义见 `crf_bench_wbtest.mbt`。

## 参考数据

环境：Windows 11 x64，moon 0.1.20261119 / moonc v0.10.22，wasm release。
数值为 10 轮采样的均值 ± 标准差，仅供量级参考，不同机器会有差异。

| 操作 | 规模 | 耗时（mean ± σ） |
| --- | --- | --- |
| Viterbi `decode` | K=5, T=100 | 19.27 µs ± 0.30 µs |
| Viterbi `decode` | K=20, T=200 | 320.31 µs ± 8.54 µs |
| `log_partition` | K=10, T=200 | 480.15 µs ± 4.35 µs |
| `marginals` | K=10, T=200 | 1.07 ms ± 0.02 ms |
| `nbest` n=5 | K=5, T=50 | 1.43 ms ± 0.03 ms |

## 复杂度说明

- `decode` / `log_partition`：O(T·K²)；前者每格只做 max/add，
  后者需 logsumexp（exp/ln），故同规模下更慢。
- `marginals`：前向 + 后向两遍 O(T·K²)，外加逐位置归一化。
- `nbest`：每状态保留至多 n 条部分路径，时间 O(T·K²·n)，
  内存 O(T·K·n)；长序列大 n 时注意规模上界。