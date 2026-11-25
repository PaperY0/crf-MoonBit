# PaperY0/crf

纯 MoonBit 的**线性链 CRF 推理库**：Viterbi 解码、精确 n-best、配分函数 log Z、前向–后向边际概率、
路径打分与条件对数似然，内置 BIO / BIOES 结构约束。全程 log 域、纯计算无 IO，权重布局与
[pytorch-crf](https://github.com/kmkurn/pytorch-crf) 互通。

A pure-MoonBit **linear-chain CRF inference library**: Viterbi decoding, exact n-best, log partition
function, forward–backward marginals, path scoring and conditional log-likelihood, with built-in
BIO / BIOES constraints. Log-domain, pure computation, no IO; weight layout interoperates with
pytorch-crf.

## 生态空白 / Why

三轮查重（mooncakes 3917 模块全量扫描 + 官方 `moon search crf` → No modules found + GitHub）
确认 MoonBit 生态没有线性链 CRF 推理库。最接近的 `ppyq882/moonnlp` 是可训练 BMES **HMM**，
非判别式 CRF。本库填补"拿训练好的 CRF 权重在 MoonBit 做结构化推理"的空白。

## 用法 / Usage

```moonbit nocheck
///|
let crf = @crf.Crf::from_json(model_json) catch { e => panic() }

///|
let ca = @crf.bio_mask(["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]) catch {
  e => panic()
}
// emissions : Array[Array[Double]]，形状 [T][K]，由上游编码器给出

///|
let path = crf.decode(emissions, None, Some(ca)) catch { e => panic() }

///|
let marg = crf.marginals(emissions, None, Some(ca)) catch { e => panic() }

///|
let lz = crf.log_partition(emissions, None, Some(ca)) catch { e => panic() }

///|
let top3 = crf.nbest(emissions, None, Some(ca), 3) catch { e => panic() }
```

权重 JSON：`{ "num_tags", "start_transitions"[K], "end_transitions"[K], "transitions"[K][K] }`，
其中 `transitions[i][j]` = 从 i 到 j（与 pytorch-crf 一致，导出无需转置）。

## 核心 API / Core API

- `Crf::new` / `Crf::uniform` / `Crf::from_json` / `Crf::to_json`
- `Crf::decode` / `Crf::viterbi` / `Crf::nbest`
- `Crf::posterior_decode` / `Crf::validate_path` / `Crf::confidence`
- `Crf::log_partition` / `Crf::marginals`
- `Crf::score_path` / `Crf::log_likelihood` / `Crf::decode_batch`
- `bio_mask` / `bioes_mask` / `Constraints::allow_all`
- `logsumexp`

## 正确性 / Correctness

与独立 numpy 参考实现（精确复刻 torchcrf 递推）交叉对拍：13 条固定种子向量（7 无约束 + 3 BIO + 3 BIOES），
Viterbi 路径精确相等、score / log Z / 边际误差 < 1e-6。**30 个测试在 wasm / wasm-gc / js / native 四后端全过**，
库代码行覆盖率 100%，`moon check --target all` 四后端 0 警告。

非目标：神经网络前向、CRF 训练、高阶 / 树状 CRF。推理-only。

MIT License. 完整文档见 [README.md](README.md)。