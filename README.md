# PaperY0/crf

> 纯 MoonBit 的**线性链条件随机场（linear-chain CRF）推理库**。给定上游编码器输出的发射分
> （emissions）与一组转移权重，做 **Viterbi 最优解码、精确 n-best、配分函数 log Z、
> 前向–后向边际概率、路径打分与条件对数似然**，并内置 **BIO / BIOES 结构约束**。
> 全程 log 域、纯计算、无 IO，权重布局与 [pytorch-crf](https://github.com/kmkurn/pytorch-crf) 互通。

[![CI](https://github.com/PaperY0/crf-MoonBit/actions/workflows/ci.yml/badge.svg)](https://github.com/PaperY0/crf-MoonBit/actions/workflows/ci.yml)
[English below](#papery0crf-en)

## 为什么是它

MoonBit 生态里**没有线性链 CRF 推理库**。三轮查重确认空白：

1. **mooncakes 全量扫描**：对本地注册表索引 3917 个模块文件、250+ 候选方向关键词做全量匹配；
2. **官方 `moon search`**（深搜 README 与包文档，比 name/keywords 更深）：`moon search crf`
   → **No modules found**；
3. **GitHub 交叉核验**。

最接近的 `ppyq882/moonnlp` 提供的是**可训练的 BMES HMM + Viterbi + 词性标注**——是隐马尔可夫
模型，不是 CRF：HMM 对观测做生成式建模、转移概率归一化；CRF 是判别式模型，直接对给定观测的
标签序列打分，转移权重无需归一化，且能接入任意上游编码器（BiLSTM / BERT / 任何特征）的发射分。
二者数学形式与适用场景不同。本库填补的是"**拿训练好的 CRF 权重，在 MoonBit 里做结构化推理**"
这一空白——尤其适合把 PyTorch 训练出的 BiLSTM-CRF / BERT-CRF 序列标注模型部署到 wasm / 边缘端。

## 能力范围

- **Viterbi 解码** `decode` / `viterbi`：最优标签路径（后者附未归一化 log 分数）
- **精确 n-best** `nbest`：每个状态保留至多 n 条最优部分路径，逐位置扩展，返回降序前 n 条
- **配分函数** `log_partition`：前向算法求 log Z（max-shift logsumexp，数值稳定）
- **边际概率** `marginals`：前向–后向算法求每个位置每个标签的 P(y_t = j | x)
- **路径打分** `score_path` 与**条件对数似然** `log_likelihood`（= score − log Z ≤ 0）
- **批解码** `decode_batch`
- **后验解码** `posterior_decode`：逐位置边际最大（MBR 准则），与 Viterbi 全局最优互补；
  `validate_path` 校验任意路径在约束下的合法性
- **置信度** `confidence`：逐位置最大边际概率 + Shannon 熵，用于低置信筛查与人工复核分流
- **结构约束** `bio_mask` / `bioes_mask`：从标签名（如 `B-PER`/`I-PER`）自动生成合法转移掩码；
  也可用 `Constraints` 自定义任意 start/end/transition 布尔掩码
- **变长掩码** `mask`：padding 位置不参与计算，等价于截断到有效长度
- **JSON 互通** `from_json` / `to_json`：与 torchcrf `state_dict` 导出的权重格式互转

## 权重布局与 JSON schema

与 pytorch-crf 一致，三个权重张量：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `start_transitions` | `[K]` | 起始转移到各标签的分数 |
| `end_transitions` | `[K]` | 各标签转移到结束的分数 |
| `transitions` | `[K][K]` | **`transitions[i][j]` = 从标签 i 转移到标签 j 的分数** |

发射分 `emissions[T][K]`（T 个位置、K 个标签）**不属于 CRF 权重**，由你的上游编码器在推理时
给出，单独传入 `decode` / `marginals` 等。本库只做结构化推理，不做神经网络前向、不做训练。

```json
{
  "num_tags": 5,
  "start_transitions": [0.0, 0.0, -10000.0, 0.0, -10000.0],
  "end_transitions":   [0.0, -10000.0, 0.0, -10000.0, 0.0],
  "transitions":       [[...], [...], [...], [...], [...]]
}
```

> **方向约定提醒**：pytorch-crf 的 `transitions[i][j]` 是 **i→j**（源码
> `self.transitions[tags[i-1], tags[i]]`），与本库一致，`tools/export_torchcrf.py` 直接导出、
> **不转置**。但 **PyTorch 官方 NLP 教程**（`advanced_tutorial.py`）用的是**转置约定**
> （`P_{j,k}` = k→j）；若你的权重来自该教程，导入前需先转置 `transitions`。

## 安装

已发布到 mooncakes（当前版本 0.2.0）：

```bash
moon add PaperY0/crf
```

在你的包的 `moon.pkg` 中导入：

```
import {
  "PaperY0/crf" @crf,
}
```

工具链要求：**moonc ≥ 0.10.16**；本仓库由 `.moonbit-version` 钉住 `0.10.22+124d61d05`，
CI 用同一版本安装，保证可复现。

## 快速开始

```moonbit nocheck
// 1) 加载权重：从 JSON（torchcrf 导出），或构造一个零权重模型
let crf = @crf.Crf::from_json(model_json) catch { e => panic() }
// 或：let crf = @crf.Crf::uniform(5) catch { e => panic() }

// 2) 可选：从标签名生成 BIO 结构约束
let tags = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
let ca = @crf.bio_mask(tags) catch { e => panic() }

// 3) 上游编码器给出的发射分（行=位置，列=标签）
let emissions = [
  [0.2, 2.0, 0.1, 0.0, 0.0],
  [0.1, 0.0, 1.8, 0.0, 0.0],
  [2.5, 0.0, 0.0, 0.1, 0.0],
  [0.0, 0.0, 0.0, 2.2, 0.1],
]

// 4) 解码与边际概率（mask=None 表示无 padding）
let path = crf.decode(emissions, None, Some(ca)) catch { e => panic() }
// path == [1, 2, 0, 3]，即 B-PER I-PER O B-LOC（合法 BIO 序列）
let marg = crf.marginals(emissions, None, Some(ca)) catch { e => panic() }
let lz = crf.log_partition(emissions, None, Some(ca)) catch { e => panic() }
let top3 = crf.nbest(emissions, None, Some(ca), 3) catch { e => panic() }
```

## API

```moonbit nocheck
// 构造
pub fn Crf::new(num_tags : Int32, start : Array[Double], end : Array[Double],
                transitions : Array[Array[Double]]) -> Crf raise CrfError
pub fn Crf::uniform(num_tags : Int32) -> Crf raise CrfError   // 全零权重
pub fn Crf::from_json(String) -> Crf raise CrfError
pub fn Crf::to_json(Crf) -> String
pub fn Crf::tag_count(Crf) -> Int32

// 推理（mask / constraints 传 None 表示不启用）
pub fn Crf::decode(Crf, emissions, mask : Array[Bool]?, ca : Constraints?) -> Array[Int32] raise CrfError
pub fn Crf::viterbi(Crf, emissions, mask?, ca?) -> DecodedPath raise CrfError
pub fn Crf::nbest(Crf, emissions, mask?, ca?, n : Int32) -> Array[DecodedPath] raise CrfError
pub fn Crf::log_partition(Crf, emissions, mask?, ca?) -> Double raise CrfError
pub fn Crf::marginals(Crf, emissions, mask?, ca?) -> Array[Array[Double]] raise CrfError
pub fn Crf::score_path(Crf, emissions, path : Array[Int32], mask?) -> Double raise CrfError
pub fn Crf::log_likelihood(Crf, emissions, path, mask?, ca?) -> Double raise CrfError
pub fn Crf::decode_batch(Crf, batch, masks?, ca?) -> Array[Array[Int32]] raise CrfError
pub fn Crf::marginals_batch(Crf, batch, masks?, ca?) -> Array[Array[Array[Double]]] raise CrfError
pub fn Crf::confidence_batch(Crf, batch, masks?, ca?) -> Array[Confidence] raise CrfError
pub fn Crf::decode_labels(Crf, tags : Array[String], emissions, mask?, ca?) -> Array[String] raise CrfError
pub fn Crf::posterior_decode(Crf, emissions, mask?, ca?) -> Array[Int32] raise CrfError
pub fn Crf::validate_path(Crf, path : Array[Int32], ca?) -> Bool raise CrfError
pub fn Crf::confidence(Crf, emissions, mask?, ca?) -> Confidence raise CrfError
pub struct Confidence { max_prob : Array[Double]; entropy : Array[Double] }

// 约束
pub struct Constraints { start : Array[Bool]; end : Array[Bool]; transitions : Array[Array[Bool]] }
pub fn Constraints::allow_all(num_tags : Int32) -> Constraints
pub fn bio_mask(tags : Array[String]) -> Constraints raise CrfError
pub fn bioes_mask(tags : Array[String]) -> Constraints raise CrfError

// 结果与工具
pub struct DecodedPath { tags : Array[Int32]; score : Double }
pub fn DecodedPath::labels(DecodedPath, tags : Array[String]) -> Array[String] raise CrfError
pub fn logsumexp(Array[Double]) -> Double
pub suberror CrfError { InvalidShape(String); EmptySequence; InvalidTags(String);
                        InvalidConstraint(String); JsonError(String) }
```

`emissions : Array[Array[Double]]`，形状 `[T][K]`。所有推理函数校验维度，非法输入 raise
`CrfError`（用 `e.message()` 取可读信息）。

## 正确性验证

不靠"看起来对"，而是**与一个独立的 numpy 参考实现交叉对拍**：

- `tools/gen_vectors.py` 用 numpy **精确复刻 torchcrf 的前向 / Viterbi / 前向–后向递推**，
  生成 **13 条参考向量**：7 条无约束（K∈{2,3,4,5}，T∈{1,2,3,5,6,8,12}）+ 3 条 BIO 约束
  （T∈{5,7,9}）+ 3 条 BIOES 约束（T∈{4,6,8}，标签 `O/B-PER/I-PER/E-PER/S-PER`），
  固定随机种子 20261125。
  所有输入/输出统一量化到 **10 位有效数字**（远小于 1e-6 断言容差），消除跨平台 libm 的
  末位 ULP 噪声，保证 Windows / Linux 逐字节可复现；CI 钉住 numpy 版本。
- `reference_vectors_wbtest.mbt` 把这些向量转成 MoonBit 白盒测试，逐条断言：
  **Viterbi 路径精确相等**、`score` / `log Z` / 每个边际概率误差 **< 1e-6**、
  边际每行和为 1、`log_likelihood ≤ 0`。
- 另有手写不变量测试：零权重模型退化为逐位置 softmax、BIO 禁止非法 I-after-O、
  n-best 首条等于 Viterbi 且降序、mask padding 等价、JSON 往返、错误路径。

**36 个测试在 wasm / wasm-gc / js 三后端全部通过**，`moon check --target all` 四后端 0 警告；
**库代码行覆盖率 100%**（见 [`COVERAGE.md`](COVERAGE.md)，CLI 由 CI 冒烟覆盖）；
性能基准见 [`BENCH.md`](BENCH.md)（`moon bench --release`）。

```bash
moon test --target wasm     # 36/36
moon test --target wasm-gc  # 36/36
moon test --target js       # 36/36
moon bench --release        # 性能基准
python tools/gen_vectors.py # 重新生成参考向量（CI 校验逐字节可复现）
```

> 本机若装有 torch，可用 `tools/export_torchcrf.py state_dict.pt model.json` 导出**真实训练权重**
> 再做端到端验证；参考向量本身由独立 numpy 实现生成，不依赖 torch。
> CI 另有手动触发的 `torchcrf-diff` workflow：在 ubuntu 安装 CPU 版 torch + pytorch-crf，
> 用 torchcrf 自身的 decode/score/normalizer（float64）生成 5 条差分向量
> （权重经导出脚本往返），与 MoonBit 库对拍，最近一次运行 41/41 全过。

## 示例

```bash
moon run examples/ner   # JSON 权重 → BIO 约束 → 标签名解码 → 置信度 → n-best
```

详见 [`examples/README.md`](examples/README.md)。

## CLI

```bash
moon run cmd/main                 # 内置 BIO 命名实体识别演示
moon run cmd/main -- --model "<crf-json>" --emit "[[1.0,0.0],[0.0,2.0]]"
moon run cmd/main -- --model "<crf-json>" --emit "<json>" --bio "O,B-PER,I-PER" --nbest 5
moon run cmd/main -- --help
```

`--bio` / `--bioes` 接逗号分隔标签名（互斥），`--nbest` 指定候选条数；
输出含解码路径、标签名、score / log Z / log L、逐位置边际与置信度（max_prob / 熵）、n-best 候选。

演示输出（节选）：

```
== BIO-constrained NER demo ==
decoded path : 1 2 0 3
tag names    : B-PER I-PER O B-LOC
path score   : 8.5
log Z        : 9.461046665438056
log L        : -0.9610466654380563
top-3 paths:
  [1 2 0 3]  score=8.5
  [1 0 0 3]  score=6.8
  [1 1 0 3]  score=6.7
```

## 三个使用场景

1. **边缘端 / 浏览器序列标注部署**：PyTorch 训练 BiLSTM-CRF / BERT-CRF 命名实体识别模型，
   `tools/export_torchcrf.py` 导出权重 JSON，MoonBit 编到 wasm / wasm-gc / js 做推理——
   编码器输出发射分，本库负责结构化决策，BIO/BIOES 约束保证输出合法（不会出现 "I-PER 跟在 O 后"）。
2. **RAG / 信息抽取的幻觉筛查**：对抽取出的标签序列用 `marginals` + `confidence` 计算逐位置
   后验置信度与熵，低 max_prob / 高熵的位置标记为"模型不确定"，触发人工复核或拒答，
   而不是把低置信抽取结果直接写进知识库。
3. **候选重排与人工分流**：`nbest` 给出精确的前 n 条合法路径及分数，供下游重排器
   （规则 / 小模型 / 人工）在多个候选间选择；`posterior_decode` 提供 MBR 准则的第二意见，
   `validate_path` 对外部系统产生的路径做合法性把关。

## 非目标（明确不做）

神经网络前向（BiLSTM / BERT 编码器）、CRF 训练 / 梯度、高阶 / 树状 / 跳过链 CRF、
半马尔可夫 CRF。本库是**推理-only** 的线性链 CRF：拿现成权重做解码与概率计算。

## 开发

工具链版本固定在 [`.moonbit-version`](.moonbit-version)：

```bash
curl -fsSL https://cli.moonbitlang.com/install/unix.sh | bash -s -- "$(cat .moonbit-version)"
moon check          # 静态检查（--target all 覆盖四后端）
moon info           # 更新 .mbti 接口
moon fmt            # 格式化
moon test           # 测试
moon run cmd/main   # CLI 演示
```

> **native 后端说明**：本库是纯 MoonBit、无后端特定代码，`moon check --target all` 在
> native 上同样通过。native 的**测试执行**需要可用的 C 工具链来编译 MoonBit runtime；
> 在 Windows + msys2(ucrt64) 上，需把 `C:\msys64\ucrt64\bin` 加入 PATH，且当前 msys2 头文件
> 缺少 `SO_REUSE_MULTICASTPORT` 会导致 runtime 的 `socket.c` 编译失败——这是工具链问题，
> 与本库无关。CI 在 ubuntu 上运行 wasm / wasm-gc / js 三后端测试矩阵，
> 并单独为 native 后端跑 check 与测试。

## 许可

MIT，见 [`LICENSE`](LICENSE)。权重 JSON schema 与 pytorch-crf（MIT）互通；参考实现独立编写。

---

<a id="papery0crf-en"></a>

## PaperY0/crf (EN)

A pure-MoonBit **linear-chain Conditional Random Field (CRF) inference library**. Given emission
scores from your upstream encoder and a set of transition weights, it performs **Viterbi decoding,
exact n-best, the log partition function, forward–backward marginals, path scoring and conditional
log-likelihood**, with built-in **BIO / BIOES structural constraints**. All computation is in the
log domain, pure and IO-free; the weight layout interoperates with
[pytorch-crf](https://github.com/kmkurn/pytorch-crf).

**Why it exists.** Three-round deduplication (full scan of 3917 mooncakes registry modules,
official `moon search crf` → *No modules found*, GitHub cross-check) confirms there is no
linear-chain CRF inference library in the MoonBit ecosystem. The closest package,
`ppyq882/moonnlp`, ships a trainable BMES **HMM** + Viterbi + POS tagger — a generative model with
normalized transitions, not a discriminative CRF that scores tag sequences given arbitrary
encoder emissions. This library fills the "run a trained CRF's structured inference in MoonBit"
gap, ideal for deploying PyTorch BiLSTM-CRF / BERT-CRF taggers to wasm / edge targets.

**Install** (published on mooncakes; requires moonc ≥ 0.10.16, pinned to 0.10.22 via `.moonbit-version`):

```bash
moon add PaperY0/crf
```

```moonbit nocheck
let crf = @crf.Crf::from_json(model_json) catch { e => panic() }
let ca = @crf.bio_mask(["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]) catch { e => panic() }
let path = crf.decode(emissions, None, Some(ca)) catch { e => panic() }
let marg = crf.marginals(emissions, None, Some(ca)) catch { e => panic() }
```

**Weight layout** matches pytorch-crf: `start_transitions[K]`, `end_transitions[K]`,
`transitions[K][K]` where `transitions[i][j]` is the score **from tag i to tag j**
(source: `self.transitions[tags[i-1], tags[i]]`). `tools/export_torchcrf.py` exports without
transposing. Note the PyTorch official NLP tutorial uses the *transposed* convention; transpose
first if your weights come from it. Emissions `[T][K]` are supplied by your encoder, not stored
in the CRF weights. This library does inference only — no neural forward pass, no training.

**Correctness** is cross-validated against an independent numpy reference that replicates
torchcrf's forward / Viterbi / forward–backward recursions: 13 fixed-seed vectors (7 unconstrained
+ 3 BIO + 3 BIOES) assert exact Viterbi paths and score / log Z / marginals within 1e-6.
Beyond Viterbi, the library offers posterior (marginal-argmax) decoding, path validation,
per-position confidence (max marginal probability and Shannon entropy), and exact n-best.
**36 tests pass on wasm, wasm-gc, js, and native (ubuntu CI)** with 100% library line coverage;
`moon check --target all` is warning-free on all four backends. Local native test execution needs a working C toolchain for the MoonBit runtime (a Windows
msys2 header issue is unrelated to this library); CI also covers native on ubuntu.

MIT licensed. See the Chinese section above for the full API, CLI usage and development notes.