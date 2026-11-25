# 项目申报书：crf-MoonBit

## 1. 项目名称

**crf-MoonBit**（MoonBit 模块名：`PaperY0/crf`，当前版本 0.1.1，MIT 许可证）

## 2. 项目简介

crf-MoonBit 是一个**纯 MoonBit 实现的线性链条件随机场（linear-chain CRF）推理库**。
给定上游编码器（BiLSTM / BERT / 任意特征模型）输出的发射分与一组转移权重，提供：
Viterbi 最优路径解码、精确 n-best 候选、配分函数 log Z、前向–后向边际概率、
后验（MBR）解码、路径合法性校验、逐位置置信度（最大后验概率 + Shannon 熵）、
路径打分与条件对数似然，并内置 BIO / BIOES 结构约束。

全程 log 域计算、纯函数式无 IO，权重 JSON 与 PyTorch 生态的 pytorch-crf 互通：
**训练在 PyTorch 完成，推理部署到 MoonBit 的 wasm / wasm-gc / js / native 四后端**，
覆盖浏览器、边缘设备与服务端。正确性由独立 numpy 参考实现的 13 条交叉验证向量、
20 个伪随机模型的性质测试与 100% 库代码行覆盖率共同保证（28 个测试三后端全过）。

## 3. 项目方向与通用性说明

**方向**：结构化预测（structured prediction）/ 序列标注的**推理基础设施**。

**通用性**体现在四个层面：

- **任务无关**：库只认"发射分矩阵 + 转移权重"，不绑定任何具体任务。命名实体识别、
  词性标注、组块分析、事件抽取（论元角色标注）、语义槽填充、生物序列标注等
  一切可建模为线性链标注的问题都直接适用。
- **模型无关**：发射分来自任意上游编码器——BiLSTM、BERT、CNN、甚至手工特征线性模型。
  本库不做神经网络前向，与上游架构完全解耦。
- **语言无关**：约束掩码由标签名自动生成（BIO/BIOES），中文、英文或任何语言的
  标签体系均可；也支持完全自定义的 start/end/transition 布尔掩码。
- **平台无关**：纯 MoonBit、无 IO、无后端特定代码，`moon check --target all`
  四后端零警告，wasm / wasm-gc / js 三后端测试矩阵在 CI 全绿。

**生态定位（三轮查重确认的空白）**：对 mooncakes 注册表 3917 个模块全量关键词扫描、
官方 `moon search crf` 实时深搜（返回 No modules found）、GitHub 交叉核验，
确认生态内没有线性链 CRF 推理库。最接近的 `ppyq882/moonnlp` 提供的是可训练 BMES
**HMM**（生成式、转移归一化），与判别式 CRF 数学形式不同；jieba/pinyin/moonbio 等
内部虽用 Viterbi，但均绑定各自领域，不提供通用 CRF 推理。本项目填补
"拿训练好的 CRF 权重在 MoonBit 做结构化推理"这一空白。

## 4. 预期使用场景（3 个完整场景）

### 场景一：浏览器 / 边缘端命名实体识别部署

- **背景**：隐私敏感场景（医疗文本、本地文档）不能把文本发到云端，需要在浏览器
  或边缘设备本地完成 NER。
- **输入**：PyTorch 训练好的 BiLSTM-CRF / BERT-CRF 模型；待标注文本的编码器发射分。
- **流程**：
  1. 训练侧运行 `python tools/export_torchcrf.py state_dict.pt model.json` 导出 CRF 权重；
  2. 部署侧 `Crf::from_json` 加载权重，`bio_mask(["O","B-PER","I-PER",...])` 生成约束；
  3. 上游编码器（如 ONNX/Transformers.js）逐 token 产出发射分 `[T][K]`；
  4. `crf.decode(emissions, mask, Some(ca))` 得到合法 BIO 标签序列。
- **输出**：实体边界与类型（如 `B-PER I-PER O B-LOC`），约束保证不出现
  "I-PER 跟在 O 后"这类非法标注；MoonBit 编 wasm 在浏览器内运行，文本不出设备。
- **用到的 API**：`from_json`、`bio_mask`、`decode`、`decode_batch`（变长批）。

### 场景二：RAG 知识抽取的幻觉筛查

- **背景**：从检索文档抽取结构化三元组写入知识库时，低置信抽取结果是幻觉的主要来源，
  需要在入库前量化每个位置的不确定性。
- **输入**：抽取模型的发射分；待评估的标签路径。
- **流程**：
  1. `crf.marginals(emissions, None, None)` 计算逐位置后验分布；
  2. `crf.confidence(...)` 得到每个位置的 `max_prob` 与 Shannon 熵；
  3. 设定阈值（如 max_prob < 0.6 或熵 > 0.8·ln K）标记"模型不确定"位置；
  4. `crf.log_likelihood(emissions, path, ...)` 对整条路径给出条件对数似然，
     低于阈值的路径整体拒收。
- **输出**：高置信结果自动入库；低置信位置触发人工复核或拒答，而不是把
  猜测写进知识库。`validate_path` 同时把关外部系统产生的路径合法性。
- **用到的 API**：`marginals`、`confidence`、`log_likelihood`、`validate_path`。

### 场景三：事件抽取的候选重排与人工分流

- **背景**：事件论元标注中，单一最优路径常因局部歧义出错；下游有规则或小模型
  重排器，需要多个高质量候选而非一个答案。
- **输入**：事件抽取编码器的发射分；BIOES 标签体系。
- **流程**：
  1. `bioes_mask(tags)` 生成 B/I/E/S/O 约束（禁止悬空 B、非法起始 I）；
  2. `crf.nbest(emissions, None, Some(ca), 5)` 取精确前 5 条合法路径及分数；
  3. 重排器在候选间选择；`crf.posterior_decode(...)` 提供 MBR 准则的第二意见，
     与 Viterbi 结果不一致的位置提示歧义；
  4. 分歧大的样本分流给人工标注。
- **输出**：重排后的最终论元标注 + 歧义位置清单 + 人工复核队列。
- **用到的 API**：`bioes_mask`、`nbest`、`posterior_decode`、`viterbi`、`score_path`。

## 5. 拟实现的核心功能

**已实现（v0.1.1，全部有测试与参考向量支撑）**：

1. log 域 Viterbi 解码（`decode` / `viterbi`），支持任意位置 padding mask；
2. 精确 n-best（每状态保留至多 n 条部分路径，非近似 beam）；
3. 前向算法配分函数 `log_partition`（max-shift logsumexp，数值稳定）；
4. 前向–后向边际概率 `marginals`（每行和为 1，交叉验证误差 <1e-6）；
5. 后验解码 `posterior_decode` 与路径校验 `validate_path`；
6. 逐位置置信度 `confidence`（max_prob + 熵）；
7. 路径打分 `score_path`、条件对数似然 `log_likelihood`、批解码 `decode_batch`；
8. BIO / BIOES 约束掩码自动生成 + 任意自定义 `Constraints`；
9. torchcrf 兼容权重 JSON I/O（`from_json` / `to_json`）与导出脚本；
10. CLI（演示 / JSON 解码 / 约束选项 / n-best / 置信度输出）。

**后续规划（v0.2 路线）**：

- BMES / BIOES 之外的更多标注体系掩码（如 BILOU）；
- 性能基准与大批量推理优化（缓冲复用）；
- 浏览器可视化 demo 页（js 后端，输入发射分热图实时解码）；
- 与真实 torchcrf 训练权重的端到端差分对拍（CI 可选 job）。

**明确不做**：神经网络前向、CRF 训练/梯度、高阶/树状/半马尔可夫 CRF（推理-only 定位）。

## 6. 原创性声明

**本项目为原创项目**，不是移植项目：

- 全部 MoonBit 代码（约 1100 行库代码 + 700 行测试）为本项目独立编写，
  未复制任何现有项目的源码；
- 实现的算法是经典线性链 CRF 推理（Lafferty, McCallum & Pereira, 2001），
  属于公有领域的标准算法，不存在对某一实现的代码依赖；
- 与 pytorch-crf 的关系是**权重布局与递推语义的互操作对齐**（便于导入训练好的参数），
  而非代码移植：pytorch-crf 是 PyTorch 训练模块（含 NN 前向与梯度），
  本项目是纯推理库，二者代码语言、范围、结构均不同；
- 交叉验证用的 numpy 参考实现（`tools/gen_vectors.py`）亦为独立编写，
  与 MoonBit 库不共享任何源码，保证验证的独立性。

## 7. 参考项目与许可证（透明披露）

虽非移植，仍列出设计参考：

| 参考 | 来源 | 许可证 | 参考内容 |
| --- | --- | --- | --- |
| pytorch-crf | https://github.com/kmkurn/pytorch-crf | MIT | 权重布局（start/end/transitions）、`transitions[i][j]`=i→j 方向约定、forward/Viterbi 递推语义（已核对 0.7.2 源码） |
| CRF 原始论文 | Lafferty et al., ICML 2001 | 学术文献 | 线性链 CRF 模型定义 |
| PyTorch NLP 教程 | https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html | BSD-3 | 对照其转置约定并在 README 中提醒差异 |

本项目以 **MIT** 发布，LICENSE 见仓库根目录。

## 8. GitHub 仓库与有效提交

**仓库**：https://github.com/PaperY0/crf-MoonBit （公开，CI 全绿）

共 **11 个有效提交**，每个都是完整的功能/测试/文档/发版单元，无空提交、
无意义拆分或重复提交：

| # | 提交 | 内容 |
| --- | --- | --- |
| 1 | `edf3696` | 初始版本 v0.1.0：核心推理库（Viterbi/n-best/logZ/边际/打分/似然）、BIO/BIOES 约束、torchcrf I/O、CLI、双语 README、CI、MIT |
| 2 | `7ce00f0` | 参考向量跨平台稳定化：量化到 10 位有效数字，CI 钉住 numpy 版本 |
| 3 | `1832b07` | 新功能：后验（边际最大）解码 + 约束路径合法性校验（含测试） |
| 4 | `bc086d6` | 新功能：逐位置置信度（最大边际概率 + Shannon 熵，含测试） |
| 5 | `681c472` | BIOES 约束交叉验证：生成器扩展 + 3 条新参考向量 |
| 6 | `afe6cc9` | 性质测试：20 个确定性伪随机模型上的跨模型不变量 |
| 7 | `1940b61` | 覆盖率补测：补齐全部错误分支与批处理 API，库代码行覆盖率 100%，COVERAGE.md |
| 8 | `2362246` | CLI 增强：--bio/--bioes/--nbest、置信度与标签名输出、CI 冒烟扩展 |
| 9 | `a4dd9f4` | 文档：新 API、三个使用场景、28 测试与覆盖率说明 |
| 10 | `e7899fa` | 发版 v0.1.1：版本号 + CHANGELOG |
| 11 | `21fd476` | 项目申报书 APPLICATION.md |

**质量证据**：28 个测试 × wasm/wasm-gc/js 三后端全过；`moon check --target all`
四后端 0 警告；13 条独立 numpy 参考向量交叉验证（路径精确相等、数值误差 <1e-6）；
库代码行覆盖率 100%；CI 含 fmt/check/接口/参考可复现四道门禁。
mooncakes 已发布：https://mooncakes.io/docs/PaperY0/crf （`moon add PaperY0/crf`）。