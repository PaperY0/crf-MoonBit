# 项目申报书：crf-MoonBit

## 1. 项目名称

crf-MoonBit（MoonBit 模块名 `PaperY0/crf`，MIT 许可证，当前版本 0.2.0）。

## 2. 项目简介

纯 MoonBit 实现的线性链条件随机场（CRF）推理库。上游编码器（BiLSTM / BERT 等）给出发射分后，本库负责结构化决策：Viterbi 解码、精确 n-best、配分函数 log Z、前向–后向边际概率、后验解码、逐位置置信度（最大后验概率 + Shannon 熵）、路径打分与条件似然，内置 BIO / BIOES 约束。全程 log 域计算、纯函数无 IO，权重 JSON 与 pytorch-crf 互通——PyTorch 训练、MoonBit 推理，部署到 wasm / wasm-gc / js / native 四后端，覆盖浏览器、边缘设备与服务端。

## 3. 项目方向与通用性

方向是序列标注的推理基础设施，不绑定具体任务、模型或语言：只认"发射分矩阵 + 转移权重"，命名实体识别、词性标注、事件抽取、语义槽填充等线性链问题都适用；约束掩码由标签名自动生成，也支持自定义 start/end/transition 掩码。纯 MoonBit、无后端特定代码，四后端 CI 全绿。

生态空白经三轮查重确认：mooncakes 注册表 3917 个模块全量关键词扫描、官方 `moon search crf`（返回 No modules found）、GitHub 交叉核验，生态内没有通用线性链 CRF 推理库；最接近的 moonnlp 提供的是可训练 HMM，与判别式 CRF 数学形式不同。

## 4. 预期使用场景

**场景一：浏览器 / 边缘端命名实体识别。** 隐私敏感文本（医疗、本地文档）不能上云。PyTorch 训练 BiLSTM-CRF，用 `export_torchcrf.py` 导出权重，MoonBit 编 wasm 在浏览器内 `from_json` 加载、`bio_mask` 生成约束、`decode` 输出合法 BIO 实体序列，文本不出设备。

**场景二：RAG 知识抽取的幻觉筛查。** 知识入库前用 `marginals` / `confidence` 计算逐位置后验与熵，低 max_prob、高熵的位置标记为不确定，`log_likelihood` 对整条路径设阈值拒收，低置信结果转人工复核而非直接写进知识库。

**场景三：事件抽取的候选重排与人工分流。** `bioes_mask` 约束下用 `nbest` 取前 5 条合法路径供下游重排器选择，`posterior_decode` 给出 MBR 准则的第二意见，与 Viterbi 分歧大的样本分流给人工标注。

## 5. 拟实现的核心功能

已实现（v0.2.0，均有测试与参考向量支撑）：Viterbi / n-best / log Z / 边际 / 后验解码 / 置信度 / 路径校验 / 打分 / 似然；批量解码、批量边际与置信度；标签名接口；BIO / BIOES 与自定义约束；torchcrf 兼容 JSON I/O 与导出脚本；CLI；可运行示例 `examples/ner`；`moon bench` 性能基准；真实 pytorch-crf 差分对拍 CI。

后续规划（v0.3）：BILOU 等更多标注体系掩码、大批量推理缓冲复用、浏览器可视化 demo 页。明确不做：神经网络前向、CRF 训练 / 梯度、高阶 / 树状 CRF（推理-only 定位）。

## 6. 原创性声明

原创项目，非移植。约 1000 行库代码与 1200 行测试独立编写，未复制任何现有项目源码；实现的是公有领域的经典线性链 CRF 算法（Lafferty, McCallum & Pereira, 2001）。与 pytorch-crf 仅做权重布局与递推语义的互操作对齐，便于导入训练好的参数，二者在语言、范围、结构上均不同；交叉验证用的 numpy 参考实现亦独立编写，不与库共享源码。

## 7. 参考项目与许可证

非移植项目，仍透明披露设计参考：pytorch-crf（https://github.com/kmkurn/pytorch-crf ，MIT，参考权重布局与 forward / Viterbi 递推语义，已核对 0.7.2 源码）；CRF 原始论文（Lafferty et al., ICML 2001）。本项目以 MIT 发布，LICENSE 见仓库根目录。

## 8. GitHub 仓库与有效提交

仓库：https://github.com/PaperY0/crf-MoonBit （公开，CI 全绿；mooncakes 已发布 `PaperY0/crf@0.2.0`）。

共 32 个有效提交，无空提交、无意义拆分或重复提交，完整清单见仓库 git 历史。代表性提交：

- `edf3696` 初始版本：核心推理库 + 约束 + I/O + CLI + CI
- `1832b07` 后验解码 + 约束路径校验
- `bc086d6` 逐位置置信度（最大概率 + 熵）
- `afe6cc9` 性质测试：20 个伪随机模型上的不变量
- `c020c19` NaN / Inf 非有限输入守卫
- `3f9cae6` 批量边际 / 置信度 API
- `d9863f6` 标签名解码接口
- `e19fcd9` 端到端示例 + 四后端冒烟
- `ef66962` 性能基准 + BENCH.md
- `0e9ea45` 真实 pytorch-crf 差分对拍 CI
- `2f8ee18` CLI 冒烟补全 + 每后端覆盖率门禁
- `b2434d9` 发版 v0.2.0

质量证据：36 个测试 × wasm / wasm-gc / js / native 四后端全过；13 条独立 numpy 参考向量交叉验证（路径精确相等、数值误差 <1e-6）；库代码行覆盖率三后端均 100%（CI 每后端设门禁）；torchcrf 差分对拍 41/41 全过。