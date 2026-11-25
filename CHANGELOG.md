# Changelog

## 0.1.2 (2026-11-25)

### 加固
- 输入守卫：构造模型与推理时校验权重 / 发射分均为有限数（非 NaN / ±Inf），
  非有限值抛出 `InvalidShape`，不再静默传播成 NaN 边际。
- 掩码测试补齐非前缀（头部、中间位置）padding 的等价性断言。
- CI 新增 ubuntu native 后端 check 与测试岗位，四后端兼容性在 CI 坐实。

### 文档与质量
- README API 类型对齐 `moon info` 生成的 Int32 签名；申报书补齐提交哈希。
- moon.mod 描述与关键词收录后验解码、边际、置信度、熵等可检索词。
- 测试总数 30，wasm / wasm-gc / js 三后端全过。

## 0.1.1

### 新增
- `Crf::posterior_decode`：逐位置边际最大（MBR 准则）解码，与 Viterbi 互补。
- `Crf::validate_path`：校验路径在约束下的合法性（范围 / 起始 / 结束 / 相邻转移）。
- `Crf::confidence`：逐位置最大边际概率与 Shannon 熵（`Confidence` 结构），
  用于低置信筛查与人工复核分流。
- CLI：`--bio` / `--bioes` 逗号分隔标签约束、`--nbest` 候选条数、
  置信度与标签名输出；标签列表与模型维度校验。

### 测试与质量
- BIOES 约束参考向量 3 条（交叉验证向量增至 13 条：7 无约束 + 3 BIO + 3 BIOES）。
- 性质测试：20 个确定性伪随机模型上的不变量（score≤logZ、边际归一、
  n-best 一致性、约束路径恒合法）。
- 覆盖率补测：库代码行覆盖率 100%（`COVERAGE.md`）；测试总数 28，三后端全过。
- 参考向量量化到 10 位有效数字，跨平台逐字节可复现；CI 钉住 numpy==2.5.5。

## 0.1.0

首个版本。纯 MoonBit 线性链 CRF 推理库。

### 推理
- `Crf::decode` / `Crf::viterbi`：log 域 Viterbi 最优路径解码。
- `Crf::nbest`：精确 n-best（每状态保留至多 n 条部分路径）。
- `Crf::log_partition`：前向算法求配分函数 log Z（max-shift logsumexp）。
- `Crf::marginals`：前向–后向边际概率 P(y_t = j | x)。
- `Crf::score_path` / `Crf::log_likelihood`：路径打分与条件对数似然。
- `Crf::decode_batch`：批解码。

### 约束与变长
- `bio_mask` / `bioes_mask`：从标签名生成 BIO / BIOES 合法转移掩码。
- `Constraints`：自定义 start / end / transition 布尔掩码；`Constraints::allow_all`。
- `mask`：padding 位置不参与计算。

### 权重互通
- `Crf::from_json` / `Crf::to_json`：与 pytorch-crf `state_dict` 导出的 JSON schema 互转。
- `transitions[i][j]` = 从 i 到 j，与 pytorch-crf 一致。
- `tools/export_torchcrf.py`：从训练好的 torchcrf 导出权重 JSON。

### 正确性
- 独立 numpy 参考实现 `tools/gen_vectors.py`，10 条固定种子参考向量。
- 12 个测试（白盒交叉验证 + 不变量 + 黑盒 API），wasm / wasm-gc / js 三后端全过。
- `moon check --target all` 四后端 0 警告。

### 工程
- CLI `cmd/main`：BIO NER 演示 + `--model` / `--emit` JSON 解码。
- CI：三后端测试矩阵 + fmt / check / 接口门禁 + CLI 冒烟 + numpy 参考可复现门禁。
- MIT License，双语 README。