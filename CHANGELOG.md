# Changelog

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