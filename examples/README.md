# examples

## ner — 最小端到端示例

JSON 权重加载 → BIO 约束 → 标签名解码 → 逐位置置信度 → n-best 候选。

```bash
moon run examples/ner
```

真实项目中，权重 JSON 由 `tools/export_torchcrf.py` 从 PyTorch 训练好的
CRF `state_dict` 导出；发射分由上游编码器（BiLSTM / BERT 等）逐 token 给出。
本示例手写演示权重与发射分，便于直接跑通。