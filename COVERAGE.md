# 覆盖率报告

生成方式：`moon coverage analyze > uncovered.log`（MoonBit 行覆盖率）。

## 结论

- **库代码（10 个 `.mbt` 源文件）：行覆盖率 100%**，无未覆盖行；
  该结论在 **wasm / wasm-gc / js 三后端分别验证**（本地各后端
  `moon test --enable-coverage` + `moon coverage analyze` 均为库代码零未覆盖），
  CI 对每个后端设覆盖率门禁：根目录 `.mbt` 出现未覆盖行即失败。
- `cmd/main/main.mbt`（CLI，56 行）与 `examples/ner/main.mbt`（示例，9 行）
  不在单元测试覆盖内：二者由 CI 的四后端冒烟测试覆盖
  （无参演示、`--model/--emit` JSON 模式、`--bio` 约束、非法选项非零退出、示例实体断言）。
- 行覆盖率 100% 不证明算法无遗漏；数值正确性由独立 numpy 参考的
  13 条交叉验证向量与性质测试保证（见 README「正确性验证」）。

## 覆盖构成（36 个测试）

| 层 | 文件 | 内容 |
| --- | --- | --- |
| 参考交叉验证 | `crf_wbtest.mbt` + `reference_vectors_wbtest.mbt` | 13 条 numpy 参考向量（7 无约束 + 3 BIO + 3 BIOES），路径精确相等、score/logZ/边际 <1e-6 |
| 不变量 | `crf_wbtest.mbt` | 均匀模型 softmax 退化、BIO/BIOES 掩码语义、n-best 排序、前缀与非前缀（头部/中间）mask 等价、JSON 往返 |
| 性质测试 | `properties_wbtest.mbt` | 20 个确定性伪随机模型：score≤logZ、边际归一、nbest[0]==Viterbi、约束路径恒合法 |
| 新 API | `posterior_wbtest.mbt` / `confidence_wbtest.mbt` | 后验解码、路径校验、置信度/熵与手算对照 |
| 批量与标签名 | `batch_wbtest.mbt` / `labels_wbtest.mbt` | marginals_batch/confidence_batch 与逐条等价、空批、掩码长度错误；decode_labels/DecodedPath::labels 映射与越界错误 |
| 错误分支 | `coverage_wbtest.mbt` | 空序列、维度校验、不可满足约束、JSON 错误、标签集错误、批解码、logsumexp 边界、NaN/Inf 非有限输入守卫 |
| 黑盒 | `crf_test.mbt` | 仅 `@crf` 公开 API 冒烟 |

## 复现

```bash
moon test                      # 36/36（wasm）
moon test --target wasm-gc,js  # 36/36 × 2
moon test --target <t> --enable-coverage && moon coverage analyze  # 逐后端覆盖率
```