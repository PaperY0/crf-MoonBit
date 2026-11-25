# 验收指南对照清单

逐条对应赛事验收指南，给出可核验证据。工具链版本由 `.moonbit-version` 钉住，CI 用同一版本复现。

| # | 验收要求 | 状态 | 证据 |
| --- | --- | --- | --- |
| 1 | MoonBit 为主要实现语言，moonc ≥ 0.10.16 | 满足 | 库代码全部为 MoonBit（10 个 `.mbt` 源文件）；`.moonbit-version` = `0.10.22+124d61d05`；`moon version --all` 可核验 |
| 2 | GitHub 仓库公开、提交记录清晰 | 满足 | https://github.com/PaperY0/crf-MoonBit 公开；30+ 个语义化提交，无空提交 / 无意义拆分 / 重复提交 |
| 3 | 源码结构清晰、完成声明的核心功能 | 满足 | 按职责分文件（math / model / constraints / decode / partition / io / labels / crf）；核心功能见 README「能力范围」，由 36 测试 + 13 参考向量验证 |
| 4 | README 含目标 / 安装 / 用法 / 示例、可复现 | 满足 | 双语 README：目标（为什么是它）、安装（`moon add`）、用法（快速开始 + API）、示例（`examples/ner`）；CI 校验参考向量逐字节可复现 |
| 5 | CI 覆盖检查 / 构建 / 测试 | 满足 | `ci.yml`：`moon fmt --check` + `moon check`（检查）、`moon build`（构建）、`moon test --enable-coverage` + 覆盖率门禁（测试）；wasm / wasm-gc / js 矩阵 + native |
| 6 | 至少一个可运行示例 | 满足 | `examples/ner`：`moon run examples/ner`；CI 四后端冒烟断言解码输出 |
| 7 | 完整测试覆盖核心功能路径 | 满足 | 36 测试，库代码行覆盖率三后端均 100%（CI 每后端门禁）；另有真实 pytorch-crf 差分对拍 41/41 |
| 8 | 发布到 mooncakes.io | 满足 | `PaperY0/crf@0.2.0`，https://mooncakes.io/docs/PaperY0/crf |
| 9 | OSI 认可许可证；参考 / 移植符合原许可证 | 满足 | MIT（OSI 认可）；参考 pytorch-crf（MIT）仅做语义对齐、未复制代码，许可证兼容 |

## 复现命令

```bash
moon version --all                    # 核验 moonc 版本（要求 ≥ 0.10.16）
moon check --target all               # 检查
moon build --target wasm              # 构建
moon test --target wasm,wasm-gc,js    # 测试
moon run examples/ner                 # 可运行示例
moon coverage analyze                 # 覆盖率
```

作为依赖安装：`moon add PaperY0/crf`，在 `moon.pkg` 中 `import { "PaperY0/crf" @crf }`。

## 许可证合规说明

本项目以 MIT 发布。设计参考 pytorch-crf（https://github.com/kmkurn/pytorch-crf ，MIT）：
仅对齐权重张量布局与 forward / Viterbi 递推语义以便互操作，未复制其源码（语言、范围、结构均不同），
符合 MIT 的宽松要求；交叉验证用的 numpy 参考实现为独立编写。详见 `APPLICATION.md` 第 6、7 项。