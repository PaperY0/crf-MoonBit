#!/usr/bin/env python3
"""Differential cross-check against real pytorch-crf (CI job torchcrf-diff).

Requires: pip install torch pytorch-crf
Run from the project root:
  python tools/diff_torchcrf.py   # generates torchcrf_diff_wbtest.mbt (gitignored)
  moon test --target wasm         # runs the differential assertions

Expected values come from torchcrf's own decode / _compute_score /
_compute_normalizer in float64. Weights round-trip through
tools/export_torchcrf.py, so the exporter is validated end-to-end as well.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torchcrf import CRF

ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(20261125)


def q(x):
    """Quantize to 10 significant digits (well below the 1e-6 test tolerance)."""
    return float(f"{float(x):.10g}")


def arr(xs):
    return "[" + ", ".join(repr(q(x)) for x in xs) + "]"


def arr2(rows):
    return "[" + ", ".join(arr(r) for r in rows) + "]"


def main():
    blocks = [
        "// CI 差分对拍：由 tools/diff_torchcrf.py 生成（需 torch），勿手改、勿提交。",
        "",
    ]
    configs = [(2, 1, False), (3, 5, False), (4, 8, False), (5, 12, False), (6, 7, True)]
    for idx, (k, t, padded) in enumerate(configs):
        crf = CRF(k, batch_first=True).double()
        with torch.no_grad():
            crf.start_transitions.copy_(torch.randn(k, dtype=torch.float64))
            crf.end_transitions.copy_(torch.randn(k, dtype=torch.float64))
            crf.transitions.copy_(torch.randn(k, k, dtype=torch.float64))
        emit = torch.randn(1, t, k, dtype=torch.float64)
        mask = torch.ones(1, t, dtype=torch.bool)
        if padded:
            mask[0, -2:] = False
        # 权重经导出脚本往返，顺带验证 export_torchcrf.py。
        with tempfile.TemporaryDirectory() as td:
            pt = Path(td) / "sd.pt"
            js = Path(td) / "model.json"
            torch.save(crf.state_dict(), pt)
            subprocess.run(
                [sys.executable, str(ROOT / "tools" / "export_torchcrf.py"), str(pt), str(js)],
                check=True,
                capture_output=True,
            )
            weights = json.loads(js.read_text(encoding="utf-8"))
        path = [int(v) for v in crf.decode(emit, mask)[0]]
        # _compute_score 要求 tags 与 emissions 同形；padding 位补 0，由 mask 剔除。
        tags_full = path + [0] * (t - len(path))
        tags_t = torch.tensor([tags_full], dtype=torch.long)
        # 内部方法恒为 seq-first 布局（batch_first 只影响公开 API），需转置。
        emit_sf = emit.transpose(0, 1).contiguous()
        mask_sf = mask.transpose(0, 1).contiguous()
        tags_sf = tags_t.transpose(0, 1).contiguous()
        score = float(crf._compute_score(emit_sf, tags_sf, mask_sf)[0].detach())
        logz = float(crf._compute_normalizer(emit_sf, mask_sf)[0].detach())
        mask_lit = (
            "Some([" + ", ".join("true" if b else "false" for b in mask[0]) + "])"
            if padded
            else "None"
        )
        blocks += [
            "///|",
            f'test "torchcrf differential case {idx}: K={k} T={t} padded={padded}" {{',
            f"  let crf = Crf::new({k}, {arr(weights['start_transitions'])}, "
            f"{arr(weights['end_transitions'])}, {arr2(weights['transitions'])}) catch {{ e => fail(e.message()) }}",
            f"  let emit = {arr2(emit[0].tolist())}",
            f"  let mask : Array[Bool]? = {mask_lit}",
            "  let path = crf.decode(emit, mask, None) catch { e => fail(e.message()) }",
            f"  assert_eq(path, {arr(path)})",
            "  let vp = crf.viterbi(emit, mask, None) catch { e => fail(e.message()) }",
            f'  close(vp.score, {q(score)!r}, "torchcrf score")',
            "  let lz = crf.log_partition(emit, mask, None) catch { e => fail(e.message()) }",
            f'  close(lz, {q(logz)!r}, "torchcrf logz")',
            "}",
            "",
        ]
    out = ROOT / "torchcrf_diff_wbtest.mbt"
    out.write_text("\n".join(blocks), encoding="utf-8")
    print(f"generated {len(configs)} differential cases -> {out.name}")


if __name__ == "__main__":
    main()