#!/usr/bin/env python3
"""Independent numpy reference for linear-chain CRF inference.

Mirrors the torchcrf (pytorch-crf) recurrences exactly:
  forward log-partition, Viterbi, forward-backward marginals, path score.
Used ONLY to generate cross-validation vectors for the MoonBit library;
the MoonBit code shares no source with this file.

Run:  python tools/gen_vectors.py
Writes: reference_vectors_wbtest.mbt  (embedded test data, all backends)
        data/reference_vectors.json   (human-readable inspection copy)
"""
import json
import os
import numpy as np

NEG = -1e308
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lse(xs):
    xs = np.asarray(xs, dtype=np.float64)
    m = float(xs.max())
    if m <= -1e300:
        return NEG
    return float(m + np.log(np.exp(xs - m).sum()))


def unpack(ca, K):
    if ca is None:
        return (np.ones(K, bool), np.ones(K, bool), np.ones((K, K), bool))
    return ca


def forward_logz(start, end, trans, emit, ca=None):
    L, K = emit.shape
    sa, ea, ta = unpack(ca, K)
    score = np.where(sa, start + emit[0], NEG)
    for t in range(1, L):
        nxt = np.full(K, NEG)
        for j in range(K):
            nxt[j] = lse(np.where(ta[:, j], score + trans[:, j], NEG)) + emit[t, j]
        score = nxt
    return lse(np.where(ea, score + end, NEG))


def viterbi(start, end, trans, emit, ca=None):
    L, K = emit.shape
    sa, ea, ta = unpack(ca, K)
    score = np.where(sa, start + emit[0], NEG)
    back = []
    for t in range(1, L):
        nxt = np.full(K, NEG)
        bt = np.zeros(K, int)
        for j in range(K):
            cand = np.where(ta[:, j], score + trans[:, j], NEG)
            i = int(cand.argmax())
            nxt[j] = cand[i] + emit[t, j]
            bt[j] = i
        back.append(bt)
        score = nxt
    cand = np.where(ea, score + end, NEG)
    bk = int(cand.argmax())
    best = float(cand[bk])
    path = [0] * L
    path[L - 1] = bk
    cur = bk
    for t in range(L - 1, 0, -1):
        cur = int(back[t - 1][cur])
        path[t - 1] = cur
    return path, best


def marginals(start, end, trans, emit, ca=None):
    L, K = emit.shape
    sa, ea, ta = unpack(ca, K)
    alpha = np.full((L, K), NEG)
    alpha[0] = np.where(sa, start + emit[0], NEG)
    for t in range(1, L):
        for j in range(K):
            alpha[t, j] = lse(np.where(ta[:, j], alpha[t - 1] + trans[:, j], NEG)) + emit[t, j]
    logz = lse(np.where(ea, alpha[L - 1] + end, NEG))
    beta = np.full((L, K), NEG)
    beta[L - 1] = np.where(ea, end, NEG)
    for t in range(L - 2, -1, -1):
        for i in range(K):
            beta[t, i] = lse(np.where(ta[i, :], trans[i, :] + emit[t + 1] + beta[t + 1], NEG))
    return np.exp(alpha + beta - logz)


def score_path(start, end, trans, emit, path):
    L, _ = emit.shape
    s = start[path[0]] + emit[0, path[0]]
    for t in range(1, L):
        s += trans[path[t - 1], path[t]] + emit[t, path[t]]
    s += end[path[L - 1]]
    return float(s)


def bio_mask(tags):
    K = len(tags)
    pref = [t.split("-")[0] for t in tags]
    typ = [t.split("-", 1)[1] if "-" in t else "" for t in tags]
    sa = np.array([pref[i] != "I" for i in range(K)])
    ea = np.ones(K, bool)
    ta = np.array([
        [(pref[j] != "I") or ((pref[i] in ("B", "I")) and typ[i] == typ[j]) for j in range(K)]
        for i in range(K)
    ])
    return sa, ea, ta


def fmt_double(x):
    return repr(float(x))


def fmt_arr(a):
    return "[" + ", ".join(fmt_double(x) for x in a) + "]"


def fmt_mat(m):
    return "[" + ", ".join(fmt_arr(row) for row in m) + "]"


def fmt_int_arr(a):
    return "[" + ", ".join(str(int(x)) for x in a) + "]"


def fmt_str_arr(a):
    return "[" + ", ".join('"%s"' % s for s in a) + "]"


def main():
    rng = np.random.default_rng(20261125)
    cases = []

    # Unconstrained cases across shapes.
    for (K, L) in [(2, 1), (2, 3), (3, 2), (3, 5), (4, 8), (5, 6), (2, 12)]:
        start = rng.normal(0, 0.7, K)
        end = rng.normal(0, 0.7, K)
        trans = rng.normal(0, 0.9, (K, K))
        emit = rng.normal(0, 1.0, (L, K))
        path, sc = viterbi(start, end, trans, emit)
        lz = forward_logz(start, end, trans, emit)
        mg = marginals(start, end, trans, emit)
        cases.append(dict(name=f"u_k{K}_l{L}", K=K, start=start, end=end, trans=trans,
                          emit=emit, bio=False, tags=[], path=path, score=sc, logz=lz, mg=mg))

    # BIO-constrained NER cases.
    bio_tags = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
    for (L, seed) in [(5, 1), (7, 2), (9, 3)]:
        r = np.random.default_rng(seed)
        K = len(bio_tags)
        start = r.normal(0, 0.5, K)
        end = r.normal(0, 0.5, K)
        trans = r.normal(0, 0.8, (K, K))
        emit = r.normal(0, 1.2, (L, K))
        ca = bio_mask(bio_tags)
        path, sc = viterbi(start, end, trans, emit, ca)
        lz = forward_logz(start, end, trans, emit, ca)
        mg = marginals(start, end, trans, emit, ca)
        cases.append(dict(name=f"bio_l{L}", K=K, start=start, end=end, trans=trans,
                          emit=emit, bio=True, tags=bio_tags, path=path, score=sc, logz=lz, mg=mg))

    # Emit MoonBit whitebox test data.
    lines = []
    lines.append("// Auto-generated by tools/gen_vectors.py — DO NOT EDIT.")
    lines.append("// Independent numpy reference (mirrors torchcrf recurrences).")
    lines.append("")
    lines.append("///|")
    lines.append("struct RefCase {")
    lines.append("  name : String")
    lines.append("  num_tags : Int")
    lines.append("  start : Array[Double]")
    lines.append("  end : Array[Double]")
    lines.append("  trans : Array[Array[Double]]")
    lines.append("  emissions : Array[Array[Double]]")
    lines.append("  use_bio : Bool")
    lines.append("  tags : Array[String]")
    lines.append("  expected_path : Array[Int]")
    lines.append("  expected_score : Double")
    lines.append("  expected_logz : Double")
    lines.append("  expected_marginals : Array[Array[Double]]")
    lines.append("}")
    lines.append("")
    lines.append("///|")
    lines.append("let ref_cases : Array[RefCase] = [")
    for c in cases:
        lines.append("  RefCase::{")
        lines.append('    name: "%s",' % c["name"])
        lines.append("    num_tags: %d," % c["K"])
        lines.append("    start: %s," % fmt_arr(c["start"]))
        lines.append("    end: %s," % fmt_arr(c["end"]))
        lines.append("    trans: %s," % fmt_mat(c["trans"]))
        lines.append("    emissions: %s," % fmt_mat(c["emit"]))
        lines.append("    use_bio: %s," % ("true" if c["bio"] else "false"))
        lines.append("    tags: %s," % fmt_str_arr(c["tags"]))
        lines.append("    expected_path: %s," % fmt_int_arr(c["path"]))
        lines.append("    expected_score: %s," % fmt_double(c["score"]))
        lines.append("    expected_logz: %s," % fmt_double(c["logz"]))
        lines.append("    expected_marginals: %s," % fmt_mat(c["mg"]))
        lines.append("  },")
    lines.append("]")
    lines.append("")
    with open(os.path.join(ROOT, "reference_vectors_wbtest.mbt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Human-readable JSON copy.
    jcases = []
    for c in cases:
        jcases.append(dict(name=c["name"], num_tags=c["K"],
                           start_transitions=c["start"].tolist(),
                           end_transitions=c["end"].tolist(),
                           transitions=c["trans"].tolist(),
                           emissions=c["emit"].tolist(),
                           use_bio=c["bio"], tags=c["tags"],
                           expected_path=c["path"], expected_score=c["score"],
                           expected_logz=c["logz"], expected_marginals=c["mg"].tolist()))
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "reference_vectors.json"), "w", encoding="utf-8") as f:
        json.dump(jcases, f, indent=2)

    print(f"generated {len(cases)} reference cases")


if __name__ == "__main__":
    main()