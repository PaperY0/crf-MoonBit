#!/usr/bin/env python3
"""Export a trained torchcrf (pytorch-crf) model to the JSON schema PaperY0/crf loads.

Requires: pip install torch pytorch-crf
Usage:
  python tools/export_torchcrf.py path/to/state_dict.pt model.json

The state_dict must contain keys: start_transitions, end_transitions, transitions.
Emissions are produced by YOUR upstream encoder at inference time and passed to
Crf::decode / Crf::marginals separately; they are not part of the CRF weights.
"""
import json
import sys


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    import torch  # noqa: delayed import so the script is inspectable without torch

    sd = torch.load(sys.argv[1], map_location="cpu")
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    # strip a possible "crf." prefix
    clean = {}
    for k, v in sd.items():
        key = k.split("crf.")[-1]
        if key in ("start_transitions", "end_transitions", "transitions"):
            clean[key] = v.detach().cpu().double().tolist()
    for need in ("start_transitions", "end_transitions", "transitions"):
        if need not in clean:
            raise SystemExit(f"missing key in state_dict: {need}")
    num_tags = len(clean["start_transitions"])
    out = {
        "num_tags": num_tags,
        "start_transitions": clean["start_transitions"],
        "end_transitions": clean["end_transitions"],
        "transitions": clean["transitions"],
    }
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"exported num_tags={num_tags} -> {sys.argv[2]}")


if __name__ == "__main__":
    main()