#!/usr/bin/env python3
"""Put a trained checkpoint and its decode priors into ``src/deeprok/data/``.

Keeps only the weights and the architecture keys the loader reads (no optimizer
state, no training arguments), writes ``model_card.json`` with the provenance
printed in every GFF header, and checks that the stripped file loads to the same
weights as the original.

    python package/dev/stage_model.py \\
      --checkpoint training_runs/v4_variant_c_prok_joint642_keephyp_cnn3_lstm1/best.pt \\
      --priors stats/gtdb_prok_keephyp_train666_pooled.json \\
      --name prok666_keephyp --training-run "polaris job 26153, best epoch 13"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parents[1] / "src" / "deeprok" / "data"
# exactly the keys model_v4.load_prokaryotic_model_v4_from_checkpoint reads, plus the
# head layout for the record
KEEP = ("model_state_dict", "cnn_stride", "dilated_branch_channels", "dilated_dilations",
        "enable_aux_head", "enable_dilated_stem", "enable_phase_channel", "enable_sd_channel",
        "filter_size", "initial_kernel_size", "inp_size", "kernel_size", "numb_conv",
        "numb_lstm", "pool_size", "units", "output_size", "label_mode", "head_features")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--priors", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--training-run", required=True)
    ap.add_argument("--training-data", default="666 GTDB r232 family representatives "
                                               "(620 bacteria, 46 archaea)")
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ck:
        ck = {"model_state_dict": ck}
    slim = {k: ck[k] for k in KEEP if k in ck}
    dropped = sorted(set(ck) - set(slim))
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "model.pt"
    torch.save(slim, out)
    back = torch.load(out, map_location="cpu", weights_only=True)
    for k, v in ck["model_state_dict"].items():
        if not torch.equal(v, back["model_state_dict"][k]):
            raise SystemExit(f"weight {k} changed while stripping")
    shutil.copyfile(a.priors, DATA / "priors.json")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    card = {
        "name": a.name,
        "sha256": sha,
        "source_checkpoint": str(a.checkpoint),
        "source_checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        "training_run": a.training_run,
        "training_data": a.training_data,
        "priors": a.priors.name,
        "n_parameters": int(sum(v.numel() for v in ck["model_state_dict"].values())),
    }
    (DATA / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, sha256 {sha[:12]}); "
          f"dropped keys: {dropped}")


if __name__ == "__main__":
    main()
