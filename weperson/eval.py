"""Evaluate baseline and optional SQA on Market-1501."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from weperson.data import Market1501, build_transforms, resolve_market_root
from weperson.metrics import evaluate_rank
from weperson.models import build_baseline
from weperson.models.sqa import SQA, collect_params, configure_model
from weperson.train import extract_features, load_config, run_eval


@torch.no_grad()
def _gallery_feats(model, loader, device):
    model.eval()
    return extract_features(model, loader, device)


def eval_sqa(model, cfg, device) -> dict:
    """Per-query episodic SQA; gallery features from frozen source model."""
    root = resolve_market_root(cfg)
    tf = build_transforms(cfg, train=False)
    qds = Market1501(root, "query", transform=tf, max_samples=cfg["data"].get("max_eval_samples"))
    gds = Market1501(root, "gallery", transform=tf, max_samples=cfg["data"].get("max_eval_samples"))
    nw = int(cfg["data"].get("num_workers", 2))
    bs = int(cfg["eval"].get("batch_size", 64))
    g_loader = DataLoader(gds, batch_size=bs, shuffle=False, num_workers=nw)
    gg = _gallery_feats(model, g_loader, device)

    sqa_cfg = cfg.get("sqa", {})
    adapted = configure_model(model, w=float(sqa_cfg.get("w", 0.5)), alpha=float(sqa_cfg.get("alpha", 0.9)))
    params, _ = collect_params(adapted)
    opt = torch.optim.SGD(params, lr=float(sqa_cfg.get("lr", 1e-4)))
    sqa = SQA(
        adapted, opt,
        lambd=float(sqa_cfg.get("lambd", 0.0051)),
        aug_n=int(sqa_cfg.get("aug_n", 8)),
        steps=int(sqa_cfg.get("steps", 1)),
        episodic=True,
    )

    q_feats = []
    q_pids, q_cams = [], []
    q_loader = DataLoader(qds, batch_size=1, shuffle=False, num_workers=0)
    for batch in tqdm(q_loader, desc="SQA queries"):
        x = batch["image"].to(device)
        feat = sqa(x)
        q_feats.append(feat.detach().cpu())
        q_pids.append(int(batch["pid"]))
        q_cams.append(int(batch["camid"]))
    qq = {
        "feats": torch.cat(q_feats, 0),
        "pids": np.array(q_pids),
        "cams": np.array(q_cams),
    }
    return evaluate_rank(
        qq["feats"], qq["pids"], qq["cams"], gg["feats"], gg["pids"], gg["cams"],
        ranks=list(cfg["eval"].get("ranks", [1, 5, 10])),
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--sqa", action="store_true", help="Run Single Query Adaptation eval")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    n = int(state["classifier.weight"].shape[0])
    model = build_baseline(cfg, num_classes=n).to(device)
    model.load_state_dict(state)

    base = run_eval(model, cfg, device)
    out = {"baseline": {k: base[k] for k in base if k != "cmc"}, "checkpoint": args.checkpoint}
    if args.sqa or cfg.get("sqa", {}).get("enabled", False):
        # reload clean weights for SQA configure
        model2 = build_baseline(cfg, num_classes=n).to(device)
        model2.load_state_dict(state)
        sqa_m = eval_sqa(model2, cfg, device)
        out["sqa"] = {k: sqa_m[k] for k in sqa_m if k != "cmc"}
    Path(cfg["paths"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["log_dir"]) / "eval_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
