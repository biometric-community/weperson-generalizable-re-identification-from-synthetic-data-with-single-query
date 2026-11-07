"""Train Market-1501 ReID baseline (WePerson synthetic stand-in)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from weperson.data import Market1501, RandomIdentitySampler, build_transforms, resolve_market_root
from weperson.losses import reid_loss
from weperson.metrics import evaluate_rank
from weperson.models import build_baseline


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _ensure_dirs(cfg: dict) -> None:
    for k in ("checkpoint_dir", "log_dir", "pred_dir", "figure_dir"):
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def extract_features(model, loader, device) -> Dict[str, Any]:
    model.eval()
    feats, pids, cams = [], [], []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        out = model(img)
        feats.append(out["feat"].cpu())
        pids.append(batch["pid"].numpy())
        cams.append(batch["camid"].numpy())
    import numpy as np

    return {"feats": torch.cat(feats, 0), "pids": np.concatenate(pids), "cams": np.concatenate(cams)}


def run_eval(model, cfg, device) -> Dict[str, float]:
    root = resolve_market_root(cfg)
    tf = build_transforms(cfg, train=False)
    q = Market1501(root, "query", transform=tf, max_samples=cfg["data"].get("max_eval_samples"))
    g = Market1501(root, "gallery", transform=tf, max_samples=cfg["data"].get("max_eval_samples"))
    nw = int(cfg["data"].get("num_workers", 4))
    bs = int(cfg["eval"].get("batch_size", 64))
    qq = extract_features(model, DataLoader(q, batch_size=bs, shuffle=False, num_workers=nw), device)
    gg = extract_features(model, DataLoader(g, batch_size=bs, shuffle=False, num_workers=nw), device)
    return evaluate_rank(
        qq["feats"], qq["pids"], qq["cams"], gg["feats"], gg["pids"], gg["cams"],
        ranks=list(cfg["eval"].get("ranks", [1, 5, 10])),
    )


def lr_at(epoch: int, cfg: dict) -> float:
    base = float(cfg["train"]["lr"])
    drops = cfg["train"].get("lr_drops", [20])
    lr = base
    for d in drops:
        if epoch >= int(d):
            lr *= 0.1
    return lr


def main(argv=None):
    p = argparse.ArgumentParser(description="Train WePerson Market baseline")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    _ensure_dirs(cfg)
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg["train"].get("seed", 42)))
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])

    root = resolve_market_root(cfg)
    train_ds = Market1501(
        root, "train", transform=build_transforms(cfg, True),
        max_samples=cfg["data"].get("max_train_samples"),
    )
    model = build_baseline(cfg, num_classes=train_ds.num_classes).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 5e-4)),
    )
    ce = nn.CrossEntropyLoss(label_smoothing=float(cfg["train"].get("label_smoothing", 0.1)))
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    batch = int(cfg["train"]["batch_size"])
    ni = int(cfg["train"].get("num_instances", 4))
    sampler = RandomIdentitySampler(train_ds, num_instances=ni, batch_size=batch)
    loader = DataLoader(
        train_ds, batch_size=batch, sampler=sampler,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True, drop_last=True,
    )
    epochs = int(cfg["train"]["epochs"])
    eval_every = int(cfg["eval"].get("every", 10))
    max_steps = cfg["train"].get("max_steps")
    history, best, global_step = [], {"mAP": -1.0}, 0

    for epoch in range(epochs):
        t0 = time.time()
        cur_lr = lr_at(epoch, cfg)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        model.train()
        running, n = 0.0, 0
        for bd in loader:
            x = bd["image"].to(device, non_blocking=True)
            y = bd["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = model(x)
                losses = reid_loss(
                    out, y, ce,
                    margin=float(cfg["train"].get("margin", 0.3)),
                    lambda_tri=float(cfg["train"].get("lambda_tri", 1.0)),
                )
                loss = losses["loss"]
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach())
            n += 1
            global_step += 1
            if max_steps is not None and global_step >= int(max_steps):
                break
        row = {"epoch": epoch + 1, "lr": cur_lr, "loss": running / max(n, 1), "sec": time.time() - t0}
        if ((epoch + 1) % eval_every == 0) or (epoch + 1 == epochs):
            metrics = run_eval(model, cfg, device)
            row.update({k: metrics[k] for k in ("mAP", "Rank-1", "Rank-5") if k in metrics})
            print(f"[eval] epoch={epoch+1} mAP={metrics['mAP']:.2f} R1={metrics['Rank-1']:.2f}", flush=True)
            # No validation split: keep the latest checkpoint, not the best on the test set
            best = {k: metrics[k] for k in metrics if k != "cmc"}
            best["epoch"] = epoch + 1
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "best": best, "cfg": cfg}, ckpt_dir / "best.pt")
        history.append(row)
        print(f"[epoch] {epoch+1}/{epochs} loss={row['loss']:.4f} lr={cur_lr:.6f} sec={row['sec']:.1f}", flush=True)
        (log_dir / "train_history.json").write_text(json.dumps(history, indent=2))
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "history": history, "best": best}, ckpt_dir / "last.pt")
        if max_steps is not None and global_step >= int(max_steps):
            break

    summary = {
        "best": best,
        "note": "Trained on Market-1501 (WePerson synthetic blocked — see DEVIATIONS.md)",
        "epochs_ran": len(history),
        "config": args.config,
    }
    (log_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] best={best}", flush=True)


if __name__ == "__main__":
    main()
