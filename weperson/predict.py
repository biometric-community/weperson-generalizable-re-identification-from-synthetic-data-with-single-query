"""Extract embeddings for images in a folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from weperson.data import Market1501, build_transforms, resolve_market_root
from weperson.models import build_baseline
from weperson.train import load_config


class ImageFolder(Dataset):
    def __init__(self, root: Path, transform):
        self.paths = sorted(list(Path(root).glob("*.jpg")) + list(Path(root).glob("*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return {"image": self.transform(Image.open(self.paths[i]).convert("RGB")), "path": str(self.paths[i])}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    p.add_argument("--input", required=True)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    n = int(state["classifier.weight"].shape[0])
    model = build_baseline(cfg, num_classes=n).to(device)
    model.load_state_dict(state)
    model.eval()
    ds = ImageFolder(Path(args.input), build_transforms(cfg, False))
    rows = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=32):
            feat = model(batch["image"].to(device))["feat"].cpu()
            for path, f in zip(batch["path"], feat):
                rows.append({"path": path, "feat": f.tolist()})
    out = Path(cfg["paths"]["pred_dir"]) / "predictions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    print(f"Wrote {len(rows)} → {out}")


if __name__ == "__main__":
    main()
