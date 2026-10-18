# Report: WePerson / SQA

Market-1501 rebuild of Li et al., TBIOM 2025 (DOI [10.1109/TBIOM.2025.3540919](https://doi.org/10.1109/TBIOM.2025.3540919)). WePerson synthetic corpus not registered — trained on Market; SQA from upstream.

## Setup

- Image: 256×128, ResNet-50 BNNeck
- Epochs: 30, Adam lr=0.00035
- Dataset size gate: full_train (0.1388 GiB)

## Market-1501 results

| | mAP | Rank-1 | Rank-5 |
|--|--:|--:|--:|
| **Baseline** | 73.61 | 84.00 | 96.50 |
| **+ SQA** | 73.06 | 83.97 | 96.56 |

See `DEVIATIONS.md`.

## Figures

### `outputs/figures/market_map_r1.svg`

![](outputs/figures/market_map_r1.svg)

### `outputs/figures/train_curves.svg`

![](outputs/figures/train_curves.svg)

## Regenerate

```bash
bash scripts/report.sh
```
