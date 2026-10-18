# Implementation plan: WePerson / SQA

Paper: Li et al., TBIOM 2025, DOI [10.1109/TBIOM.2025.3540919](https://doi.org/10.1109/TBIOM.2025.3540919).

## Scope

1. ResNet-50 BNNeck trained on **Market-1501** (stand-in for WePerson — D1).
2. **SQA** (ωαBN + cross-corr calibration) refactored from upstream.
3. Eval baseline vs SQA on Market query/gallery.

## Figure inventory

| Paper fig / table | Our artifact |
|-------------------|--------------|
| Market Rank-1 / mAP | `outputs/figures/market_map_r1.svg` |
| Train curves | `outputs/figures/train_curves.svg` |

## Size gate

Market-1501 ≈ 0.14 GiB → **full_train** (30 epochs).
