# WePerson: Generalizable Re-Identification From Synthetic Data With Single Query Adaptation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

PyTorch rebuild of Li et al. (TBIOM 2025): Market-1501 ReID baseline + **Single Query Adaptation (SQA)** (ωαBN). WePerson synthetic images are not shipped here — see `DEVIATIONS.md`.

**Repository:** https://github.com/biometric-community/weperson-generalizable-re-identification-from-synthetic-data-with-single-query

**Paper:** [10.1109/TBIOM.2025.3540919](https://doi.org/10.1109/TBIOM.2025.3540919)

```bash
bash scripts/setup_env.sh
bash scripts/train.sh configs/smoke.yaml
bash scripts/train_full.sh
bash scripts/eval.sh && bash scripts/report.sh
```

## Results (Market-1501, this rebuild)

| | mAP | Rank-1 | Rank-5 |
|--|--:|--:|--:|
| **Baseline** | 73.61 | 84.00 | 96.50 |
| **+ SQA** | 73.06 | 83.97 | 96.56 |

SQA is designed for **cross-domain** shift (synth→real); in-domain Market→Market yields little gain / slight mAP drop — see `DEVIATIONS.md` / `REPORT.md`.

## Citation

```
@article{li2025weperson,
  title={WePerson: Generalizable Re-Identification From Synthetic Data With Single Query Adaptation},
  author={Li, He and Ye, Mang and Su, Kehua and Du, Bo},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2025},
  doi={10.1109/TBIOM.2025.3540919}
}
```
