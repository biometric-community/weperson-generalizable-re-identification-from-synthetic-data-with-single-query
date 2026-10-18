# Fidelity audit: WePerson / SQA

Target: Li et al., TBIOM 2025 (DOI 10.1109/TBIOM.2025.3540919). Fixed 5-pass audit; stop after Pass 5.

## Per-pass statistics

| Pass | Focus | Confidence | Key fixes / evidence |
|------|-------|------------|----------------------|
| 1 | Layout / configs / scripts / docs | **92%** | `weperson/` package, YAML, scripts, README/LICENSE/DEVIATIONS |
| 2 | SQA / WABN vs upstream | **88%** | Refactored `lihe404/WePerson` SQA+WABN; episodic BN affine + cross-corr loss |
| 3 | Train protocol / Market | **80%** | ResNet-50 BNNeck on Market (D1: no WePerson synthetic); 30 ep Adam |
| 4 | Eval baseline+SQA / REPORT | **85%** | Baseline CMC/mAP + per-query SQA; num_classes from ckpt; REPORT from JSON |
| 5 | Smoke + size gate + full train | **90%** | Full 30-ep: baseline mAP **73.61** / R1 **84.00**; SQA in-domain mAP **73.06** |

**Mean confidence after Pass 5: ~87%.** WePerson synthetic / MSMT17 / CUHK03 out of scope (D1, D3).

## Sign-off

Audit complete after Pass 5.
