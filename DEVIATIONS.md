# Deviations from the paper

| ID | Paper detail | Our choice | Reason |
|----|--------------|------------|--------|
| D1 | Train on WePerson synthetic (1.5k IDs / up to 4M imgs) | **Market-1501** supervised train | WePerson only on Baidu Pan; not under `projects/datasets/` |
| D2 | Direct-transfer QAConv 2.1 / TransReID baselines | ResNet-50 BNNeck + CE/triplet | Simpler host for SQA; SQA is the paper focus |
| D3 | CUHK03 / MSMT17_V2 / corruption suites | Market-1501 only | Only Market registered with payload |
| D4 | CIFAR-10-C classification robustness | Omitted | Out of ReID project scope |
| D5 | Single GPU vs paper hardware | `CUDA_VISIBLE_DEVICES=0` | Available hardware |
| D6 | SQA evaluated synth→real / real→real DG | In-domain Market query adaptation | No WePerson/MSMT source; SQA may not improve same-domain CMC |

## Taken from paper / upstream

- SQA: ωαBN + cross-corr BN affine calibration (`lihe404/WePerson` SQA/)
- SQA SGD lr `1e-4`, `n=8`, λ≈0.0051, episodic reset per query
