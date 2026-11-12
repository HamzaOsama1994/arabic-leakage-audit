# Arabic leakage audit: evidence-level leakage, leakage-safe splits and selective routing

Code, split manifests and result files for the ACLing 2026 paper
*Routing Uncertainty Rather Than Hiding It: A Leakage-Audited Arabic Misinformation Verification Pipeline with
Selective Prediction and Citation-Constrained Abstention* (Procedia Computer Science).

## Status

| Paper result | Reproduced here | Script |
|---|---|---|
| 57.2% of claim-grouped test rows reuse training evidence verbatim | exact | `scripts/leak_check.py` |
| Linear SVM on the historical claim-grouped split, 87.26% macro-F1 | exact | `scripts/run_svm.py historical` |
| Leakage-safe connected-component split | rebuilt from the paper's procedure (13,927/2,000/4,000; the original split was 13,999/2,000/4,000) | `scripts/build_cc_split.py` |
| Validation-only router rerun and SVM-confidence-only baseline (Table 2) | yes, on the rebuilt split | `scripts/run_router.py` |
| Equal-coverage router vs confidence-only comparison (post-hoc) | yes | `scripts/diag_router_matched.py` |
| AraStance (stance-proxy label) ablation (Section 6.1) | yes, on the rebuilt split | `scripts/run_no_arastance.py` |

The original-split figures (86.45% SVM accuracy; 76.90% coverage at 92.65% selective accuracy) came from an earlier
run whose code is not part of this release; the paper reports them as exploratory and compares rerun results only
with each other. `PREREGISTRATION.md` records each analysis as it was planned before its test set was scored.

## Inputs (not redistributed)

The scripts read harmonised claim–evidence files `source_binary_{train,validation,test}.csv` from `data/inputs/`
(or set `LEAKAUDIT_DATA_DIR`). They are built from the public datasets VERA-ARAB, ArCOV19-Rumors and AraStance,
which must be obtained from their authors under their own licences. `harmonisation/` holds the scripts that
produced them, included for provenance (paths need adapting). Required columns: `sample_id, source, claim_norm,
evidence_norm, claim_group, binary_label`. SHA-256 of the files used are in `results/cc_split_summary.json`
(`inputs`); check yours against them before running.

Released here: `data/cc_split_manifest.csv` (sample id, component, partition for every row of the rebuilt split),
per-row test predictions and all metrics files in `results/`. No claim or evidence text is included.

## Reproduce

Python 3.10+ (runs used 3.14.2), CPU is enough (BiLSTM ~10 min; MARBERT embedding ~25 min).

```bash
pip install -e .
python -m pytest                          # unit tests for split and router logic
python scripts/leak_check.py              # evidence-reuse audit of the historical split
python scripts/build_cc_split.py          # rebuild split -> data/cc_split_manifest.csv, results/cc_split_summary.json
python scripts/run_svm.py cc historical   # SVM on both splits (C tuned on validation, test scored once)
python scripts/run_router.py              # Table 2 rerun; caches model outputs in artifacts/
python scripts/diag_router_matched.py     # post-hoc equal-coverage comparison
python scripts/run_no_arastance.py        # AraStance ablation (needs run_router.py artifacts)
```

Pinned: seed 42; pandas 3.0.1, scikit-learn 1.8.0, numpy 2.4.2, scipy 1.17.1, torch 2.11.0, transformers 5.5.0,
sentence-transformers 5.5.1; `UBC-NLP/MARBERTv2` revision `fe88db9db8ccdb0c4e1627495f405c44a5f89066`;
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.
Confidence intervals are 95% bootstrap over split components (or claim groups for the historical split), not rows.

## Licence

Code: MIT (see `LICENSE`). The source datasets keep their original licences.
