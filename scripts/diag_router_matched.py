"""DIAGNOSTIC (not pre-planned, not used to pick an operating point): trust router vs SVM-confidence-only
at exactly matched test coverage, with component-bootstrap 95% CI of the difference."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results" / "router_test_decisions.csv")
correct = (df["svm_label"] == df["gold"]).to_numpy()
scores = {"trust": df["trust"].to_numpy(),
          "svm_only": np.where(df["svm_label"] == "Real", df["p_svm"], 1 - df["p_svm"])}
units = df["component"].to_numpy()

def sel_acc(score, c, idx, cov):
    s, k = score[idx], int(round(cov * len(idx)))
    top = np.argsort(-s, kind="stable")[:k]
    return c[idx][top].mean()

uniq, inv = np.unique(units, return_inverse=True)
rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
rng = np.random.default_rng(42)
boots = [np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(1000)]
full = np.arange(len(df))
out = {}
for cov in [0.7, 0.8, 0.9]:
    a, b = (sel_acc(scores[n], correct, full, cov) for n in ["trust", "svm_only"])
    diff = [sel_acc(scores["trust"], correct, i, cov) - sel_acc(scores["svm_only"], correct, i, cov) for i in boots]
    out[str(cov)] = {"trust": a, "svm_only": b, "diff": a - b, "diff_ci95": list(np.percentile(diff, [2.5, 97.5]))}
    print(f"coverage {cov:.0%}: trust {a:.4f}  svm_only {b:.4f}  diff {a-b:+.4f}  CI [{out[str(cov)]['diff_ci95'][0]:+.4f}, {out[str(cov)]['diff_ci95'][1]:+.4f}]")
(ROOT / "results" / "diag_router_matched_coverage.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
