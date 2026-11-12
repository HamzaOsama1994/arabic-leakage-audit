"""Reviewer R2.1: effect of AraStance proxy-label rows on classification, calibration and routing.

Pre-registered in PREREGISTRATION.md (2026-11-12). Condition A = existing router models (full train/validation);
condition B = same recipe with AraStance removed from train and validation. Both scored on the same
non-AraStance test rows. Output: results/no_arastance_metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from leakaudit.router import calibrated, choose_tau, fit_temperature, novelty, selective, trust  # noqa: E402
from run_router import CLASSES, SEED, TARGETS, bilstm_stage, cached, svm_stage, two_col  # noqa: E402
from run_svm import load  # noqa: E402

MODELS = ["svm", "bilstm", "marbert_probe"]
N_BOOT = 2000


def ece(prob: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, correct = prob.max(1), prob.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def condition(d, y, svm, bil, emb, nov_emb) -> dict:
    """Fit heads/calibrators/tau on this condition's train+validation; return test arrays and taus."""
    head = LogisticRegression(max_iter=3000, random_state=SEED).fit(emb["train"], y["train"])
    mar = {s: head.decision_function(emb[s]) for s in ["validation", "test"]}
    platt = LogisticRegression(C=1e6).fit(svm["validation"].reshape(-1, 1), y["validation"])
    t_bil = fit_temperature(bil["validation"], y["validation"])
    t_mar = fit_temperature(two_col(mar["validation"]), y["validation"])
    q95 = np.quantile(novelty(nov_emb["validation"], nov_emb["train"]), 0.95)
    out = {}
    for s in ["validation", "test"]:
        probs = {"svm": platt.predict_proba(svm[s].reshape(-1, 1)), "bilstm": calibrated(bil[s], t_bil),
                 "marbert_probe": calibrated(two_col(mar[s]), t_mar)}
        raw = {"bilstm": softmax(bil[s], axis=1), "marbert_probe": softmax(two_col(mar[s]), axis=1)}
        label = (svm[s] > 0).astype(int)
        high = novelty(nov_emb[s], nov_emb["train"]) > q95
        out[s] = {"label": label, "probs": probs, "raw": raw,
                  "preds": np.stack([label, bil[s].argmax(1), (mar[s] > 0).astype(int)]),
                  "trust": trust(list(probs.values()), label, high),
                  "svm_only": probs["svm"][np.arange(len(label)), label]}
    out["tau"] = {f"{n}@{t}": choose_tau(out["validation"][n], t) for n in ["trust", "svm_only"] for t in TARGETS}
    out["calibration"] = {"bilstm_T": t_bil, "marbert_T": t_mar,
                          "svm_platt": [float(platt.coef_[0, 0]), float(platt.intercept_[0])]}
    return out


def score(c: dict, y: np.ndarray) -> dict:
    te = c["test"]
    correct = te["label"] == y
    agree = (te["preds"] == te["preds"][0]).all(0)
    res = {
        "per_model": {m: {"accuracy": float((te["preds"][i] == y).mean()),
                          "macro_f1": float(f1_score(y, te["preds"][i], average="macro")),
                          "ece_calibrated": ece(te["probs"][m], y)} for i, m in enumerate(MODELS)},
        "ece_raw": {m: ece(p, y) for m, p in te["raw"].items()},
        "agreement": {"agree_rate": float(agree.mean()), "agree_acc": float(correct[agree].mean()),
                      "svm_acc_on_split": float(correct[~agree].mean())},
        "operating_points": {k: selective(te[k.split("@")[0]] >= tau, correct) for k, tau in c["tau"].items()},
        "calibration": c["calibration"],
    }
    return res


def paired_boot(a: dict, b: dict, y: np.ndarray, units: np.ndarray) -> dict:
    """95% CI of B minus A, resampling test components."""
    rng = np.random.default_rng(SEED)
    uniq, inv = np.unique(units, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    stats = {"svm_accuracy": [], "trust@0.8_selective_accuracy": [], "svm_only@0.8_selective_accuracy": []}

    def parts(c):
        te = c["test"]
        return te["label"] == y, {n: te[n] >= c["tau"][f"{n}@0.8"] for n in ["trust", "svm_only"]}

    (ca, acc_a), (cb, acc_b) = parts(a), parts(b)
    for _ in range(N_BOOT):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))])
        stats["svm_accuracy"].append(cb[idx].mean() - ca[idx].mean())
        for n in ["trust", "svm_only"]:
            sa = ca[idx][acc_a[n][idx]].mean()
            sb = cb[idx][acc_b[n][idx]].mean()
            stats[f"{n}@0.8_selective_accuracy"].append(sb - sa)
    return {k: [float(x) for x in np.percentile(v, [2.5, 97.5])] for k, v in stats.items()}


def main() -> None:
    d = load("cc")
    y = {s: d[s]["binary_label"].map(CLASSES.index).to_numpy() for s in d}
    keep = {s: (d[s]["source"] != "AraStance").to_numpy() for s in d}

    # condition A: cached full-data artifacts from run_router.py, then restricted to non-AraStance test rows
    svm_a = cached("svm_scores", lambda: svm_stage(d))
    bil_a = cached("bilstm_logits", lambda: bilstm_stage(d, y))
    emb = {s: cached(f"marbert_{s}", lambda: None)["x"] for s in d}
    nov = cached("minilm_claims", lambda: None)
    a = condition(d, y, svm_a, bil_a, emb, nov)
    a["test"] = restrict(a["test"], keep["test"])

    # condition B: AraStance removed everywhere; embeddings are training-independent, so subset the cache
    db = {s: d[s][keep[s]].reset_index(drop=True) for s in d}
    yb = {s: y[s][keep[s]] for s in d}
    svm_b = cached("noara_svm_scores", lambda: svm_stage(db))
    bil_b = cached("noara_bilstm_logits", lambda: bilstm_stage(db, yb))
    b = condition(db, yb, svm_b, bil_b, {s: emb[s][keep[s]] for s in d}, {s: nov[s][keep[s]] for s in d})

    yt, units = yb["test"], db["test"]["component"].to_numpy()
    res = {
        "sizes_B": {s: len(db[s]) for s in db},
        "arastance_rows_removed": {s: int((~keep[s]).sum()) for s in d},
        "eval_rows": len(yt),
        "A_full_training": score(a, yt),
        "B_no_arastance": score(b, yt),
        "B_minus_A_ci95": paired_boot(a, b, yt, units),
    }
    (ROOT / "results" / "no_arastance_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


def restrict(te: dict, m: np.ndarray) -> dict:
    return {"label": te["label"][m], "preds": te["preds"][:, m], "trust": te["trust"][m], "svm_only": te["svm_only"][m],
            "probs": {k: v[m] for k, v in te["probs"].items()}, "raw": {k: v[m] for k, v in te["raw"].items()}}


if __name__ == "__main__":
    main()
