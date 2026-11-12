"""Linear SVM (word+char TF-IDF on claim [SEP] evidence), settings copied from FAAV experiments/final_studies_classical.py.

Tuned on validation by macro-F1, refit on train+validation, test scored once.
Splits: 'cc' = rebuilt connected-components split (paper: 86.45 acc / 86.40 macro-F1),
        'historical' = claim-grouped source_binary split (paper: 87.26 macro-F1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakaudit import data_dir  # noqa: E402

SEED = 42
C_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
CLASS_WEIGHTS = [None, "balanced"]
PAPER = {"cc": {"accuracy": 0.8645, "macro_f1": 0.8640}, "historical": {"macro_f1": 0.8726}}
N_BOOT = 2000


def vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.995,
                                 max_features=70_000, sublinear_tf=True, strip_accents=None)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=90_000, sublinear_tf=True)),
    ])


def load(split: str) -> dict[str, pd.DataFrame]:
    parts = {p: pd.read_csv(data_dir() / f"source_binary_{p}.csv", low_memory=False).fillna("")
             for p in ["train", "validation", "test"]}
    if split == "cc":
        pool = pd.concat(parts.values(), ignore_index=True)
        manifest = pd.read_csv(ROOT / "data" / "cc_split_manifest.csv")
        pool = pool.merge(manifest, on="sample_id", how="inner")
        parts = {p: pool[pool["split"] == p].reset_index(drop=True) for p in ["train", "validation", "test"]}
        for df in parts.values():
            df["unit"] = df["component"]
    else:
        for df in parts.values():
            df["unit"] = df["claim_group"]
    for df in parts.values():
        df["text"] = df["claim_norm"].astype(str) + " [SEP] " + df["evidence_norm"].astype(str)
    return parts


def bootstrap_ci(y: np.ndarray, p: np.ndarray, units: np.ndarray) -> dict[str, list[float]]:
    """95% CI resampling whole units (components / claim groups), not rows."""
    rng = np.random.default_rng(SEED)
    uniq, inv = np.unique(units, return_inverse=True)
    rows_by_unit = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    acc, f1 = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([rows_by_unit[i] for i in rng.integers(0, len(uniq), len(uniq))])
        acc.append(accuracy_score(y[idx], p[idx]))
        f1.append(f1_score(y[idx], p[idx], average="macro"))
    return {"accuracy": list(np.percentile(acc, [2.5, 97.5])), "macro_f1": list(np.percentile(f1, [2.5, 97.5]))}


def run(split: str) -> dict:
    d = load(split)
    vec = vectorizer()
    xtr = vec.fit_transform(d["train"]["text"])
    xva = vec.transform(d["validation"]["text"])
    best = None
    for cw in CLASS_WEIGHTS:
        for c in C_VALUES:
            m = LinearSVC(C=c, class_weight=cw, max_iter=10_000, random_state=SEED).fit(xtr, d["train"]["binary_label"])
            score = f1_score(d["validation"]["binary_label"], m.predict(xva), average="macro")
            if best is None or score > best[0]:
                best = (score, c, cw)
    _, c, cw = best
    final = pd.concat([d["train"], d["validation"]], ignore_index=True)
    vec = vectorizer()
    model = LinearSVC(C=c, class_weight=cw, max_iter=10_000, random_state=SEED).fit(
        vec.fit_transform(final["text"]), final["binary_label"])
    test = d["test"]
    pred = model.predict(vec.transform(test["text"]))
    y = test["binary_label"].to_numpy()
    out = {
        "split": split,
        "sizes": {k: len(v) for k, v in d.items()},
        "selected": {"C": c, "class_weight": cw or "none", "validation_macro_f1": round(best[0], 4)},
        "test": {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")},
        "ci95": bootstrap_ci(y, pred, test["unit"].to_numpy()),
        "paper": PAPER[split],
    }
    pd.DataFrame({"sample_id": test["sample_id"], "gold": y, "pred": pred,
                  "score": model.decision_function(vec.transform(test["text"]))}).to_csv(
        ROOT / "results" / f"svm_{split}_test_predictions.csv", index=False)
    return out


def main() -> None:
    splits = sys.argv[1:] or ["cc", "historical"]
    for split in splits:
        res = run(split)
        (ROOT / "results" / f"svm_{split}_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
