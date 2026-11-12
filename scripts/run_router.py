"""Router re-implementation on the rebuilt CC split (pre-registered in PREREGISTRATION.md, 2026-11-07).

Stages cache to artifacts/ so reruns resume: svm -> bilstm -> marbert -> novelty -> route.
Outputs: results/router_test_decisions.csv, results/router_metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from leakaudit.router import calibrated, choose_tau, fit_temperature, novelty, selective, trust  # noqa: E402
from run_svm import load, vectorizer  # noqa: E402

ART = ROOT / "artifacts"
SEED = 42
CLASSES = ["Fake", "Real"]  # sorted, matches LinearSVC.classes_
TARGETS = [0.7, 0.8, 0.9, 1.0]
PAPER = {"agree_rate": 0.670, "agree_acc": 0.926, "split_majority_acc": 0.649,
         "operating_points": {"0.7": [0.7270, 0.9271], "0.8": [0.7690, 0.9265], "0.9": [0.8735, 0.9021], "1.0": [1.0, 0.8645]}}
N_BOOT = 2000


def cached(name: str, fn):
    path = ART / f"{name}.npz"
    if not path.exists():
        ART.mkdir(exist_ok=True)
        np.savez(path, **fn())
        print(f"[saved] {path.name}", flush=True)
    return dict(np.load(path))


def svm_stage(d):
    vec = vectorizer()
    model = LinearSVC(C=1.0, max_iter=10_000, random_state=SEED).fit(vec.fit_transform(d["train"]["text"]), d["train"]["binary_label"])
    assert list(model.classes_) == CLASSES
    return {s: model.decision_function(vec.transform(d[s]["text"])) for s in ["validation", "test"]}


class BiLSTM(nn.Module):
    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, 128, padding_idx=0)
        self.drop = nn.Dropout(0.30)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.out = nn.Linear(256, 2)

    def forward(self, ids, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.emb(ids)), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.out(self.drop(torch.cat([h[-2], h[-1]], dim=1)))


def bilstm_stage(d, y):
    torch.manual_seed(SEED)
    from collections import Counter
    counts = Counter(tok for t in d["train"]["text"] for tok in t.split())
    vocab = {tok: i + 2 for i, (tok, _) in enumerate(counts.most_common(50_000 - 2))}

    def encode(texts):
        rows = [[vocab.get(t, 1) for t in s.split()[:180]] or [1] for s in texts]
        ids = torch.zeros(len(rows), 180, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
        return ids, torch.tensor([len(r) for r in rows])

    X = {s: encode(d[s]["text"].tolist()) for s in d}
    n = np.bincount(y["train"], minlength=2)
    w = np.sqrt(len(y["train"]) / (2 * n)); w = torch.tensor(w / w.mean(), dtype=torch.float32)
    model = BiLSTM(len(vocab) + 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    def logits(split):
        model.eval()
        with torch.no_grad():
            ids, ln = X[split]
            return torch.cat([model(ids[i:i + 512], ln[i:i + 512]) for i in range(0, len(ids), 512)]).numpy()

    best, best_state, stale = -1.0, None, 0
    ytr = torch.tensor(y["train"])
    for epoch in range(1, 9):
        model.train()
        g = torch.Generator().manual_seed(SEED + epoch)
        for idx in torch.randperm(len(ytr), generator=g).split(128):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X["train"][0][idx], X["train"][1][idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        f1 = f1_score(y["validation"], logits("validation").argmax(1), average="macro")
        print(f"  bilstm epoch {epoch} val macro-F1 {f1:.4f}", flush=True)
        if f1 > best + 1e-5:
            best, best_state, stale = f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= 2:
                break
    model.load_state_dict(best_state)
    return {"validation": logits("validation"), "test": logits("test")}


def marbert_embed(frame: pd.DataFrame) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("UBC-NLP/MARBERTv2")
    model = AutoModel.from_pretrained("UBC-NLP/MARBERTv2").eval()
    claims = frame["claim_norm"].astype(str).str[:400].tolist()
    ev = frame["evidence_norm"].astype(str).str[:1200].tolist()
    out = []
    with torch.inference_mode():
        for i in range(0, len(claims), 64):
            enc = tok(claims[i:i + 64], ev[i:i + 64], max_length=128, truncation=True, padding=True, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).numpy())
            if i % 1280 == 0:
                print(f"  marbert {i}/{len(claims)}", flush=True)
    return np.concatenate(out)


def minilm_embed(d) -> dict:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return {s: enc.encode(d[s]["claim_norm"].astype(str).tolist(), batch_size=128, normalize_embeddings=True) for s in d}


def two_col(score: np.ndarray) -> np.ndarray:
    return np.stack([-score / 2, score / 2], axis=1)


def boot(accept, correct, units) -> dict:
    rng = np.random.default_rng(SEED)
    uniq, inv = np.unique(units, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    cov, acc = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))])
        cov.append(accept[idx].mean()); acc.append(correct[idx][accept[idx]].mean())
    return {"coverage": list(np.percentile(cov, [2.5, 97.5])), "selective_accuracy": list(np.percentile(acc, [2.5, 97.5]))}


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    d = load("cc")
    y = {s: d[s]["binary_label"].map(CLASSES.index).to_numpy() for s in d}

    svm = cached("svm_scores", lambda: svm_stage(d))
    bil = cached("bilstm_logits", lambda: bilstm_stage(d, y))
    emb = {s: cached(f"marbert_{s}", lambda s=s: {"x": marbert_embed(d[s])})["x"] for s in d}
    nov_emb = cached("minilm_claims", lambda: minilm_embed(d))

    head = LogisticRegression(max_iter=3000, random_state=SEED).fit(emb["train"], y["train"])
    mar = {s: head.decision_function(emb[s]) for s in ["validation", "test"]}

    # calibration on validation only
    platt = LogisticRegression(C=1e6).fit(svm["validation"].reshape(-1, 1), y["validation"])
    p_svm = {s: platt.predict_proba(svm[s].reshape(-1, 1)) for s in ["validation", "test"]}
    t_bil = fit_temperature(bil["validation"], y["validation"])
    t_mar = fit_temperature(two_col(mar["validation"]), y["validation"])
    p_bil = {s: calibrated(bil[s], t_bil) for s in ["validation", "test"]}
    p_mar = {s: calibrated(two_col(mar[s]), t_mar) for s in ["validation", "test"]}

    nov = {s: novelty(nov_emb[s], nov_emb["train"]) for s in ["validation", "test"]}
    q80, q95 = np.quantile(nov["validation"], [0.80, 0.95])

    res, split_out = {}, {}
    for s in ["validation", "test"]:
        label = (svm[s] > 0).astype(int)  # returned label: raw SVM
        preds = np.stack([label, bil[s].argmax(1), (mar[s] > 0).astype(int)])
        high = nov[s] > q95
        split_out[s] = {
            "label": label, "preds": preds, "high": high,
            "trust": trust([p_svm[s], p_bil[s], p_mar[s]], label, high),
            "svm_only": p_svm[s][np.arange(len(label)), label],
            "category": np.where(nov[s] <= q80, "Familiar", np.where(high, "High", "Moderate")),
        }

    te, yt = split_out["test"], y["test"]
    correct = te["label"] == yt
    agree = (te["preds"] == te["preds"][0]).all(0)
    majority = (te["preds"].sum(0) >= 2).astype(int)
    units = d["test"]["component"].to_numpy()
    per_model = {m: {"accuracy": float((te["preds"][i] == yt).mean()), "macro_f1": float(f1_score(yt, te["preds"][i], average="macro"))}
                 for i, m in enumerate(["svm", "bilstm", "marbert_probe"])}

    points = {}
    for name in ["trust", "svm_only"]:
        for target in TARGETS:
            tau = choose_tau(split_out["validation"][name], target)
            accept = te[name] >= tau
            points[f"{name}@{target}"] = {"tau": tau, **selective(accept, correct), "ci95": boot(accept, correct, units)}
            if name == "trust" and target == 0.8:
                te["accept80"] = accept

    res = {
        "sizes": {s: len(d[s]) for s in d},
        "per_model_test": per_model,
        "calibration": {"bilstm_T": t_bil, "marbert_T": t_mar, "svm_platt": [float(platt.coef_[0, 0]), float(platt.intercept_[0])]},
        "novelty_thresholds_val_p80_p95": [float(q80), float(q95)],
        "novelty_share_test": pd.Series(te["category"]).value_counts(normalize=True).round(4).to_dict(),
        "agreement": {"agree_rate": float(agree.mean()), "agree_acc": float((te["label"][agree] == yt[agree]).mean()),
                      "split_majority_acc": float((majority[~agree] == yt[~agree]).mean()),
                      "svm_acc_on_split": float(correct[~agree].mean())},
        "operating_points": points,
        "paper": PAPER,
    }
    (ROOT / "results" / "router_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pd.DataFrame({
        "sample_id": d["test"]["sample_id"], "component": units, "gold": np.array(CLASSES)[yt],
        "svm_label": np.array(CLASSES)[te["label"]], "bilstm_label": np.array(CLASSES)[te["preds"][1]],
        "marbert_label": np.array(CLASSES)[te["preds"][2]], "agree_3of3": agree,
        "p_svm": p_svm["test"][:, 1], "p_bilstm": p_bil["test"][:, 1], "p_marbert": p_mar["test"][:, 1],
        "novelty": nov["test"], "novelty_category": te["category"], "trust": te["trust"],
        "decision_80": np.where(te["accept80"], "accept", "escalate"),
    }).to_csv(ROOT / "results" / "router_test_decisions.csv", index=False)
    print(json.dumps({k: res[k] for k in ["per_model_test", "agreement", "novelty_share_test"]}, indent=1))
    for k, v in points.items():
        print(f"{k:14s} coverage {v['coverage']:.4f} sel.acc {v['selective_accuracy']:.4f}")


if __name__ == "__main__":
    main()
