"""Selective-prediction router: calibration, novelty, trust score, threshold selection."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Scalar T minimising validation NLL of softmax(logits / T)."""
    def nll(log_t: float) -> float:
        return -log_softmax(logits / np.exp(log_t), axis=1)[np.arange(len(y)), y].mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def calibrated(logits: np.ndarray, t: float) -> np.ndarray:
    return softmax(logits / t, axis=1)


def novelty(query: np.ndarray, train: np.ndarray, k: int = 10, chunk: int = 1024) -> np.ndarray:
    """Mean cosine distance to the k nearest train rows (inputs L2-normalised)."""
    out = []
    for i in range(0, len(query), chunk):
        sims = query[i:i + chunk] @ train.T
        top = np.partition(sims, -k, axis=1)[:, -k:]
        out.append(1 - top.mean(axis=1))
    return np.concatenate(out)


def trust(probs: list[np.ndarray], label: np.ndarray, high_novelty: np.ndarray) -> np.ndarray:
    """Mean calibrated probability of the returned label across models; High novelty never accepted."""
    t = np.mean([p[np.arange(len(label)), label] for p in probs], axis=0)
    return np.where(high_novelty, -np.inf, t)


def choose_tau(t_val: np.ndarray, target: float) -> float:
    """Threshold giving ~target coverage on validation (accept t >= tau)."""
    if target >= 1:
        return -np.inf
    return float(np.quantile(t_val, 1 - target))


def selective(accept: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    return {"coverage": float(accept.mean()),
            "selective_accuracy": float(correct[accept].mean()) if accept.any() else float("nan")}
