"""Rebuild the paper's leakage-safe connected-components split from source_binary (reconstruction, not the original code)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakaudit import data_dir  # noqa: E402
from leakaudit.split import MIN_EVIDENCE_CHARS, assign, components  # noqa: E402

SIZES = {"test": 4000, "validation": 2000, "train": 14000}  # paper Table 1 targets
PAPER = {"pool": 25649, "dedup": 20222, "components": 16591, "train": 13999, "validation": 2000, "test": 4000, "unassigned": 223}
SEED = 42


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    files = [data_dir() / f"source_binary_{s}.csv" for s in ["train", "validation", "test"]]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pool = len(df)
    df = df.drop_duplicates(subset="claim_norm", keep="first").reset_index(drop=True)
    df["component"] = components(df["claim_norm"], df["evidence_norm"])
    df["split"] = assign(df["component"].to_numpy(), SIZES, seed=SEED)

    parts = {p: df[df["split"] == p] for p in SIZES}
    ev = df["evidence_norm"].fillna("").astype(str)
    long_ev = df.assign(ev=ev)[ev.str.len() >= MIN_EVIDENCE_CHARS]
    overlap = {}
    for a in SIZES:
        for b in SIZES:
            if a < b:
                overlap[f"{a}|{b}"] = {
                    "claims": len(set(parts[a]["claim_norm"]) & set(parts[b]["claim_norm"])),
                    "components": len(set(parts[a]["component"]) & set(parts[b]["component"])),
                    "evidence_40plus": len(set(long_ev[long_ev["split"] == a]["ev"]) & set(long_ev[long_ev["split"] == b]["ev"])),
                }
    sizes = df["component"].value_counts()
    summary = {
        "inputs": {f.name: sha256(f) for f in files},
        "seed": SEED,
        "min_evidence_chars": MIN_EVIDENCE_CHARS,
        "rebuilt": {
            "pool": pool,
            "dedup": len(df),
            "components": int(df["component"].nunique()),
            "largest_component": int(sizes.max()),
            **{p: len(parts[p]) for p in SIZES},
            "unassigned": int((df["split"] == "").sum()),
        },
        "paper": PAPER,
        "labels": {p: parts[p]["binary_label"].value_counts().to_dict() for p in SIZES},
        "sources_test": parts["test"]["source"].value_counts().to_dict(),
        "overlap_must_be_zero": overlap,
    }
    assert all(v == 0 for o in overlap.values() for v in o.values()), overlap

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    df[["sample_id", "component", "split"]].to_csv(ROOT / "data" / "cc_split_manifest.csv", index=False)
    (ROOT / "results" / "cc_split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["rebuilt", "paper", "labels", "sources_test"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
