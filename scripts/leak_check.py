"""Check paper claim: 57.2% of claim-grouped test rows reuse training evidence verbatim."""
import sys, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from leakaudit import data_dir
D = sys.argv[1] if len(sys.argv) > 1 else str(data_dir())
cols = ["claim_group", "evidence_group", "evidence_norm", "source", "binary_label", "three_class_label"]
for task in ["source_binary", "source_three_class", "combined_binary"]:
    tr, va, te = (pd.read_csv(f"{D}/{task}_{s}.csv", usecols=cols) for s in ["train", "validation", "test"])
    ev_tr = set(tr["evidence_norm"].dropna())
    nonempty = te["evidence_norm"].fillna("").str.strip() != ""
    leak = te["evidence_norm"].isin(ev_tr) & nonempty
    leak_g = te["evidence_group"].isin(set(tr["evidence_group"])) & nonempty
    claim_leak = te["claim_group"].isin(set(tr["claim_group"]))
    print(f"{task}: train/val/test = {len(tr)}/{len(va)}/{len(te)} | "
          f"test evidence verbatim in train: {leak.mean():.1%} (by evidence_group {leak_g.mean():.1%}) | "
          f"claim overlap {claim_leak.mean():.1%} | empty evidence {(~nonempty).mean():.1%}")
    if task == "source_binary":
        print("  test leak by source:", (leak.groupby(te['source']).mean().round(3)).to_dict(),
              "| test rows by source:", te["source"].value_counts().to_dict())
