import numpy as np
import pandas as pd

from leakaudit.split import assign, components

LONG = "x" * 40


def test_rows_sharing_claim_or_long_evidence_join():
    claims = pd.Series(["a", "a", "b", "c", "d"])
    evidence = pd.Series(["e1", "e2", LONG, LONG, "short"])
    comp = components(claims, evidence)
    assert comp[0] == comp[1]           # same claim
    assert comp[2] == comp[3]           # same long evidence
    assert len(set(comp)) == 3          # {0,1} {2,3} {4}


def test_short_evidence_does_not_link():
    comp = components(pd.Series(["a", "b"]), pd.Series(["short", "short"]))
    assert comp[0] != comp[1]


def test_assign_keeps_components_whole_and_respects_targets():
    comp = np.array([0, 0, 1, 2, 2, 2, 3])
    parts = assign(comp, {"test": 2, "val": 1, "train": 4}, seed=0)
    for c in np.unique(comp):
        assert len(set(parts[comp == c])) == 1
    assert (parts == "test").sum() <= 2 and (parts == "val").sum() <= 1
