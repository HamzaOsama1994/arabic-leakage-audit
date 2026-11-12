from __future__ import annotations

import os
from pathlib import Path

# Harmonised inputs (source_binary_{train,validation,test}.csv); see README "Inputs".
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "inputs"


def data_dir() -> Path:
    return Path(os.environ.get("LEAKAUDIT_DATA_DIR", DEFAULT_DATA_DIR))
