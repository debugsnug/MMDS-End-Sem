"""Append-only CSV log of every run (resumable): a run whose run_id is already logged is skipped.

All statistics and plots are computed from this log and from the saved score files, never from
in-memory state, so a Colab disconnect only costs the run that was in flight.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Set

import pandas as pd


class RunLog:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    def has(self, run_id: str) -> bool:
        return not self.df.empty and run_id in set(self.df["run_id"])

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        if not self.has(run_id):
            return None
        return self.df[self.df.run_id == run_id].iloc[-1].to_dict()

    def append(self, row: Dict[str, Any]) -> None:
        """Add one run. New columns are allowed (the file is rewritten with the union of columns)."""
        row = {k: (json.dumps(v) if isinstance(v, (dict, list, tuple)) else v) for k, v in row.items()}
        if "run_id" not in row:
            raise KeyError("every logged run needs a run_id")
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.path, index=False)

    def run_ids(self) -> Set[str]:
        return set(self.df["run_id"]) if not self.df.empty else set()
