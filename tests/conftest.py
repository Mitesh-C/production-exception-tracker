import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COLUMNS = ["date", "machine_id", "shift", "units_produced", "units_rejected", "downtime_minutes"]


@pytest.fixture
def write_csv(tmp_path):
    """Write rows (list of dicts/lists matching COLUMNS) to a temp CSV and return its path."""
    def _write(rows, columns=COLUMNS, name="data.csv"):
        path = tmp_path / name
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
        return path
    return _write


def row(date="2024-01-15", machine="M1", shift="Day", produced=100, rejected=1, downtime=0):
    return [date, machine, shift, produced, rejected, downtime]
