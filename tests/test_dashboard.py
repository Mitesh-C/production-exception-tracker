"""Dashboard tests using streamlit.testing.v1.AppTest.

dashboard.py hard-codes its CSV path next to itself, so each test copies the unmodified
dashboard.py into a temp folder beside a tiny CSV. The real dashboard.py and CSV are only read.
"""
import shutil

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from conftest import COLUMNS, ROOT, row

DASHBOARD = ROOT / "dashboard.py"
CSV_NAME = "synthetic_production_2023_2025.csv"

ROWS = [
    row("2024-02-01", "M1", "Day", produced=100, rejected=3),    # 3%
    row("2024-02-02", "M1", "Day", produced=100, rejected=5),    # 5% exactly
    row("2024-02-03", "M2", "Night", produced=100, rejected=8),  # 8%
    row("2024-02-04", "M2", "Night", produced=100, rejected=12), # 12%
]


@pytest.fixture
def app(tmp_path):
    shutil.copy(DASHBOARD, tmp_path / "dashboard.py")
    pd.DataFrame(ROWS, columns=COLUMNS).to_csv(tmp_path / CSV_NAME, index=False)
    at = AppTest.from_file(str(tmp_path / "dashboard.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def flagged(at):
    return int(next(m.value for m in at.metric if m.label == "Flagged records"))


def table(at):
    return at.dataframe[0].value


def test_default_threshold_is_5_percent(app):
    box = app.sidebar.number_input[0]
    assert box.label == "Rejection alert threshold (%)"
    assert box.value == 5.0


def test_default_threshold_flags_strictly_above_5(app):
    assert flagged(app) == 2
    assert table(app)["rejection_rate_pct"].tolist() == [12.0, 8.0]  # 5% row not flagged


def test_raising_threshold_reduces_exceptions(app):
    app.sidebar.number_input[0].set_value(10.0).run()
    assert not app.exception
    assert flagged(app) == 1
    assert table(app)["rejection_rate_pct"].tolist() == [12.0]


def test_lowering_threshold_adds_exceptions(app):
    app.sidebar.number_input[0].set_value(2.5).run()
    assert flagged(app) == 4
    assert len(table(app)) == 4


def test_threshold_zero_flags_all_nonzero_rates_and_hundred_flags_none(app):
    app.sidebar.number_input[0].set_value(0.0).run()
    assert flagged(app) == 4
    app.sidebar.number_input[0].set_value(100.0).run()
    assert flagged(app) == 0
    assert len(table(app)) == 0


def test_threshold_shown_in_caption(app):
    app.sidebar.number_input[0].set_value(7.5).run()
    assert any("greater than 7.5%" in c.value for c in app.caption)


def test_kpis_use_weighted_totals(app):
    vals = {m.label: m.value for m in app.metric}
    assert vals["Units produced"] == "400"
    assert vals["Units rejected"] == "28"
    assert vals["Overall rejection rate"] == "7.00%"


def test_machine_filter_limits_exceptions(app):
    app.sidebar.multiselect[0].unselect("M2").run()  # machines
    assert flagged(app) == 0
