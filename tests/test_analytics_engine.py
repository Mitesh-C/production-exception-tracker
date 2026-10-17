"""Tests for analytics_engine.py (run against the unmodified implementation)."""
import pandas as pd
import pytest

import analytics_engine as ae
from conftest import COLUMNS, row


# ---------- A. CSV loading and validation ----------

def test_valid_data_loads_with_row_rejection_rate(write_csv):
    path = write_csv([row(produced=200, rejected=5), row(machine="M2", produced=50, rejected=0)])
    df = ae.load_data(path)
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["rejection_rate_pct"].tolist() == pytest.approx([2.5, 0.0])


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ae.load_data(tmp_path / "nope.csv")


def test_missing_required_column_rejected(write_csv):
    cols = [c for c in COLUMNS if c != "shift"]
    path = write_csv([[r for i, r in enumerate(row()) if COLUMNS[i] != "shift"]], columns=cols)
    with pytest.raises(ValueError, match="Missing columns.*shift"):
        ae.load_data(path)


def test_invalid_date_rejected(write_csv):
    with pytest.raises(ValueError, match="invalid"):
        ae.load_data(write_csv([row(date="not-a-date")]))


def test_missing_date_rejected(write_csv):
    with pytest.raises(ValueError, match="invalid"):
        ae.load_data(write_csv([row(date=None)]))


@pytest.mark.parametrize("col_idx", [3, 4, 5])
def test_non_numeric_value_rejected(write_csv, col_idx):
    r = row()
    r[col_idx] = "abc"
    with pytest.raises(ValueError, match="invalid"):
        ae.load_data(write_csv([r]))


@pytest.mark.parametrize("col_idx", [3, 4, 5])
def test_missing_numeric_value_rejected(write_csv, col_idx):
    r = row()
    r[col_idx] = None
    with pytest.raises(ValueError, match="invalid"):
        ae.load_data(write_csv([r]))


@pytest.mark.parametrize("produced", [0, -10])
def test_zero_or_negative_production_rejected(write_csv, produced):
    with pytest.raises(ValueError, match="units_produced"):
        ae.load_data(write_csv([row(produced=produced, rejected=0)]))


def test_negative_rejected_rejected(write_csv):
    with pytest.raises(ValueError, match="Rejected units"):
        ae.load_data(write_csv([row(produced=100, rejected=-1)]))


def test_rejected_greater_than_produced_rejected(write_csv):
    with pytest.raises(ValueError, match="Rejected units"):
        ae.load_data(write_csv([row(produced=100, rejected=101)]))


def test_rejected_equal_to_produced_allowed(write_csv):
    df = ae.load_data(write_csv([row(produced=100, rejected=100)]))
    assert df["rejection_rate_pct"].iloc[0] == pytest.approx(100.0)


def test_negative_downtime_rejected(write_csv):
    with pytest.raises(ValueError, match="Downtime"):
        ae.load_data(write_csv([row(downtime=-1)]))


def test_one_bad_row_rejects_whole_file(write_csv):
    with pytest.raises(ValueError):
        ae.load_data(write_csv([row(), row(produced=0, rejected=0)]))


# ---------- B. Overall KPIs (computed inline in main(); verified via its report output) ----------

@pytest.fixture
def run_main(tmp_path, monkeypatch, write_csv, capsys):
    """Run ae.main() against a temp CSV and temp report dir; return report tables."""
    def _run(rows):
        data = write_csv(rows)
        out = tmp_path / "out_reports"
        monkeypatch.setattr(ae, "DATA_FILE", data)
        monkeypatch.setattr(ae, "OUTPUT_DIR", out)
        ae.main()
        capsys.readouterr()
        return {p.stem: pd.read_csv(p) for p in out.glob("*.csv")}
    return _run


UNEQUAL = [
    row("2024-01-10", "M1", "Day", produced=1000, rejected=10, downtime=30),    # 1%
    row("2024-01-11", "M2", "Night", produced=100, rejected=50, downtime=15),   # 50%
]


def test_overall_kpis_use_weighted_totals(run_main):
    kpi = run_main(UNEQUAL)["overall_kpis"].iloc[0]
    assert kpi["records"] == 2
    assert kpi["total_produced"] == 1100
    assert kpi["total_rejected"] == 60
    assert kpi["total_downtime_minutes"] == 45
    assert kpi["overall_rejection_rate_pct"] == pytest.approx(round(60 / 1100 * 100, 2))


def test_overall_rate_is_not_unweighted_mean_of_row_rates(run_main):
    kpi = run_main(UNEQUAL)["overall_kpis"].iloc[0]
    unweighted = (1.0 + 50.0) / 2  # 25.5
    assert kpi["overall_rejection_rate_pct"] != pytest.approx(unweighted, abs=1)
    assert kpi["overall_rejection_rate_pct"] == pytest.approx(5.45)


# ---------- C. Machine and shift summaries ----------

GROUPED = [
    row("2024-01-01", "M1", "Day", produced=1000, rejected=10),   # M1/Day 1%
    row("2024-01-02", "M1", "Night", produced=100, rejected=20, downtime=5),  # M1/Night 20%
    row("2024-01-03", "M2", "Day", produced=200, rejected=30, downtime=7),    # M2/Day 15%
    row("2024-01-04", "M2", "Day", produced=200, rejected=0, downtime=3),     # M2/Day 0%
]


def _by(df, col, key):
    return df[df[col] == key].iloc[0]


def test_machine_aggregate_totals_and_weighted_rate():
    df = ae.aggregate(_df(GROUPED), "machine_id")
    m1, m2 = _by(df, "machine_id", "M1"), _by(df, "machine_id", "M2")
    assert (m1.total_produced, m1.total_rejected, m1.total_downtime_minutes) == (1100, 30, 5)
    assert (m2.total_produced, m2.total_rejected, m2.total_downtime_minutes) == (400, 30, 10)
    assert m1.rejection_rate_pct == pytest.approx(30 / 1100 * 100)
    assert m2.rejection_rate_pct == pytest.approx(7.5)
    # M1 mean of row rates would be (1+20)/2 = 10.5, clearly different from weighted 2.727
    assert m1.rejection_rate_pct != pytest.approx(10.5)


def test_shift_aggregate_keeps_shifts_separate():
    df = ae.aggregate(_df(GROUPED), "shift")
    assert set(df["shift"]) == {"Day", "Night"}
    day, night = _by(df, "shift", "Day"), _by(df, "shift", "Night")
    assert (day.total_produced, day.total_rejected) == (1400, 40)
    assert day.rejection_rate_pct == pytest.approx(40 / 1400 * 100)
    assert (night.total_produced, night.total_rejected) == (100, 20)
    assert night.rejection_rate_pct == pytest.approx(20.0)


def test_aggregate_sorted_by_rate_descending():
    df = ae.aggregate(_df(GROUPED), "shift")
    assert df["rejection_rate_pct"].is_monotonic_decreasing
    assert df["shift"].iloc[0] == "Night"


def test_machine_and_shift_groups_not_mixed():
    df = ae.aggregate(_df(GROUPED), "machine_id")
    assert set(df["machine_id"]) == {"M1", "M2"}
    assert "shift" not in df.columns
    assert df["total_produced"].sum() == 1500


def test_main_machine_and_shift_reports_rounded_and_sorted(run_main):
    reports = run_main(GROUPED)
    machine, shift = reports["machine_summary"], reports["shift_summary"]
    assert machine["machine_id"].tolist() == ["M2", "M1"]  # 7.5% > 2.73%
    assert machine["rejection_rate_pct"].tolist() == pytest.approx([7.5, 2.73])
    assert shift["shift"].tolist() == ["Night", "Day"]


# ---------- D. Monthly summaries ----------

MONTHLY = [
    row("2023-01-05", "M1", "Day", produced=100, rejected=10, downtime=10),
    row("2023-01-31", "M1", "Day", produced=300, rejected=10, downtime=20),
    row("2024-01-15", "M1", "Day", produced=200, rejected=2, downtime=5),
    row("2023-02-01", "M1", "Day", produced=50, rejected=5, downtime=1),
]


def _df(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_monthly_summary_values_and_year_separation(run_main):
    monthly = run_main(MONTHLY)["monthly_summary"].set_index("month")
    assert list(monthly.index) == ["2023-01", "2023-02", "2024-01"]  # Jan 2023 != Jan 2024
    jan23 = monthly.loc["2023-01"]
    assert (jan23.total_produced, jan23.total_rejected, jan23.total_downtime_minutes) == (400, 20, 30)
    assert jan23.rejection_rate_pct == pytest.approx(5.0)
    jan24 = monthly.loc["2024-01"]
    assert (jan24.total_produced, jan24.total_rejected, jan24.total_downtime_minutes) == (200, 2, 5)
    assert jan24.rejection_rate_pct == pytest.approx(1.0)
    assert monthly.loc["2023-02", "total_produced"] == 50  # month-end/start boundaries respected


def test_aggregate_monthly_uses_weighted_rate():
    df = _df(MONTHLY)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    out = ae.aggregate_monthly(df).set_index("month")
    # row rates in Jan 2023 are 10% and 3.33%; mean 6.67% vs weighted 5%
    assert out.loc["2023-01", "rejection_rate_pct"] == pytest.approx(5.0)


# ---------- E. Rejection exceptions (inline in main(); verified via rejection_exceptions.csv) ----------

def test_default_threshold_is_5_percent():
    assert ae.ALERT_THRESHOLD_PCT == 5.0


def test_exceptions_strictly_greater_than_threshold(run_main):
    rows = [
        row("2024-03-01", "M1", produced=100, rejected=4),     # 4%   below
        row("2024-03-02", "M2", produced=100, rejected=5),     # 5%   exactly -> not flagged
        row("2024-03-03", "M3", produced=100, rejected=6),     # 6%   flagged
        row("2024-03-04", "M4", produced=1000, rejected=51),   # 5.1% flagged
    ]
    ex = run_main(rows)["rejection_exceptions"]
    assert ex["machine_id"].tolist() == ["M3", "M4"]  # sorted desc by rate
    assert ex["date"].tolist() == ["2024-03-03", "2024-03-04"]
    assert ex["rejection_rate_pct"].tolist() == pytest.approx([6.0, 5.1])


def test_no_exceptions_yields_empty_report_with_header(run_main):
    reports = run_main([row(produced=100, rejected=5), row(produced=100, rejected=0)])
    ex = reports["rejection_exceptions"]
    assert len(ex) == 0
    assert "rejection_rate_pct" in ex.columns
