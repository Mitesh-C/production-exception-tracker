from pathlib import Path
from datetime import date
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_FILE = Path(__file__).parent / "synthetic_production_2023_2025.csv"
DEFAULT_ALERT_THRESHOLD = 5.0

st.set_page_config(
    page_title="Production Exception Tracker",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Production Exception Tracker")
st.caption("Synthetic demonstration data · 1 Jan 2023 – 31 Dec 2025")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "date", "machine_id", "shift",
        "units_produced", "units_rejected", "downtime_minutes",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_columns = ["units_produced", "units_rejected", "downtime_minutes"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df["date"].isna().any():
        raise ValueError("Dataset contains invalid dates.")
    if df[numeric_columns].isna().any().any():
        raise ValueError("Dataset contains missing or non-numeric measures.")
    if (df["units_produced"] <= 0).any():
        raise ValueError("Produced units must be greater than zero.")
    if (df["units_rejected"] < 0).any():
        raise ValueError("Rejected units cannot be negative.")
    if (df["units_rejected"] > df["units_produced"]).any():
        raise ValueError("Rejected units cannot exceed produced units.")
    if (df["downtime_minutes"] < 0).any():
        raise ValueError("Downtime cannot be negative.")

    df["rejection_rate_pct"] = (
        df["units_rejected"] / df["units_produced"] * 100
    )
    return df

if not DATA_FILE.exists():
    st.error(
        "Dataset not found. Place synthetic_production_2023_2025.csv "
        "in the same folder as dashboard.py."
    )
    st.stop()

try:
    data = load_data(str(DATA_FILE))
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

st.sidebar.header("Dashboard filters")
min_date = data["date"].min().date()
max_date = data["date"].max().date()

# Quick time presets plus a custom date range.
period_choice = st.sidebar.radio(
    "Time period",
    options=["All time", "Year to date", "Choose a year", "Custom range"],
    index=0,
)

if period_choice == "All time":
    date_range = (min_date, max_date)
elif period_choice == "Year to date":
    today = pd.Timestamp.today().date()
    effective_end = min(today, max_date)
    effective_start = max(date(effective_end.year, 1, 1), min_date)
    date_range = (effective_start, effective_end)
elif period_choice == "Choose a year":
    available_years = sorted(data["date"].dt.year.unique().tolist())
    chosen_year = st.sidebar.selectbox("Year", available_years, index=len(available_years) - 1)
    year_start = max(date(int(chosen_year), 1, 1), min_date)
    year_end = min(date(int(chosen_year), 12, 31), max_date)
    date_range = (year_start, year_end)
else:
    date_range = st.sidebar.date_input(
        "Custom date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

machines = sorted(data["machine_id"].unique())
selected_machines = st.sidebar.multiselect(
    "Machines", machines, default=machines
)
shifts = sorted(data["shift"].unique())
selected_shifts = st.sidebar.multiselect(
    "Shifts", shifts, default=shifts
)
threshold = st.sidebar.number_input(
    "Rejection alert threshold (%)",
    min_value=0.0,
    max_value=100.0,
    value=DEFAULT_ALERT_THRESHOLD,
    step=0.5,
)
st.sidebar.caption("Fictional data for demonstration; not a real factory report.")

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Choose both a start date and an end date to view the dashboard.")
    st.stop()

start_date, end_date = date_range
if start_date > end_date:
    st.error("Start date must be on or before end date.")
    st.stop()

filtered = data[
    data["date"].dt.date.between(start_date, end_date)
    & data["machine_id"].isin(selected_machines)
    & data["shift"].isin(selected_shifts)
].copy()

if filtered.empty:
    st.warning("No records match these filters. Select at least one machine and shift.")
    st.stop()

total_produced = int(filtered["units_produced"].sum())
total_rejected = int(filtered["units_rejected"].sum())
overall_rate = total_rejected / total_produced * 100 if total_produced else 0
total_downtime = int(filtered["downtime_minutes"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Units produced", f"{total_produced:,}")
k2.metric("Units rejected", f"{total_rejected:,}")
k3.metric("Overall rejection rate", f"{overall_rate:.2f}%")
k4.metric("Downtime", f"{total_downtime:,} min")

st.divider()

left, right = st.columns(2)

with left:
    monthly = (
        filtered.assign(month=filtered["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(
            units_produced=("units_produced", "sum"),
            units_rejected=("units_rejected", "sum"),
        )
    )
    monthly_long = monthly.melt(
        id_vars="month",
        value_vars=["units_produced", "units_rejected"],
        var_name="measure",
        value_name="units",
    )
    fig_monthly = px.line(
        monthly_long,
        x="month",
        y="units",
        color="measure",
        markers=True,
        title="Monthly production vs rejected units",
        labels={"month": "Month", "units": "Units", "measure": "Measure"},
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

with right:
    machine_summary = (
        filtered.groupby("machine_id", as_index=False)
        .agg(
            produced=("units_produced", "sum"),
            rejected=("units_rejected", "sum"),
        )
    )
    machine_summary["rejection_rate_pct"] = (
        machine_summary["rejected"] / machine_summary["produced"] * 100
    )
    fig_machine = px.bar(
        machine_summary.sort_values("rejection_rate_pct", ascending=False),
        x="machine_id",
        y="rejection_rate_pct",
        title="Rejection rate by machine",
        labels={"machine_id": "Machine", "rejection_rate_pct": "Rejection rate (%)"},
        text_auto=".2f",
    )
    fig_machine.add_hline(
        y=threshold,
        line_dash="dash",
        annotation_text=f"Alert threshold: {threshold:g}%",
    )
    st.plotly_chart(fig_machine, use_container_width=True)

left, right = st.columns(2)

with left:
    shift_summary = (
        filtered.groupby("shift", as_index=False)
        .agg(
            produced=("units_produced", "sum"),
            rejected=("units_rejected", "sum"),
        )
    )
    shift_summary["rejection_rate_pct"] = (
        shift_summary["rejected"] / shift_summary["produced"] * 100
    )
    fig_shift = px.bar(
        shift_summary.sort_values("rejection_rate_pct", ascending=False),
        x="shift",
        y="rejection_rate_pct",
        title="Rejection rate by shift",
        labels={"shift": "Shift", "rejection_rate_pct": "Rejection rate (%)"},
        text_auto=".2f",
    )
    fig_shift.add_hline(y=threshold, line_dash="dash")
    st.plotly_chart(fig_shift, use_container_width=True)

with right:
    filtered["year"] = filtered["date"].dt.year
    yearly = (
        filtered.groupby("year", as_index=False)
        .agg(
            produced=("units_produced", "sum"),
            rejected=("units_rejected", "sum"),
        )
    )
    yearly["rejection_rate_pct"] = yearly["rejected"] / yearly["produced"] * 100
    fig_year = px.line(
        yearly,
        x="year",
        y="rejection_rate_pct",
        markers=True,
        title="Year-over-year rejection rate",
        labels={"year": "Year", "rejection_rate_pct": "Rejection rate (%)"},
        text=yearly["rejection_rate_pct"].map(lambda x: f"{x:.2f}%"),
    )
    st.plotly_chart(fig_year, use_container_width=True)

st.subheader("Production exceptions")
st.caption(
    f"Records with rejection rates greater than {threshold:g}%. "
    "These are flagged records, not confirmed root causes or separate incidents."
)
exceptions = filtered[filtered["rejection_rate_pct"] > threshold].copy()
exceptions["date"] = exceptions["date"].dt.strftime("%Y-%m-%d")
exceptions = exceptions[
    ["date", "machine_id", "shift", "units_produced",
     "units_rejected", "rejection_rate_pct", "downtime_minutes"]
].sort_values("rejection_rate_pct", ascending=False)
exceptions["rejection_rate_pct"] = exceptions["rejection_rate_pct"].round(2)

st.metric("Flagged records", f"{len(exceptions):,}")
st.dataframe(exceptions, use_container_width=True, hide_index=True)

csv_bytes = exceptions.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered exceptions (CSV)",
    data=csv_bytes,
    file_name="production_exceptions.csv",
    mime="text/csv",
)

with st.expander("About this demo"):
    st.write(
        "This dashboard uses synthetic data created for software development practice. "
        "Patterns and alert thresholds are illustrative only and should not be treated "
        "as validated industrial benchmarks."
    )
