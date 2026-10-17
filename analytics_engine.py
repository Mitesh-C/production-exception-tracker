from pathlib import Path
import pandas as pd

DATA_FILE = Path("synthetic_production_2023_2025.csv")
OUTPUT_DIR = Path("reports")
ALERT_THRESHOLD_PCT = 5.0

def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Put the CSV beside this script.")
    df = pd.read_csv(path)
    required = {"date", "machine_id", "shift", "units_produced", "units_rejected", "downtime_minutes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["units_produced", "units_rejected", "downtime_minutes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["date", "units_produced", "units_rejected", "downtime_minutes"]].isna().any().any():
        raise ValueError("Found missing or invalid dates/numeric values.")
    if (df["units_produced"] <= 0).any():
        raise ValueError("units_produced must be greater than zero.")
    if (df["units_rejected"] < 0).any() or (df["units_rejected"] > df["units_produced"]).any():
        raise ValueError("Rejected units must be between zero and produced units.")
    if (df["downtime_minutes"] < 0).any():
        raise ValueError("Downtime cannot be negative.")
    df["rejection_rate_pct"] = df["units_rejected"] / df["units_produced"] * 100
    return df

def aggregate(df, group_column):
    result = df.groupby(group_column, as_index=False).agg(
        total_produced=("units_produced", "sum"),
        total_rejected=("units_rejected", "sum"),
        total_downtime_minutes=("downtime_minutes", "sum"),
    )
    result["rejection_rate_pct"] = result["total_rejected"] / result["total_produced"] * 100
    return result.sort_values("rejection_rate_pct", ascending=False)

def main():
    df = load_data(DATA_FILE)
    OUTPUT_DIR.mkdir(exist_ok=True)
    total_produced = int(df["units_produced"].sum())
    total_rejected = int(df["units_rejected"].sum())
    overall = pd.DataFrame([{
        "records": len(df),
        "total_produced": total_produced,
        "total_rejected": total_rejected,
        "overall_rejection_rate_pct": round(total_rejected / total_produced * 100, 2),
        "total_downtime_minutes": int(df["downtime_minutes"].sum()),
    }])
    by_machine = aggregate(df, "machine_id").round(2)
    by_shift = aggregate(df, "shift").round(2)
    monthly = df.assign(month=df["date"].dt.to_period("M").astype(str))
    monthly = aggregate_monthly(monthly)
    exceptions = df[df["rejection_rate_pct"] > ALERT_THRESHOLD_PCT].copy()
    exceptions["date"] = exceptions["date"].dt.strftime("%Y-%m-%d")
    exceptions = exceptions.sort_values("rejection_rate_pct", ascending=False)

    outputs = {
        "overall_kpis.csv": overall,
        "machine_summary.csv": by_machine,
        "shift_summary.csv": by_shift,
        "monthly_summary.csv": monthly,
        "rejection_exceptions.csv": exceptions,
    }
    for filename, table in outputs.items():
        table.to_csv(OUTPUT_DIR / filename, index=False)

    print("\n=== OVERALL KPIs ===")
    print(overall.to_string(index=False))
    print("\n=== BY MACHINE ===")
    print(by_machine.to_string(index=False))
    print("\n=== BY SHIFT ===")
    print(by_shift.to_string(index=False))
    print(f"\nRows exceeding {ALERT_THRESHOLD_PCT:.1f}% rejection rate: {len(exceptions):,}")
    print(f"CSV reports saved in: {OUTPUT_DIR.resolve()}")

def aggregate_monthly(df):
    result = df.groupby("month", as_index=False).agg(
        total_produced=("units_produced", "sum"),
        total_rejected=("units_rejected", "sum"),
        total_downtime_minutes=("downtime_minutes", "sum"),
    )
    result["rejection_rate_pct"] = result["total_rejected"] / result["total_produced"] * 100
    return result

if __name__ == "__main__":
    main()
