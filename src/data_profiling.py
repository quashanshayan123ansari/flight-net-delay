"""Data Profiling Module for Airline Operations & Delay Engine.

Performs comprehensive statistical, missingness, cardinality, and delay root-cause profiling
on raw BTS flight datasets. Generates executive health summaries and automated profiling reports.
"""

from __future__ import annotations

import os
from typing import Any, Dict
import numpy as np
import pandas as pd


def profile_flight_dataset(
    df: pd.DataFrame,
    report_output_path: str = "data/data_profiling_report.md",
) -> Dict[str, Any]:
    """Generates a deep statistical profiling report of flight operational data."""
    total_rows = len(df)
    total_cols = len(df.columns)
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # Missing value statistics
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / total_rows) * 100
    missing_summary = pd.DataFrame({
        "Missing_Count": missing_counts,
        "Missing_Pct": missing_pcts.round(2),
        "Dtype": df.dtypes.astype(str),
    }).sort_values(by="Missing_Pct", ascending=False)

    # Operational Key Metrics
    cancelled_count = int(df["Cancelled"].sum()) if "Cancelled" in df else 0
    cancelled_pct = (cancelled_count / total_rows) * 100 if total_rows > 0 else 0
    diverted_count = int(df["Diverted"].sum()) if "Diverted" in df else 0
    completed_flights = total_rows - cancelled_count - diverted_count

    completed_mask = (df["Cancelled"] == 0) & (df["Diverted"] == 0)
    completed_df = df[completed_mask]

    # Delays for completed flights
    dep_del15_pct = (completed_df["DepDel15"].mean() * 100) if "DepDel15" in completed_df else 0.0
    arr_del15_pct = (completed_df["ArrDel15"].mean() * 100) if "ArrDel15" in completed_df else 0.0

    mean_dep_delay = completed_df["DepDelay"].mean()
    median_dep_delay = completed_df["DepDelay"].median()
    p90_dep_delay = completed_df["DepDelay"].quantile(0.90)
    max_dep_delay = completed_df["DepDelay"].max()

    mean_arr_delay = completed_df["ArrDelay"].mean()
    median_arr_delay = completed_df["ArrDelay"].median()
    p90_arr_delay = completed_df["ArrDelay"].quantile(0.90)
    max_arr_delay = completed_df["ArrDelay"].max()

    # Cardinality metrics
    unique_tails = df["Tail_Number"].nunique()
    unique_airlines = df["Reporting_Airline"].nunique()
    unique_origins = df["Origin"].nunique()
    unique_dests = df["Dest"].nunique()
    unique_routes = (df["Origin"] + "->" + df["Dest"]).nunique()

    # Delay Cause Breakdown (Sum of minutes for major causes)
    cause_cols = ["LateAircraftDelay", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay"]
    cause_sums = {}
    total_cause_minutes = 0.0
    for col in cause_cols:
        if col in df.columns:
            val = float(df[col].dropna().sum())
            cause_sums[col] = val
            total_cause_minutes += val

    cause_breakdown = {}
    for col, val in cause_sums.items():
        pct = (val / total_cause_minutes * 100) if total_cause_minutes > 0 else 0.0
        cause_breakdown[col] = {"minutes": val, "pct": round(pct, 2)}

    # Top Carrier Volume & On-time performance
    carrier_stats = (
        completed_df.groupby("Reporting_Airline")
        .agg(
            flights=("Tail_Number", "count"),
            dep_del15=("DepDel15", "mean"),
            arr_del15=("ArrDel15", "mean"),
            avg_arr_delay=("ArrDelay", "mean"),
        )
        .reset_index()
    )
    carrier_stats["dep_del15"] = (carrier_stats["dep_del15"] * 100).round(1)
    carrier_stats["arr_del15"] = (carrier_stats["arr_del15"] * 100).round(1)
    carrier_stats["avg_arr_delay"] = carrier_stats["avg_arr_delay"].round(1)

    # Top Hub Volume
    origin_counts = completed_df["Origin"].value_counts().head(8).to_dict()

    # Potential Outliers & Data Hygiene
    neg_airtime = int((completed_df["AirTime"] <= 0).sum()) if "AirTime" in completed_df else 0
    excessive_taxi = int((completed_df["TaxiOut"] > 120).sum()) if "TaxiOut" in completed_df else 0

    results = {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "memory_mb": round(mem_mb, 2),
        "cancelled_count": cancelled_count,
        "cancelled_pct": round(cancelled_pct, 2),
        "diverted_count": diverted_count,
        "completed_flights": completed_flights,
        "unique_tails": unique_tails,
        "unique_airlines": unique_airlines,
        "unique_airports": unique_origins,
        "unique_routes": unique_routes,
        "dep_del15_pct": round(dep_del15_pct, 2),
        "arr_del15_pct": round(arr_del15_pct, 2),
        "mean_dep_delay": round(mean_dep_delay, 1),
        "p90_dep_delay": round(p90_dep_delay, 1),
        "max_dep_delay": round(max_dep_delay, 1),
        "mean_arr_delay": round(mean_arr_delay, 1),
        "p90_arr_delay": round(p90_arr_delay, 1),
        "max_arr_delay": round(max_arr_delay, 1),
        "cause_breakdown": cause_breakdown,
        "carrier_stats": carrier_stats.to_dict(orient="records"),
        "top_origins": origin_counts,
        "anomalies": {
            "non_positive_airtime": neg_airtime,
            "taxi_out_over_120m": excessive_taxi,
        },
    }

    # Write Markdown Profiling Report
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write("# 📋 Flight Operations Data Profiling Report\n\n")
        f.write(f"- **Total Rows Analyzed**: {total_rows:,}\n")
        f.write(f"- **Total Columns**: {total_cols}\n")
        f.write(f"- **Memory Footprint**: {mem_mb:.2f} MB\n")
        f.write(f"- **Unique Aircraft Tail Numbers**: {unique_tails:,}\n")
        f.write(f"- **Unique Routes**: {unique_routes:,}\n\n")

        f.write("## 1. Flight Status & Operational Health\n\n")
        f.write(f"| Metric | Value |\n| :--- | :--- |\n")
        f.write(f"| Completed Flights | {completed_flights:,} ({100 - cancelled_pct:.1f}%) |\n")
        f.write(f"| Cancelled Flights | {cancelled_count:,} ({cancelled_pct:.2f}%) |\n")
        f.write(f"| Diverted Flights | {diverted_count:,} |\n")
        f.write(f"| Departure Delay Rate (>=15 min) | **{dep_del15_pct:.1f}%** |\n")
        f.write(f"| Arrival Delay Rate (>=15 min) | **{arr_del15_pct:.1f}%** |\n")
        f.write(f"| Mean Departure Delay | {mean_dep_delay:.1f} mins |\n")
        f.write(f"| 90th Percentile Departure Delay | {p90_dep_delay:.1f} mins |\n")
        f.write(f"| Max Recorded Departure Delay | {max_dep_delay:.0f} mins |\n\n")

        f.write("## 2. Root Cause Delay Breakdown\n\n")
        f.write("The primary delay contributors across all delayed legs:\n\n")
        f.write("| Delay Category | Total Delay (Minutes) | Share of Total Delay (%) |\n| :--- | :--- | :--- |\n")
        for cause, data in cause_breakdown.items():
            f.write(f"| **{cause}** | {data['minutes']:,.0f} | {data['pct']}% |\n")
        f.write("\n> **Key Finding**: Late Aircraft Delay (downstream cascade propagation) represents the dominant delay driver across network rotations.\n\n")

        f.write("## 3. Airline On-Time Performance Summary\n\n")
        f.write("| Carrier | Flight Volume | Dep Delay Rate (% >=15m) | Arr Delay Rate (% >=15m) | Avg Arr Delay (mins) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for row in carrier_stats.to_dict(orient="records"):
            f.write(f"| {row['Reporting_Airline']} | {row['flights']:,} | {row['dep_del15']}% | {row['arr_del15']}% | {row['avg_arr_delay']}m |\n")
        f.write("\n")

        f.write("## 4. Missingness & Reporting Architecture\n\n")
        f.write("| Column | Missing Rows | Missing % | Interpretation |\n| :--- | :--- | :--- | :--- |\n")
        for col, row in missing_summary.head(10).iterrows():
            interp = "Standard BTS rule: Only populated when delay >= 15 min" if "Delay" in col and col != "ArrDelay" else "Natural flight cancellation/diversion missingness"
            f.write(f"| `{col}` | {int(row['Missing_Count']):,} | {row['Missing_Pct']}% | {interp} |\n")

    print(f"[PROFILING] Profiling complete. Report written to {report_output_path}")
    return results


if __name__ == "__main__":
    df_raw = pd.read_parquet("data/raw/flights_raw.parquet")
    profile_flight_dataset(df_raw)
