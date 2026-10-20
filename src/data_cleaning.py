"""Data Cleaning Module for Airline Operations & Delay Engine.

Performs deterministic data cleansing, missing value imputation (BTS delay cause rules),
timestamp sanitization (HHMM to minute-of-day and ISO timestamps with midnight crossover),
and anomaly filtering.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def _hhmm_to_minutes(hhmm: pd.Series) -> pd.Series:
    """Converts integer or float HHMM (e.g., 830, 2359) to minutes from midnight (0..1439)."""
    valid = hhmm.notnull()
    s = hhmm.fillna(0).astype(int)
    # Handle edge case where 2400 represents midnight
    hours = (s // 100) % 24
    mins = s % 100
    total_mins = hours * 60 + mins
    return total_mins.where(valid, np.nan)


def _format_hhmm_string(hhmm: pd.Series) -> pd.Series:
    """Formats numeric HHMM into zero-padded HH:MM string."""
    valid = hhmm.notnull()
    s = hhmm.fillna(0).astype(int)
    hours = (s // 100) % 24
    mins = s % 100
    res = hours.map("{:02d}".format) + ":" + mins.map("{:02d}".format)
    return res.where(valid, None)


def clean_flight_data(
    input_path: str = "data/raw/flights_raw.parquet",
    output_path: str = "data/processed/flights_cleaned.parquet",
) -> pd.DataFrame:
    """Loads raw BTS data and applies institutional cleaning and imputation rules."""
    print(f"[CLEANING] Loading raw dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    initial_rows = len(df)

    # 1. Clean String Identifiers
    for col in ["Tail_Number", "Reporting_Airline", "Origin", "Dest"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # Drop any records completely missing Tail_Number or Origin/Dest
    df = df[df["Tail_Number"].notnull() & (df["Tail_Number"] != "NONE") & (df["Tail_Number"] != "NAN")]
    df = df[df["Origin"].notnull() & df["Dest"].notnull()]

    # 2. Derive Flight Status Categorization
    conditions = [
        (df["Cancelled"] == 1),
        (df["Diverted"] == 1),
    ]
    choices = ["Cancelled", "Diverted"]
    df["Flight_Status"] = np.select(conditions, choices, default="Completed")

    # 3. BTS Delay Cause Imputation
    # BTS leaves delay breakdown as NaN when delay < 15 mins. For modeling, these are 0.0 minutes.
    delay_breakdown_cols = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
    for col in delay_breakdown_cols:
        if col in df.columns:
            # Impute NaN with 0.0 for completed/diverted flights
            df[col] = df[col].fillna(0.0)

    # 4. Standardize Delay Minutes
    if "DepDelay" in df.columns:
        df["DepDelayMinutes"] = df["DepDelay"].clip(lower=0).fillna(0.0)
    if "ArrDelay" in df.columns:
        df["ArrDelayMinutes"] = df["ArrDelay"].clip(lower=0).fillna(0.0)

    # 5. Timestamp Sanitization & Minute-of-Day Conversion
    df["CRSDepMinuteOfDay"] = _hhmm_to_minutes(df["CRSDepTime"])
    df["DepMinuteOfDay"] = _hhmm_to_minutes(df["DepTime"])
    df["CRSArrMinuteOfDay"] = _hhmm_to_minutes(df["CRSArrTime"])
    df["ArrMinuteOfDay"] = _hhmm_to_minutes(df["ArrTime"])

    df["CRSDepTime_Str"] = _format_hhmm_string(df["CRSDepTime"])
    df["DepTime_Str"] = _format_hhmm_string(df["DepTime"])
    df["CRSArrTime_Str"] = _format_hhmm_string(df["CRSArrTime"])
    df["ArrTime_Str"] = _format_hhmm_string(df["ArrTime"])

    # 6. Parse Full DateTime Timestamps with Overnight Rollover Handling
    # Scheduled Departure Timestamp
    base_dates = pd.to_datetime(df["FlightDate"])
    df["CRSDepTimestamp"] = base_dates + pd.to_timedelta(df["CRSDepMinuteOfDay"], unit="m")

    # Actual Departure Timestamp (account for delays pushing past midnight)
    dep_delay_mins = df["DepDelay"].fillna(0)
    df["DepTimestamp"] = df["CRSDepTimestamp"] + pd.to_timedelta(dep_delay_mins, unit="m")

    # Scheduled Arrival Timestamp (if scheduled arrival minute < scheduled dep, next day)
    crs_overnight = (df["CRSArrMinuteOfDay"] < df["CRSDepMinuteOfDay"]).astype(int)
    df["CRSArrTimestamp"] = base_dates + pd.to_timedelta(crs_overnight, unit="D") + pd.to_timedelta(df["CRSArrMinuteOfDay"], unit="m")

    # Actual Arrival Timestamp
    arr_delay_mins = df["ArrDelay"].fillna(0)
    df["ArrTimestamp"] = df["CRSArrTimestamp"] + pd.to_timedelta(arr_delay_mins, unit="m")

    # 7. Operational Consistency & Anomaly Filtering
    # For completed flights, ensure positive air time and elapsed time
    completed_mask = df["Flight_Status"] == "Completed"
    valid_mask = ~completed_mask | (
        (df["ActualElapsedTime"] > 0)
        & (df["AirTime"] > 0)
        & (df["TaxiOut"] >= 0)
        & (df["TaxiIn"] >= 0)
    )
    df_cleaned = df[valid_mask].copy()

    # Reset index
    df_cleaned.reset_index(drop=True, inplace=True)

    # Export to Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_parquet(output_path, index=False, engine="pyarrow")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    rows_retained = len(df_cleaned)
    pct_retained = (rows_retained / initial_rows) * 100
    print(f"[CLEANING] Cleaned {rows_retained:,} rows ({pct_retained:.1f}% retained) -> {output_path} ({file_size_mb:.2f} MB)")

    return df_cleaned


if __name__ == "__main__":
    clean_flight_data()
