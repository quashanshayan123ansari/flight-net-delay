"""Data Normalization Module for Airline Operations & Delay Engine.

Performs:
1. Relational Database Normalization (Star Schema / 3NF) creating:
   - dim_airlines
   - dim_airports
   - dim_aircraft
   - dim_date
   - fact_flights
   - fact_delay_cascades
2. Statistical Feature Normalization (Z-score standardization and Min-Max scaling)
   for operational analytics, risk scoring, and predictive modeling.
3. Ingests normalized tables into DuckDB OLAP database and partitioned Parquet files.
"""

from __future__ import annotations

import os
from typing import Dict
import duckdb
import numpy as np
import pandas as pd
from src.data_ingestion import AIRLINES, AIRPORTS


def normalize_flight_data(
    input_path: str = "data/processed/flights_transformed.parquet",
    output_dir: str = "data/processed",
    output_db: str = "data/flights.duckdb",
) -> Dict[str, pd.DataFrame]:
    """Generates 3NF Star Schema relational tables and statistical feature scalings."""
    print(f"[NORMALIZATION] Loading transformed dataset from {input_path}...")
    df = pd.read_parquet(input_path)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_db), exist_ok=True)

    # ==========================================
    # 1. Relational Dimensional Normalization
    # ==========================================

    # A. Airline Dimension (dim_airlines)
    airline_records = []
    for code, info in AIRLINES.items():
        airline_records.append({
            "airline_code": code,
            "airline_name": info[0],
            "dot_id": info[1],
            "primary_hubs": ", ".join(info[2]),
        })
    dim_airlines = pd.DataFrame(airline_records)

    # B. Airport Dimension (dim_airports)
    airport_records = []
    for code, info in AIRPORTS.items():
        airport_records.append({
            "airport_code": code,
            "city_name": info[0],
            "state": info[1],
            "latitude": info[2],
            "longitude": info[3],
            "hub_category": info[4],
        })
    dim_airports = pd.DataFrame(airport_records)

    # C. Aircraft Fleet Dimension (dim_aircraft)
    unique_tails = df[["Tail_Number", "Reporting_Airline"]].drop_duplicates().copy()
    unique_tails.rename(columns={"Tail_Number": "tail_number", "Reporting_Airline": "airline_code"}, inplace=True)
    unique_tails["seat_capacity"] = 160
    unique_tails["aircraft_family"] = "Narrowbody (A320/B737)"
    dim_aircraft = unique_tails.reset_index(drop=True)

    # D. Date Dimension (dim_date)
    unique_dates = pd.to_datetime(df["FlightDate"].unique()).sort_values()
    dim_date = pd.DataFrame({
        "flight_date": unique_dates.strftime("%Y-%m-%d"),
        "day_of_week": unique_dates.dayofweek + 1,  # 1 = Monday
        "day_name": unique_dates.day_name(),
        "is_weekend": unique_dates.dayofweek.isin([5, 6]),
        "month": unique_dates.month,
        "year": unique_dates.year,
    })

    # ==========================================
    # 2. Statistical Normalization & Feature Scaling
    # ==========================================
    # Z-Score standardization: (x - mean) / std
    # Min-Max normalization: (x - min) / (max - min)
    df_norm = df.copy()

    # Delay Z-Scores
    arr_delay = df_norm["ArrDelayMinutes"].fillna(0.0)
    df_norm["ArrDelay_zscore"] = (arr_delay - arr_delay.mean()) / (arr_delay.std() + 1e-6)

    dep_delay = df_norm["DepDelayMinutes"].fillna(0.0)
    df_norm["DepDelay_zscore"] = (dep_delay - dep_delay.mean()) / (dep_delay.std() + 1e-6)

    # Congestion Z-Score
    vol = df_norm["Origin_Hourly_Volume"].fillna(0.0)
    df_norm["Hourly_Volume_zscore"] = (vol - vol.mean()) / (vol.std() + 1e-6)

    # Financial Loss Z-Score
    loss = df_norm["Total_Financial_Loss_USD"].fillna(0.0)
    df_norm["Financial_Loss_zscore"] = (loss - loss.mean()) / (loss.std() + 1e-6)

    # Min-Max Scaled Congestion Score (0.0 to 1.0)
    vol_min, vol_max = vol.min(), vol.max()
    df_norm["Congestion_Score_minmax"] = (vol - vol_min) / (vol_max - vol_min + 1e-6)

    # Min-Max Scaled Delay Propagation Severity
    prop_delay = df_norm["Delay_Propagated_Mins"].fillna(0.0)
    df_norm["Propagation_Severity_minmax"] = (prop_delay - prop_delay.min()) / (prop_delay.max() - prop_delay.min() + 1e-6)

    # ==========================================
    # 3. Fact Tables
    # ==========================================
    # E. Fact Flights (Core operational facts)
    flight_fact_cols = [
        "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Tail_Number",
        "Origin", "Dest", "Flight_Status", "CRSDepTime_Str", "DepTime_Str",
        "CRSArrTime_Str", "ArrTime_Str", "CRSElapsedTime", "ActualElapsedTime",
        "AirTime", "Distance", "DepDelayMinutes", "ArrDelayMinutes",
        "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
        "Direct_AOC_USD", "Passenger_Delay_USD", "Crew_Penalty_USD", "Total_Financial_Loss_USD",
        "ArrDelay_zscore", "DepDelay_zscore", "Hourly_Volume_zscore", "Congestion_Score_minmax"
    ]
    fact_flights = df_norm[[c for c in flight_fact_cols if c in df_norm.columns]].copy()
    fact_flights.insert(0, "flight_id", range(1, len(fact_flights) + 1))

    # F. Fact Delay Cascades (Downstream propagation & ground operations)
    cascade_fact_cols = [
        "flight_id", "FlightDate", "Tail_Number", "Reporting_Airline", "Daily_Leg_Sequence",
        "Origin", "Dest", "Prev_Origin", "Prev_Dest", "Turnaround_Buffer_Mins",
        "Actual_Turnaround_Mins", "Delay_Absorbed_Mins", "Delay_Propagated_Mins",
        "Buffer_Breach", "Is_Cascading_Delay", "Cascading_Loss_USD", "Propagation_Severity_minmax"
    ]
    fact_delay_cascades = df_norm.copy()
    fact_delay_cascades.insert(0, "flight_id", range(1, len(fact_delay_cascades) + 1))
    fact_delay_cascades = fact_delay_cascades[[c for c in cascade_fact_cols if c in fact_delay_cascades.columns]]

    # ==========================================
    # 4. Save Normalized Tables to Parquet
    # ==========================================
    tables = {
        "dim_airlines": dim_airlines,
        "dim_airports": dim_airports,
        "dim_aircraft": dim_aircraft,
        "dim_date": dim_date,
        "fact_flights": fact_flights,
        "fact_delay_cascades": fact_delay_cascades,
    }

    for name, tbl in tables.items():
        parquet_file = os.path.join(output_dir, f"{name}.parquet")
        tbl.to_parquet(parquet_file, index=False, engine="pyarrow")

    # ==========================================
    # 5. Ingest Normalized Star Schema into DuckDB
    # ==========================================
    con = duckdb.connect(output_db)
    for name in tables.keys():
        parquet_file = os.path.join(output_dir, f"{name}.parquet").replace("\\", "/")
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{parquet_file}')")

    # Create analytical view combining flight facts with cascade propagation and dimensions
    con.execute("""
        CREATE OR REPLACE VIEW v_flight_network_operations AS
        SELECT 
            f.flight_id,
            f.FlightDate,
            f.Reporting_Airline,
            a.airline_name,
            f.Flight_Number_Reporting_Airline AS flight_num,
            f.Tail_Number,
            f.Origin,
            orig.city_name AS origin_city,
            orig.hub_category AS origin_hub_category,
            f.Dest,
            dest.city_name AS dest_city,
            dest.hub_category AS dest_hub_category,
            f.Flight_Status,
            f.CRSDepTime_Str,
            f.DepTime_Str,
            f.CRSArrTime_Str,
            f.ArrTime_Str,
            f.DepDelayMinutes,
            f.ArrDelayMinutes,
            f.CarrierDelay,
            f.WeatherDelay,
            f.NASDelay,
            f.LateAircraftDelay,
            f.Total_Financial_Loss_USD,
            c.Daily_Leg_Sequence,
            c.Turnaround_Buffer_Mins,
            c.Actual_Turnaround_Mins,
            c.Delay_Absorbed_Mins,
            c.Delay_Propagated_Mins,
            c.Buffer_Breach,
            c.Is_Cascading_Delay,
            c.Cascading_Loss_USD,
            f.Congestion_Score_minmax
        FROM fact_flights f
        LEFT JOIN fact_delay_cascades c ON f.flight_id = c.flight_id
        LEFT JOIN dim_airlines a ON f.Reporting_Airline = a.airline_code
        LEFT JOIN dim_airports orig ON f.Origin = orig.airport_code
        LEFT JOIN dim_airports dest ON f.Dest = dest.airport_code
    """)

    view_count = con.execute("SELECT count(*) FROM v_flight_network_operations").fetchone()[0]
    con.close()

    print(f"[NORMALIZATION] Star Schema successfully populated in DuckDB ({output_db}):")
    for name, tbl in tables.items():
        print(f"   • {name}: {len(tbl):,} rows")
    print(f"   • Analytical View 'v_flight_network_operations': {view_count:,} rows")

    return tables


if __name__ == "__main__":
    normalize_flight_data()
