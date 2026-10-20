"""Master Data Pipeline Orchestrator for Airline Network Operations Engine.

Executes the complete end-to-end data processing workflow:
1. Ingestion: Generates/fetches BTS TranStats multi-leg flight records.
2. Profiling: Comprehensive statistical and delay root-cause health audit.
3. Cleaning: BTS delay cause imputation, timestamp sanitization & anomaly filtering.
4. Transformation: Tail-number lag chain tracking, buffer breach & FAA financial loss model.
5. Normalization: 3NF Star Schema dimension/fact creation & Z-score feature scaling in DuckDB.
"""

from __future__ import annotations

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_ingestion import ingest_flight_data
from src.data_profiling import profile_flight_dataset
from src.data_cleaning import clean_flight_data
from src.data_transformation import transform_flight_operations
from src.data_normalization import normalize_flight_data


def run_pipeline(num_aircraft: int = 500, days: int = 7) -> None:
    """Runs the 5-stage flight data engineering pipeline."""
    start_total = time.time()
    print("=" * 75)
    print(">>> STARTING AIRLINE NETWORK OPERATIONS DATA PIPELINE")
    print("=" * 75)

    # Stage 1: Data Ingestion
    t0 = time.time()
    print("\n[STAGE 1/5] DATA INGESTION")
    df_raw = ingest_flight_data(num_aircraft=num_aircraft, days=days)
    print(f"[*] Stage 1 completed in {time.time() - t0:.2f}s")

    # Stage 2: Data Profiling
    t0 = time.time()
    print("\n[STAGE 2/5] DATA PROFILING")
    profile_results = profile_flight_dataset(df_raw)
    print(f"[*] Stage 2 completed in {time.time() - t0:.2f}s")

    # Stage 3: Data Cleaning
    t0 = time.time()
    print("\n[STAGE 3/5] DATA CLEANING")
    df_clean = clean_flight_data()
    print(f"[*] Stage 3 completed in {time.time() - t0:.2f}s")

    # Stage 4: Data Transformation
    t0 = time.time()
    print("\n[STAGE 4/5] DATA TRANSFORMATION & COST MODELING")
    df_transformed = transform_flight_operations()
    print(f"[*] Stage 4 completed in {time.time() - t0:.2f}s")

    # Stage 5: Data Normalization
    t0 = time.time()
    print("\n[STAGE 5/5] DATA NORMALIZATION & STAR SCHEMA MODELING")
    normalized_tables = normalize_flight_data()
    print(f"[*] Stage 5 completed in {time.time() - t0:.2f}s")

    total_time = time.time() - start_total
    print("\n" + "=" * 75)
    print(f"SUCCESS: PIPELINE EXECUTION FINISHED IN {total_time:.2f}s!")
    print("=" * 75)
    print(f"Processed Records: {len(df_transformed):,}")
    print("Artifact Locations:")
    print("   * Raw Parquet:         data/raw/flights_raw.parquet")
    print("   * Profiling Report:    data/data_profiling_report.md")
    print("   * Cleaned Parquet:     data/processed/flights_cleaned.parquet")
    print("   * Transformed Parquet: data/processed/flights_transformed.parquet")
    print("   * Star Schema Tables:  data/processed/dim_*.parquet & fact_*.parquet")
    print("   * DuckDB OLAP DB:      data/flights.duckdb")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
