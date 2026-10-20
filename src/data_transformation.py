"""Data Transformation & Feature Engineering Module.

Constructs multi-leg aircraft tail rotation chains, computes turnaround buffer erosion,
delay absorption vs. cascade propagation metrics, airport congestion indices,
and quantifies real-world financial losses using FAA/A4A economic benchmarks.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# FAA & Airlines for America (A4A) Economic Standards
DIRECT_AOC_PER_MINUTE = 90.00       # Fuel burn, maintenance cycles, engine hours ($/min)
PASSENGER_PVOT_PER_HOUR = 47.00     # FAA Passenger Value of Time ($/hr)
DEFAULT_SEAT_CAPACITY = 160         # Standard narrowbody aircraft (A320 / B737)
DEFAULT_LOAD_FACTOR = 0.84          # 84% average industry load factor
CREW_TIMEOUT_THRESHOLD_MINS = 90    # Delay threshold where crew swap risk escalates
CREW_RECALL_PENALTY_USD = 2500.00   # Reserve crew deadhead / timeout penalty


def transform_flight_operations(
    input_path: str = "data/processed/flights_cleaned.parquet",
    output_path: str = "data/processed/flights_transformed.parquet",
) -> pd.DataFrame:
    """Transforms cleaned flight records into operational rotation chains with financial costs."""
    print(f"[TRANSFORMATION] Loading cleaned data from {input_path}...")
    df = pd.read_parquet(input_path)

    # 1. Sort by Tail Number and Scheduled Departure for Window Chain Operations
    df.sort_values(by=["Tail_Number", "CRSDepTimestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2. Window Operations: Previous Flight Leg in Aircraft Chain
    # Partition by Tail_Number
    tail_group = df.groupby("Tail_Number")

    df["Daily_Leg_Sequence"] = tail_group.cumcount() + 1
    df["Prev_FlightDate"] = tail_group["FlightDate"].shift(1)
    df["Prev_Dest"] = tail_group["Dest"].shift(1)
    df["Prev_CRSArrTimestamp"] = tail_group["CRSArrTimestamp"].shift(1)
    df["Prev_ArrTimestamp"] = tail_group["ArrTimestamp"].shift(1)
    df["Prev_ArrDelay"] = tail_group["ArrDelay"].shift(1)
    df["Prev_Flight_Status"] = tail_group["Flight_Status"].shift(1)

    # Determine if previous leg is valid for turnaround (same day rotation and continuous route)
    is_same_day = df["FlightDate"] == df["Prev_FlightDate"]
    is_continuous_tail = df["Origin"] == df["Prev_Dest"]
    valid_rotation = is_same_day & is_continuous_tail & (df["Prev_Flight_Status"] == "Completed")

    # 3. Turnaround Buffer & Ground Operations Metrics
    # Scheduled turnaround buffer between scheduled arrival of incoming flight and scheduled departure
    turnaround_buffer_delta = (df["CRSDepTimestamp"] - df["Prev_CRSArrTimestamp"]).dt.total_seconds() / 60.0
    df["Turnaround_Buffer_Mins"] = np.where(valid_rotation, turnaround_buffer_delta, np.nan)

    # Actual ground turnaround time
    actual_turnaround_delta = (df["DepTimestamp"] - df["Prev_ArrTimestamp"]).dt.total_seconds() / 60.0
    df["Actual_Turnaround_Mins"] = np.where(valid_rotation, actual_turnaround_delta, np.nan)

    # Buffer Erosion / Deficit (positive = buffer absorbed delay; negative = buffer breached)
    # When incoming delay > 0:
    # If Turnaround_Buffer >= Incoming_Delay + Min_Turnaround (40m): Delay is absorbed!
    # Else: Inbound delay overflows into next leg departure.
    inbound_delay = df["Prev_ArrDelay"].clip(lower=0)
    min_physical_turnaround = 40.0

    buffer_allowance = (df["Turnaround_Buffer_Mins"] - min_physical_turnaround).clip(lower=0)
    delay_absorbed = np.minimum(inbound_delay, buffer_allowance)
    df["Delay_Absorbed_Mins"] = np.where(valid_rotation, delay_absorbed, 0.0)

    # Delay carried forward / propagated to current departure
    delay_propagated = np.maximum(0.0, inbound_delay - buffer_allowance)
    df["Delay_Propagated_Mins"] = np.where(valid_rotation, delay_propagated, 0.0)

    # Buffer breach: True if turnaround time was cut below minimum operational threshold
    df["Buffer_Breach"] = np.where(
        valid_rotation,
        (df["Actual_Turnaround_Mins"] < min_physical_turnaround) | (df["Delay_Propagated_Mins"] > 0),
        False,
    )

    # Cascade Indicator: Was this flight delayed primarily because the incoming tail was late?
    df["Is_Cascading_Delay"] = (df["DepDelay"] >= 15) & (df["Delay_Propagated_Mins"] >= 15)

    # 4. Temporal & Hub Traffic Density Features
    df["DepHour"] = (df["CRSDepMinuteOfDay"] // 60).astype(int)
    # Peak Bank: 07:00-09:59 (Morning Bank) or 16:00-19:59 (Evening Bank)
    df["Is_Peak_Bank"] = df["DepHour"].isin([7, 8, 9, 16, 17, 18, 19])

    # Hub Concurrent Departure Volume: Flights departing same Origin within the same hour
    origin_hour_counts = df.groupby(["FlightDate", "Origin", "DepHour"])["Tail_Number"].transform("count")
    df["Origin_Hourly_Volume"] = origin_hour_counts

    # 5. Financial Cost Quantification Engine
    delay_minutes = df["ArrDelayMinutes"].fillna(0.0)
    passengers = DEFAULT_SEAT_CAPACITY * DEFAULT_LOAD_FACTOR  # ~134 passengers

    # Direct Aircraft Operating Cost (fuel, airframe, engine cycles)
    df["Direct_AOC_USD"] = delay_minutes * DIRECT_AOC_PER_MINUTE

    # Passenger Value of Time (PVOT) & Disruption
    df["Passenger_Delay_USD"] = (delay_minutes / 60.0) * passengers * PASSENGER_PVOT_PER_HOUR

    # Crew Timeout Escalation Penalty (non-linear risk when delay exceeds 90 minutes)
    df["Crew_Penalty_USD"] = np.where(delay_minutes >= CREW_TIMEOUT_THRESHOLD_MINS, CREW_RECALL_PENALTY_USD, 0.0)

    # Total Operational Financial Loss
    df["Total_Financial_Loss_USD"] = (
        df["Direct_AOC_USD"] + df["Passenger_Delay_USD"] + df["Crew_Penalty_USD"]
    )

    # Calculate proportion of financial loss caused by upstream propagation
    cascade_cost_ratio = np.where(
        delay_minutes > 0,
        np.minimum(1.0, df["Delay_Propagated_Mins"] / delay_minutes),
        0.0,
    )
    df["Cascading_Loss_USD"] = df["Total_Financial_Loss_USD"] * cascade_cost_ratio

    # Export Transformed Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    total_loss_m = df["Total_Financial_Loss_USD"].sum() / 1_000_000
    cascade_loss_m = df["Cascading_Loss_USD"].sum() / 1_000_000
    cascade_legs = int(df["Is_Cascading_Delay"].sum())

    print(f"[TRANSFORMATION] Processed {len(df):,} flights -> {output_path} ({file_size_mb:.2f} MB)")
    print(f"[FINANCIAL IMPACT] Total Estimated Delay Loss: ${total_loss_m:.2f}M")
    print(f"[CASCADES] Downstream Cascading Losses: ${cascade_loss_m:.2f}M across {cascade_legs:,} cascade legs.")

    return df


if __name__ == "__main__":
    transform_flight_operations()
