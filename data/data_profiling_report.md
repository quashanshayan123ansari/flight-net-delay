# 📋 Flight Operations Data Profiling Report

- **Total Rows Analyzed**: 13,971
- **Total Columns**: 37
- **Memory Footprint**: 9.77 MB
- **Unique Aircraft Tail Numbers**: 482
- **Unique Routes**: 240

## 1. Flight Status & Operational Health

| Metric | Value |
| :--- | :--- |
| Completed Flights | 13,938 (99.8%) |
| Cancelled Flights | 33 (0.24%) |
| Diverted Flights | 0 |
| Departure Delay Rate (>=15 min) | **44.5%** |
| Arrival Delay Rate (>=15 min) | **68.0%** |
| Mean Departure Delay | 27.7 mins |
| 90th Percentile Departure Delay | 85.0 mins |
| Max Recorded Departure Delay | 1450 mins |

## 2. Root Cause Delay Breakdown

The primary delay contributors across all delayed legs:

| Delay Category | Total Delay (Minutes) | Share of Total Delay (%) |
| :--- | :--- | :--- |
| **LateAircraftDelay** | 228,500 | 59.47% |
| **CarrierDelay** | 49,184 | 12.8% |
| **WeatherDelay** | 64,259 | 16.72% |
| **NASDelay** | 42,274 | 11.0% |
| **SecurityDelay** | 0 | 0.0% |

> **Key Finding**: Late Aircraft Delay (downstream cascade propagation) represents the dominant delay driver across network rotations.

## 3. Airline On-Time Performance Summary

| Carrier | Flight Volume | Dep Delay Rate (% >=15m) | Arr Delay Rate (% >=15m) | Avg Arr Delay (mins) |
| :--- | :--- | :--- | :--- | :--- |
| AA | 3,489 | 44.5% | 68.4% | 39.3m |
| DL | 3,448 | 45.6% | 69.2% | 41.9m |
| UA | 3,507 | 45.2% | 67.9% | 40.2m |
| WN | 3,494 | 42.8% | 66.7% | 39.1m |

## 4. Missingness & Reporting Architecture

| Column | Missing Rows | Missing % | Interpretation |
| :--- | :--- | :--- | :--- |
| `CancellationCode` | 13,938 | 99.76% | Natural flight cancellation/diversion missingness |
| `SecurityDelay` | 4,488 | 32.12% | Standard BTS rule: Only populated when delay >= 15 min |
| `CarrierDelay` | 4,488 | 32.12% | Standard BTS rule: Only populated when delay >= 15 min |
| `LateAircraftDelay` | 4,488 | 32.12% | Standard BTS rule: Only populated when delay >= 15 min |
| `NASDelay` | 4,488 | 32.12% | Standard BTS rule: Only populated when delay >= 15 min |
| `WeatherDelay` | 4,488 | 32.12% | Standard BTS rule: Only populated when delay >= 15 min |
| `ArrDelay` | 33 | 0.24% | Natural flight cancellation/diversion missingness |
| `AirTime` | 33 | 0.24% | Natural flight cancellation/diversion missingness |
| `ArrDel15` | 33 | 0.24% | Natural flight cancellation/diversion missingness |
| `TaxiOut` | 33 | 0.24% | Natural flight cancellation/diversion missingness |
