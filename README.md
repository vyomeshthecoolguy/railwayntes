# Railway NTES — Suburban Train Delay Prediction

A pipeline for predicting suburban train delays using live status data
scraped from NTES (`enquiry.indianrail.gov.in`), focused on the Chennai
Beach ↔ Tambaram corridor.

## Pipeline

| Script | Purpose |
|---|---|
| `01_generate_sample_data.py` | Generates a realistic synthetic dataset (`raw_data.csv`) for prototyping without live data |
| `02_clean_data.py` | Deduplicates, drops missing/impossible values, fixes types → `clean_data.csv` |
| `03_feature_engineering.py` | Builds model features, including the key `prev_station_delay` feature → `features.csv` |
| `04_train_model.py` | Time-based train/val/test split; compares naive baseline, linear regression, decision tree, random forest → `model_results.csv` |
| `04b_early_sanity_check.py` | Leave-one-train-out pipeline check for use before enough calendar days have been collected |
| `08_scrape_ntes_client.py` | Daily scraper: pulls live status per train from NTES, parses station-level delay data, safely merges/dedupes into `raw_data.csv` |
| `09_debug_ntes_response.py` | Dumps the raw NTES API response for a single train, for debugging field names |
| `check_data_integrity.py` | Standalone dedup pass over `raw_data.csv` |
| `run_scraper.bat` | Windows launcher for scheduling the scraper via Task Scheduler |

## Research question

Does knowing a train's delay at the previous station help predict its
delay at the next station? `prev_station_delay` is the key engineered
feature testing this.

## Setup

```bash
pip install ntes-client pandas scikit-learn numpy
```

## Running the scraper

`08_scrape_ntes_client.py` is intended to run once daily (evening, after
trains finish their runs) via cron or Windows Task Scheduler — see the
bottom of that file for scheduler setup instructions.

## Data

Data files (`raw_data.csv`, `clean_data.csv`, `features.csv`,
`model_results.csv`) are not committed — regenerate them by running the
scripts in order, or point `08_scrape_ntes_client.py` at live NTES data.
