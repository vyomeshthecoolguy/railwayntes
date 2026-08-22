"""
STEP 2: WASH AND CHOP (CLEAN THE DATA)
========================================
Real scraped data is always messy: missing values, duplicate rows,
weird formats. This script fixes that. Think of it like washing
vegetables and throwing away the bad ones before cooking.
"""

import pandas as pd

df = pd.read_csv("raw_data.csv")
print(f"Started with {len(df)} rows.")

# 1) Remove exact duplicate rows (same train, same station, same date twice)
before = len(df)
df = df.drop_duplicates(subset=["date", "train_id", "station_name"])
print(f"Removed {before - len(df)} duplicate rows.")

# 2) Handle missing delay values
# Option: drop rows where we don't know the delay (we can't learn from a blank)
before = len(df)
df = df.dropna(subset=["delay_minutes"])
print(f"Removed {before - len(df)} rows with missing delay values.")

# 3) Remove impossible/weird values
#    (e.g. negative delay bigger than -5 min might mean bad data, not "early train")
before = len(df)
df = df[(df["delay_minutes"] >= 0) & (df["delay_minutes"] <= 120)]
print(f"Removed {before - len(df)} rows with impossible delay values.")

# 4) Make sure data types are correct
df["date"] = pd.to_datetime(df["date"])
df["delay_minutes"] = df["delay_minutes"].astype(float)

# 5) Sort so each train's journey is in the correct station order
#    (this matters a LOT for step 3, where we look at "delay at previous station")
df = df.sort_values(["date", "train_id", "station_order"]).reset_index(drop=True)

df.to_csv("clean_data.csv", index=False)
print(f"\nSaved clean_data.csv with {len(df)} clean rows.")
print(df.head())
