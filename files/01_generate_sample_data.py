"""
STEP 1: GET THE RAW INGREDIENTS (DATA)
========================================
Think of this like collecting puzzle pieces before you can build the puzzle.

In real life, you would scrape this data from NTES (enquiry.indianrail.gov.in)
or the Moovit app, station by station, day by day, for a few weeks.

Since this sandbox has no internet access, this script CREATES a fake dataset
that looks EXACTLY like what real scraped data would look like. This way,
you can run the whole pipeline today and understand it -- then later, you
just replace the file this script creates (raw_data.csv) with your real
scraped data, and every other script still works without changes.

WHAT EACH ROW MEANS:
One row = one train, at one station, on one day.
Example: "EMU 4021, at Tambaram station, on 2026-01-05, was scheduled to
arrive at 8:15 AM but actually arrived at 8:22 AM (7 min late)."
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Make results repeatable (same "random" numbers every time we run this)
random.seed(42)
np.random.seed(42)

# ---- Pretend line: Chennai Beach -> Tambaram (a real, short suburban line) ----
stations = ["Chennai Beach", "Fort", "Egmore", "Chetpet", "Mambalam",
            "Guindy", "St Thomas Mount", "Pallavaram", "Chromepet", "Tambaram"]

# Distance of each station from the origin (in km) -- roughly realistic
distances_km = [0, 2, 4, 7, 9, 13, 15, 19, 22, 25]

train_ids = [f"EMU_{n}" for n in range(4001, 4021)]  # 20 pretend trains

rows = []
start_date = datetime(2026, 1, 1)
num_days = 45  # simulate 45 days of data, like 6-7 weeks of real scraping

for day in range(num_days):
    date = start_date + timedelta(days=day)
    is_weekend = date.weekday() >= 5  # Saturday=5, Sunday=6

    for train_id in train_ids:
        # Each train has a base "scheduled" start time
        base_hour = random.choice([6, 7, 8, 9, 17, 18, 19])  # peak-ish hours
        base_minute = random.choice([0, 15, 30, 45])
        sched_time = date.replace(hour=base_hour, minute=base_minute)

        # Is this during peak hours? (rough Chennai suburban peak: 7-10am, 5-8pm)
        is_peak = base_hour in [7, 8, 9, 17, 18]

        # A train's delay tends to GROW as it moves along the route
        # (this is the "delay propagation" idea from our research question)
        previous_delay = 0
        for i, station in enumerate(stations):
            # Simulate some randomness in delay:
            # - peak hours -> more delay
            # - weekends -> less delay (fewer trains, less congestion)
            noise = np.random.normal(loc=0, scale=1.5)
            peak_effect = 2.5 if is_peak else 0
            weekend_effect = -1.5 if is_weekend else 0

            # delay tends to build on top of the previous station's delay
            delay = max(0, previous_delay * 0.85 + peak_effect + weekend_effect + noise)
            previous_delay = delay

            scheduled_arrival = sched_time + timedelta(
                minutes=distances_km[i] * 2)  # ~2 min per km, rough
            actual_arrival = scheduled_arrival + timedelta(minutes=delay)

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "train_id": train_id,
                "station_name": station,
                "station_order": i + 1,          # 1st stop, 2nd stop, ...
                "distance_km": distances_km[i],
                "scheduled_arrival": scheduled_arrival.strftime("%H:%M"),
                "actual_arrival": actual_arrival.strftime("%H:%M"),
                "delay_minutes": round(delay, 1),
            })

df = pd.DataFrame(rows)

# Sprinkle in a few messy/missing values, because REAL scraped data is never clean
# (this makes our cleaning script in Step 2 actually necessary and realistic)
messy_idx = df.sample(frac=0.03, random_state=1).index
df.loc[messy_idx, "delay_minutes"] = np.nan  # some missing delay readings

dup_rows = df.sample(frac=0.01, random_state=2)
df = pd.concat([df, dup_rows], ignore_index=True)  # a few accidental duplicates

df.to_csv("raw_data.csv", index=False)
print(f"Created raw_data.csv with {len(df)} rows (fake but realistic).")
print(df.head(10))
