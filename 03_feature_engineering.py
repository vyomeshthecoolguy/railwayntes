"""
STEP 3: PREP AND SEASON (FEATURE ENGINEERING)
================================================
A model can't "understand" a train the way you do. We need to turn
what we know into NUMBERS it can learn patterns from.

Example: instead of the model seeing "8:15 AM", we give it "hour = 8".
Instead of "this is the 3rd station", we give it "station_order = 3".

The MOST IMPORTANT feature we build here is:
   "prev_station_delay" = how late was THIS SAME TRAIN at the PREVIOUS station?

This is our research question in code form: does knowing the delay one
stop ago help predict the delay at the next stop?
"""

import pandas as pd

df = pd.read_csv("clean_data.csv", parse_dates=["date"])

# --- Drop origin-station rows ---
# The FIRST station of each journey (e.g. Chennai Beach for a Beach->Tambaram
# train) has no "scheduled ARRIVAL" - the train starts there, it doesn't arrive.
# So scheduled_arrival is blank for these rows. There's also no real "arrival
# delay" concept at the origin, so these rows aren't useful prediction targets
# anyway - we drop them here rather than faking a value.
before = len(df)
df = df.dropna(subset=["scheduled_arrival"])
print(f"Dropped {before - len(df)} origin-station rows (no scheduled arrival).")

# --- Basic time features ---
df["scheduled_hour"] = df["scheduled_arrival"].str.split(":").str[0].astype(int)
df["day_of_week"] = df["date"].dt.dayofweek       # 0=Monday ... 6=Sunday
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# --- Peak hour flag (Chennai suburban peak: ~7-10am, 5-8pm) ---
df["is_peak_hour"] = df["scheduled_hour"].isin([7, 8, 9, 17, 18]).astype(int)

# --- THE KEY FEATURE: delay at the previous station, for the SAME train, SAME day ---
# We sort by train+date+station_order (already done in step 2), then "shift" the
# delay column down by 1 row WITHIN each train's journey.
df = df.sort_values(["date", "train_id", "station_order"])
df["prev_station_delay"] = (
    df.groupby(["date", "train_id"])["delay_minutes"].shift(1)
)

# For the FIRST station of each journey, there's no "previous station" --
# it hasn't started yet, so delay is naturally 0.
df["prev_station_delay"] = df["prev_station_delay"].fillna(0)

# --- Final feature set the model will use ---
feature_cols = [
    "station_order",       # which stop number this is (1st, 2nd, 3rd...)
    "distance_km",         # how far from the origin station
    "scheduled_hour",      # hour of day the train is due
    "day_of_week",         # Monday=0 ... Sunday=6
    "is_weekend",          # 1 if Sat/Sun
    "is_peak_hour",        # 1 if rush hour
    "prev_station_delay",  # <-- our key research feature
]
target_col = "delay_minutes"

df_features = df[["date", "train_id", "station_name"] + feature_cols + [target_col]]
df_features.to_csv("features.csv", index=False)

print(f"Saved features.csv with {len(df_features)} rows and these columns:")
print(feature_cols, "-> predicting:", target_col)
print(df_features.head(10))
