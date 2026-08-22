"""
EARLY SANITY CHECK - NOT your real results yet
==================================================
You currently have only 1 DAY of real data (4 trains). Script 04's
proper time-based split (train on early dates, test on later dates)
needs MULTIPLE DAYS to work at all - with 1 day, there's no "later"
to test on.

This script is a STAND-IN just to check the pipeline behaves
sensibly on real data: it trains on 3 of your 4 trains and tests on
the 4th (rotating through all 4), instead of splitting by date.

DO NOT put these numbers in your paper as real findings - with only
68 rows total, any MAE here is extremely noisy and not meaningful.
This is purely a "does the pipeline work correctly on real data"
check. Come back to script 04 (the real evaluation) once you have
at least 2-3 weeks of dates.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("features.csv")

feature_cols = [
    "station_order", "distance_km", "scheduled_hour",
    "day_of_week", "is_weekend", "is_peak_hour", "prev_station_delay",
]
target_col = "delay_minutes"

trains = df["train_id"].unique()
print(f"Running leave-one-train-out check across {len(trains)} trains: {list(trains)}\n")

all_true, all_pred_naive, all_pred_rf = [], [], []

for test_train in trains:
    train_df = df[df["train_id"] != test_train]
    test_df = df[df["train_id"] == test_train]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # Naive: just use prev_station_delay as the guess
    naive_pred = X_test["prev_station_delay"]

    rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    mae_naive = mean_absolute_error(y_test, naive_pred)
    mae_rf = mean_absolute_error(y_test, rf_pred)
    print(f"Test train {test_train}: naive MAE={mae_naive:.2f} | RF MAE={mae_rf:.2f}")

    all_true.extend(y_test.tolist())
    all_pred_naive.extend(naive_pred.tolist())
    all_pred_rf.extend(rf_pred.tolist())

print("\n--- Overall (pooled across all 4 folds) ---")
print(f"Naive baseline : MAE={mean_absolute_error(all_true, all_pred_naive):.2f} min")
print(f"Random Forest  : MAE={mean_absolute_error(all_true, all_pred_rf):.2f} min")

print("\nReminder: this is a PIPELINE CHECK on 68 rows / 1 day / 4 trains.")
print("Treat these numbers as 'does it run correctly', not as real findings.")
print("Come back to 04_train_model.py once you have 2-3+ weeks of dates.")
