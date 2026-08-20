"""
STEP 4: COOK (TRAIN THE MODELS)
==================================
Now we teach a computer to guess "delay_minutes" using our features.

We use a TIME-BASED split, not a random split:
  - Train on the FIRST 70% of days
  - Validate on the NEXT 15% of days
  - Test on the LAST 15% of days
This matters because in real life, you always predict the FUTURE using
the PAST -- never let the model "peek" at data from later days.

We compare 4 models, simplest to smartest:
  1. Naive baseline   -> "predict delay = delay at previous station" (no ML at all!)
  2. Linear Regression -> a simple straight-line ML model
  3. Decision Tree     -> simple rules, more flexible than a straight line
  4. Random Forest     -> many decision trees voting together (usually the winner)

We measure with:
  - MAE  (Mean Absolute Error)  -> "on average, how many minutes off were we?"
  - RMSE (Root Mean Squared Error) -> like MAE but punishes BIG mistakes more
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("features.csv", parse_dates=["date"])

feature_cols = [
    "station_order", "distance_km", "scheduled_hour",
    "day_of_week", "is_weekend", "is_peak_hour", "prev_station_delay",
]
target_col = "delay_minutes"

# --- TIME-BASED SPLIT (never split randomly for time-series data!) ---
unique_dates = sorted(df["date"].unique())
n = len(unique_dates)
train_dates = unique_dates[: int(n * 0.70)]
val_dates   = unique_dates[int(n * 0.70): int(n * 0.85)]
test_dates  = unique_dates[int(n * 0.85):]

train_df = df[df["date"].isin(train_dates)]
val_df   = df[df["date"].isin(val_dates)]
test_df  = df[df["date"].isin(test_dates)]

print(f"Train days: {len(train_dates)} | Val days: {len(val_dates)} | Test days: {len(test_dates)}")

X_train, y_train = train_df[feature_cols], train_df[target_col]
X_test,  y_test  = test_df[feature_cols],  test_df[target_col]

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:22s} | MAE = {mae:5.2f} min | RMSE = {rmse:5.2f} min")
    return mae, rmse

results = {}

# --- MODEL 1: Naive baseline (no learning, just copy previous station's delay) ---
naive_pred = X_test["prev_station_delay"]
results["Naive baseline"] = evaluate("1. Naive baseline", y_test, naive_pred)

# --- MODEL 2: Linear Regression ---
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)
results["Linear Regression"] = evaluate("2. Linear Regression", y_test, lin_pred)

# --- MODEL 3: Decision Tree ---
tree_model = DecisionTreeRegressor(max_depth=6, random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
results["Decision Tree"] = evaluate("3. Decision Tree", y_test, tree_pred)

# --- MODEL 4: Random Forest (usually the best of these 4) ---
rf_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
results["Random Forest"] = evaluate("4. Random Forest", y_test, rf_pred)

print("\n--- Which feature does Random Forest think matters most? ---")
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
print(importances.sort_values(ascending=False))

# --- BONUS: Ablation test -- does "prev_station_delay" actually help? ---
print("\n--- Ablation: Random Forest WITHOUT prev_station_delay ---")
feature_cols_no_prev = [c for c in feature_cols if c != "prev_station_delay"]
rf_no_prev = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
rf_no_prev.fit(train_df[feature_cols_no_prev], y_train)
pred_no_prev = rf_no_prev.predict(test_df[feature_cols_no_prev])
evaluate("RF (no prev delay)", y_test, pred_no_prev)

# Save a results table
summary = pd.DataFrame(results, index=["MAE", "RMSE"]).T
summary.to_csv("model_results.csv")
print("\nSaved model_results.csv")
