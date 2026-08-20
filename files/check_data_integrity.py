import pandas as pd
df = pd.read_csv("raw_data.csv")
df = df.drop_duplicates(subset=["date", "train_id", "station_name"], keep="last")
df.to_csv("raw_data.csv", index=False)
print(f"Cleaned. {len(df)} rows remain.")
