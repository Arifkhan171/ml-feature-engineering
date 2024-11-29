# Datetime Features - Extract Meaningful Info From Dates
# Extract: year, month, day, weekday, hour, is_weekend, days_since

import numpy as np
import pandas as pd

# --- Example 1: Basic datetime extraction ---
df = pd.DataFrame({
    "order_date": ["2023-01-15 09:30:00",
                   "2023-03-22 14:45:00",
                   "2023-07-04 20:00:00",
                   "2023-12-25 08:15:00"]
})

# Convert to datetime
df["order_date"] = pd.to_datetime(df["order_date"])

# Extract features
df["year"]       = df["order_date"].dt.year
df["month"]      = df["order_date"].dt.month
df["day"]        = df["order_date"].dt.day
df["hour"]       = df["order_date"].dt.hour
df["weekday"]    = df["order_date"].dt.dayofweek   # 0=Mon, 6=Sun
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
df["quarter"]    = df["order_date"].dt.quarter
df["week"]       = df["order_date"].dt.isocalendar().week

print("Datetime features extracted:")
print(df.to_string())

# --- Example 2: Days since reference date ---
reference_date = pd.Timestamp("2023-01-01")
df["days_since_start"] = (df["order_date"] - reference_date).dt.days
print("\nDays since Jan 1 2023:")
print(df[["order_date", "days_since_start"]])

# --- Example 3: IPL dataset dates ---
try:
    ipl = pd.read_csv("../data/ipl_matches.csv")
    if "date" in ipl.columns:
        ipl["date"]    = pd.to_datetime(ipl["date"])
        ipl["year"]    = ipl["date"].dt.year
        ipl["month"]   = ipl["date"].dt.month
        ipl["weekday"] = ipl["date"].dt.day_name()
        print("\nIPL match date features:")
        print(ipl[["date", "year", "month", "weekday"]].head())
except Exception:
    print("\nIPL dataset not available - skip")
