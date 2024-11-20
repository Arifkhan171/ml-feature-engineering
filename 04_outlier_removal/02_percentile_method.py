# Outlier Removal - Percentile / Winsorization Method
# Clip extreme values at specific percentiles (e.g. 1st and 99th)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/weight_height.csv")

X = df[["Height"]]
y = df["Weight"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Calculate percentile boundaries
lower = X_train["Height"].quantile(0.01)
upper = X_train["Height"].quantile(0.99)

print(f"Lower (1st percentile):  {lower:.2f}")
print(f"Upper (99th percentile): {upper:.2f}")

# Count outliers
outliers_count = X_train[(X_train["Height"] < lower) |
                          (X_train["Height"] > upper)].shape[0]
print(f"\nOutliers found: {outliers_count}")

# Remove outliers
X_train_clean = X_train[(X_train["Height"] >= lower) &
                         (X_train["Height"] <= upper)]

print(f"Shape before: {X_train.shape}")
print(f"Shape after:  {X_train_clean.shape}")

# Winsorization - clip instead of remove
X_train_clipped = X_train.copy()
X_train_clipped["Height"] = X_train_clipped["Height"].clip(lower, upper)
print(f"\nAfter clipping - min: {X_train_clipped['Height'].min():.2f}")
print(f"After clipping - max: {X_train_clipped['Height'].max():.2f}")
