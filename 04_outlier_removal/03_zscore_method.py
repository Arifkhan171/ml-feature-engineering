# Outlier Removal - Z-Score Method
# Z-score measures how many standard deviations a value is from mean
# Rule: Z-score > 3 or < -3 is considered an outlier

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/weight_height.csv")

X = df[["Height"]]
y = df["Weight"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Calculate Z-scores
z_scores = np.abs(stats.zscore(X_train["Height"]))
print(f"Max Z-score: {z_scores.max():.2f}")
print(f"Min Z-score: {z_scores.min():.2f}")

# Find outliers (Z-score > 3)
outlier_mask = z_scores > 3
print(f"\nOutliers found (Z > 3): {outlier_mask.sum()}")
print("Outlier values:")
print(X_train[outlier_mask])

# Remove outliers
X_train_clean = X_train[~outlier_mask]
print(f"\nShape before: {X_train.shape}")
print(f"Shape after:  {X_train_clean.shape}")

# Compare statistics
print("\nBefore removal:")
print(X_train["Height"].describe().round(2))
print("\nAfter removal:")
print(X_train_clean["Height"].describe().round(2))
