# Outlier Removal - IQR / Box Plot Method
# IQR = Interquartile Range (Q3 - Q1)
# Outliers: values below Q1-1.5*IQR or above Q3+1.5*IQR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/weight_height.csv")
print("Dataset shape:", df.shape)
print(df.head())

X = df[["Height"]]
y = df["Weight"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Calculate IQR boundaries
Q1  = X_train["Height"].quantile(0.25)
Q3  = X_train["Height"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nQ1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")

# Find outliers
outliers = X_train[(X_train["Height"] < lower_bound) |
                   (X_train["Height"] > upper_bound)]
print(f"\nOutliers found: {len(outliers)}")

# Remove outliers
X_train_clean = X_train[(X_train["Height"] >= lower_bound) &
                         (X_train["Height"] <= upper_bound)]
print(f"Shape before: {X_train.shape}")
print(f"Shape after:  {X_train_clean.shape}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].boxplot(X_train["Height"])
axes[0].set_title("Before Outlier Removal")
axes[0].set_ylabel("Height")

axes[1].boxplot(X_train_clean["Height"])
axes[1].set_title("After Outlier Removal (IQR)")
axes[1].set_ylabel("Height")

plt.tight_layout()
plt.show()
