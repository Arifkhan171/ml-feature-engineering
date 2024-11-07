# Feature Scaling - Standardization (Z-score Normalization)
# Standardization transforms data to mean=0 and std=1
# Best used when data follows normal distribution

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/titanic.csv",
                 usecols=["Age", "Fare", "Survived"])
df.dropna(inplace=True)

print("Original Data Stats:")
print(df[["Age", "Fare"]].describe())

# Train-test split
X = df[["Age", "Fare"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nAfter Standardization (mean=0, std=1):")
scaled_df = pd.DataFrame(X_train_scaled, columns=["Age", "Fare"])
print(scaled_df.describe().round(2))

print(f"\nScaler mean: {scaler.mean_}")
print(f"Scaler std:  {scaler.scale_}")
