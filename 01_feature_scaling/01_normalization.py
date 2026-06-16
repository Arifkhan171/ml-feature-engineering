# Feature Scaling - Normalization (Min-Max Scaling)
# Normalization scales all values between 0 and 1
# Best used when data does NOT follow normal distribution

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/titanic.csv",
                 usecols=["Age", "Fare", "Survived"])
df.dropna(inplace=True)

print("Original Data:")
print(df.head())
print(f"\nAge range: {df['Age'].min():.1f} - {df['Age'].max():.1f}")
print(f"Fare range: {df['Fare'].min():.1f} - {df['Fare'].max():.1f}")

# Train-test split
X = df[["Age", "Fare"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Apply MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nAfter Normalization (0 to 1):")
print(pd.DataFrame(X_train_scaled, columns=["Age", "Fare"]).head())
print(f"\nAge range after scaling: {X_train_scaled[:,0].min():.2f} - {X_train_scaled[:,0].max():.2f}")
print(f"Fare range after scaling: {X_train_scaled[:,1].min():.2f} - {X_train_scaled[:,1].max():.2f}")
