# Missing Values - Numerical Data
# Strategies: mean, median, arbitrary, random sample imputation

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/titanic.csv",
                 usecols=["Age", "Fare", "Survived"])

print("Missing values before imputation:")
print(df.isnull().sum())
print(f"Age missing: {df['Age'].isnull().sum()} / {len(df)}")

X = df[["Age", "Fare"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- 1. Mean Imputation ---
mean_imputer = SimpleImputer(strategy="mean")
X_train_mean = mean_imputer.fit_transform(X_train)
print(f"\nMean imputation value for Age: {mean_imputer.statistics_[0]:.2f}")

# --- 2. Median Imputation (better for skewed data) ---
median_imputer = SimpleImputer(strategy="median")
X_train_median = median_imputer.fit_transform(X_train)
print(f"Median imputation value for Age: {median_imputer.statistics_[0]:.2f}")

# --- 3. Arbitrary Value Imputation ---
arbitrary_imputer = SimpleImputer(strategy="constant", fill_value=999)
X_train_arbitrary = arbitrary_imputer.fit_transform(X_train)
print(f"\nArbitrary value used: {arbitrary_imputer.statistics_[0]}")

# --- 4. Check after imputation ---
result = pd.DataFrame(X_train_mean, columns=["Age", "Fare"])
print(f"\nMissing values after mean imputation: {result.isnull().sum().sum()}")
