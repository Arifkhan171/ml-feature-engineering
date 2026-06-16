# Missing Values - Categorical Data
# Strategies: most frequent, missing indicator, random sample

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/titanic.csv",
                 usecols=["Embarked", "Cabin", "Survived"])

print("Missing values:")
print(df.isnull().sum())

X = df[["Embarked"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- 1. Most Frequent Imputation ---
freq_imputer = SimpleImputer(strategy="most_frequent")
X_train_freq = freq_imputer.fit_transform(X_train)
print(f"\nMost frequent value for Embarked: {freq_imputer.statistics_[0]}")

# --- 2. Constant/Unknown Imputation ---
const_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
X_train_const = const_imputer.fit_transform(X_train)
print(f"Constant fill value: {const_imputer.statistics_[0]}")

# --- 3. Missing Indicator (add flag column) ---
df_flag = df[["Embarked"]].copy()
df_flag["Embarked_missing"] = df_flag["Embarked"].isnull().astype(int)
df_flag["Embarked"].fillna("Unknown", inplace=True)
print("\nWith missing indicator column:")
print(df_flag[df_flag["Embarked_missing"] == 1].head())

print("\nMissing after freq imputation:",
      pd.DataFrame(X_train_freq).isnull().sum().sum())
