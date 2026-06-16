# Feature Encoding - Categorical Variables
# Convert text categories into numbers that ML models understand

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/titanic.csv",
                 usecols=["Sex", "Embarked", "Survived"])
df.dropna(inplace=True)

print("Original Data:")
print(df.head())
print(f"\nUnique Sex values: {df['Sex'].unique()}")
print(f"Unique Embarked values: {df['Embarked'].unique()}")

X = df[["Sex", "Embarked"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- 1. Label Encoding (for binary categories) ---
le = LabelEncoder()
X_train["Sex_encoded"] = le.fit_transform(X_train["Sex"])
print("\nLabel Encoding (Sex):")
print(X_train[["Sex", "Sex_encoded"]].head())

# --- 2. Ordinal Encoding (for ordered categories) ---
oe = OrdinalEncoder(categories=[["S", "C", "Q"]])
X_train["Embarked_encoded"] = oe.fit_transform(X_train[["Embarked"]])
print("\nOrdinal Encoding (Embarked):")
print(X_train[["Embarked", "Embarked_encoded"]].head())

# --- 3. One Hot Encoding (for nominal categories) ---
ohe = OneHotEncoder(sparse_output=False, drop="first")
encoded = ohe.fit_transform(X_train[["Embarked"]])
ohe_df  = pd.DataFrame(encoded, columns=ohe.get_feature_names_out())
print("\nOne Hot Encoding (Embarked):")
print(ohe_df.head())
