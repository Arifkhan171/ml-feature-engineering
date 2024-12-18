# Feature Transformation - Function Transformer
# Apply custom mathematical functions: log, sqrt, square
# Used to reduce skewness in data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/titanic.csv", usecols=["Age", "Fare", "Survived"])
df.dropna(inplace=True)
df = df[df["Fare"] > 0]  # log needs positive values

X = df[["Age", "Fare"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- Log Transformation ---
log_transformer = FunctionTransformer(np.log1p)  # log1p = log(1+x), safe for 0
X_train_log = log_transformer.fit_transform(X_train)
print("Log transformation applied")
print("Fare skewness before:", round(X_train["Fare"].skew(), 3))
print("Fare skewness after: ", round(X_train_log[:, 1].std(), 3))

# --- Square Root Transformation ---
sqrt_transformer = FunctionTransformer(np.sqrt)
X_train_sqrt = sqrt_transformer.fit_transform(X_train)
print("\nSqrt transformation applied")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(X_train["Fare"], bins=30)
axes[0].set_title("Original Fare")

axes[1].hist(X_train_log[:, 1], bins=30, color="green")
axes[1].set_title("Log Transformed Fare")

axes[2].hist(X_train_sqrt[:, 1], bins=30, color="orange")
axes[2].set_title("Sqrt Transformed Fare")

plt.tight_layout()
plt.show()
