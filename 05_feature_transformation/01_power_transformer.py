# Feature Transformation - Power Transformer
# Transforms skewed data to normal distribution
# Methods: Yeo-Johnson (works with negative values), Box-Cox (positive only)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/titanic.csv", usecols=["Age", "Fare", "Survived"])
df.dropna(inplace=True)

X = df[["Age", "Fare"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- Yeo-Johnson Transformation ---
pt = PowerTransformer(method="yeo-johnson")
X_train_transformed = pt.fit_transform(X_train)
X_test_transformed  = pt.transform(X_test)

print("Lambda values (transformation parameters):")
print(pt.lambdas_)

# Before vs After comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].hist(X_train["Age"], bins=30, color="steelblue")
axes[0, 0].set_title("Age - Before Transformation")

axes[0, 1].hist(X_train_transformed[:, 0], bins=30, color="orange")
axes[0, 1].set_title("Age - After Yeo-Johnson")

axes[1, 0].hist(X_train["Fare"], bins=30, color="steelblue")
axes[1, 0].set_title("Fare - Before Transformation")

axes[1, 1].hist(X_train_transformed[:, 1], bins=30, color="orange")
axes[1, 1].set_title("Fare - After Yeo-Johnson")

plt.tight_layout()
plt.show()
