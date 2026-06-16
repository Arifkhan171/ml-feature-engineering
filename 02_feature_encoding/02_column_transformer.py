# Feature Encoding - ColumnTransformer
# Apply different transformations to different columns in one step

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/titanic.csv",
                 usecols=["Age", "Fare", "Sex", "Embarked", "Survived"])
df.dropna(inplace=True)

X = df[["Age", "Fare", "Sex", "Embarked"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define transformations per column type
numeric_features     = ["Age", "Fare"]
categorical_features = ["Sex", "Embarked"]

# ColumnTransformer applies each transformer to specified columns
transformer = ColumnTransformer(transformers=[
    ("num", StandardScaler(),                        numeric_features),
    ("cat", OneHotEncoder(sparse_output=False,
                          drop="first",
                          handle_unknown="ignore"), categorical_features)
])

X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed  = transformer.transform(X_test)

print("Original shape:", X_train.shape)
print("Transformed shape:", X_train_transformed.shape)
print("\nTransformed sample (first row):")
print(X_train_transformed[0])
print("\nFeature names after transformation:")
print(transformer.get_feature_names_out())
