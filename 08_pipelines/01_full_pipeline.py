# ML Pipeline - Combine All Steps Into One Clean Flow
# Pipeline ensures same transformations applied to train and test
# Prevents data leakage - fit only on train, transform both

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../data/titanic.csv",
                 usecols=["Age", "Fare", "Sex", "Embarked", "Survived"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define column types
numeric_features     = ["Age", "Fare"]
categorical_features = ["Sex", "Embarked"]

# Numeric pipeline: impute missing → scale
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

# Categorical pipeline: impute missing → encode
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False,
                              drop="first"))
])

# Combine both pipelines
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline,     numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Full pipeline with model at the end
full_pipeline = Pipeline([
    ("preprocessor",        preprocessor),
    ("classifier", LogisticRegression(max_iter=500))
])

# Train
full_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = full_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Pipeline Steps:")
for step in full_pipeline.steps:
    print(f"  {step[0]}: {step[1].__class__.__name__}")

print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nSample Predictions vs Actual:")
results = pd.DataFrame({"Actual": y_test.values[:10],
                         "Predicted": y_pred[:10]})
print(results)
