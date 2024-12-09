# Mixed Features - Variables Containing Both Numbers and Text
# Example: "30 years", "5km", "$100" - need to extract the numeric part

import numpy as np
import pandas as pd

# --- Example 1: Mixed variable with units ---
df = pd.DataFrame({
    "price": ["$10,000", "$15,500", "$8,200", "$22,000"],
    "distance": ["5km", "12km", "3km", "20km"],
    "age": ["25 years", "30 years", "45 years", "22 years"]
})

print("Original mixed data:")
print(df)

# Extract numbers using regex
df["price_clean"]    = df["price"].str.replace("[$,]", "", regex=True).astype(float)
df["distance_clean"] = df["distance"].str.replace("km", "").astype(float)
df["age_clean"]      = df["age"].str.replace(" years", "").astype(float)

print("\nCleaned numeric data:")
print(df[["price_clean", "distance_clean", "age_clean"]])

# --- Example 2: Zomato dataset - mixed ratings ---
try:
    zomato = pd.read_csv("../data/zomato.csv")
    if "rate" in zomato.columns:
        print("\nZomato ratings sample:")
        print(zomato["rate"].head(10))
        # Clean rating: "4.1/5" -> 4.1
        zomato["rate_clean"] = zomato["rate"].str.split("/").str[0]
        zomato["rate_clean"] = pd.to_numeric(
            zomato["rate_clean"], errors="coerce")
        print("\nCleaned ratings:")
        print(zomato["rate_clean"].describe())
except Exception:
    print("\nZomato dataset not available - skip")
