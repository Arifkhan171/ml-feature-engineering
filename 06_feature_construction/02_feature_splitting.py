# Feature Construction - Feature Splitting
# Split one column into multiple meaningful columns
# Example: "Full Name" -> "First Name" + "Last Name"

import numpy as np
import pandas as pd

# --- Example 1: Split full name ---
df = pd.DataFrame({
    "full_name": ["Arif Khan", "Ajab Gul", "Sattar Ahmad", "Gul Faiz"],
    "location":  ["Loralai, Balochistan", "Quetta, Balochistan",
                  "Karachi, Sindh", "Lahore, Punjab"]
})

print("Original data:")
print(df)

# Split name into first and last
df[["first_name", "last_name"]] = df["full_name"].str.split(" ", expand=True)

# Split location into city and province
df[["city", "province"]] = df["location"].str.split(", ", expand=True)

print("\nAfter feature splitting:")
print(df[["first_name", "last_name", "city", "province"]])

# --- Example 2: Split IP address ---
df2 = pd.DataFrame({
    "ip_address": ["192.168.1.1", "10.0.0.5", "172.16.254.1"]
})
df2[["octet1", "octet2", "octet3", "octet4"]] = \
    df2["ip_address"].str.split(".", expand=True).astype(int)

print("\nIP address splitting:")
print(df2)

# --- Example 3: Email domain extraction ---
df3 = pd.DataFrame({
    "email": ["arif@gmail.com", "ajab@yahoo.com", "gul@outlook.com"]
})
df3["username"] = df3["email"].str.split("@").str[0]
df3["domain"]   = df3["email"].str.split("@").str[1]

print("\nEmail feature splitting:")
print(df3)
