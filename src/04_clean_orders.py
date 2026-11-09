
# STEP 1: Import libraries
import pandas as pd
import os

# STEP 2: Load orders dataset
orders = pd.read_csv("data/raw/orders.csv")

print("ORIGINAL DATASET")
print("Shape:", orders.shape)

# STEP 3: Convert columns into correct data types
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

orders["quantity"] = pd.to_numeric(
    orders["quantity"], errors="coerce"
)

orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
)

# STEP 4: Remove exact duplicate rows
orders = orders.drop_duplicates()

# STEP 5: Remove earlier duplicate order records
# Keep the latest record for each order_id

orders = orders.sort_values("order_ts")

orders = orders.drop_duplicates(
    subset="order_id",
    keep="last"
)

# STEP 6: Clean negative delivery times
orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = pd.NA

# STEP 7: Handle invalid order amounts
# Do not invent missing order amounts.
# Keep them as missing values.

# STEP 8: Check the cleaned dataset
print("\nCLEANED DATASET")
print("Shape:", orders.shape)

print("\nMISSING VALUES")
print(orders.isnull().sum())

print("\nDUPLICATE ORDER IDs")
print(orders["order_id"].duplicated().sum())

print("\nORDER STATUS")
print(orders["status"].value_counts())

# STEP 9: Save cleaned dataset
os.makedirs("data/clean", exist_ok=True)

orders.to_csv(
    "data/clean/orders_clean.csv",
    index=False
)

print("\nORDERS DATA CLEANED AND SAVED SUCCESSFULLY!")
