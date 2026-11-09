
# STEP 1: Import required libraries
import pandas as pd
import os

# STEP 2: Load customer dataset
customers = pd.read_csv("data/raw/customers.csv")

print("Original dataset shape:", customers.shape)

# STEP 3: Remove duplicate customer records
customers = customers.drop_duplicates(subset="customer_id")

# STEP 4: Clean city names
customers["city"] = (
    customers["city"]
    .str.strip()
    .str.title()
)

# STEP 5: Clean gender values
customers["gender"] = (
    customers["gender"]
    .str.strip()
    .str.upper()
    .map({
        "M": "Male",
        "MALE": "Male",
        "F": "Female",
        "FEMALE": "Female"
    })
)

# Replace missing gender with Unknown
customers["gender"] = customers["gender"].fillna("Unknown")

# STEP 6: Clean age values
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

# Ages below 18 or above 100 are invalid
customers.loc[
    (customers["age"] < 18) | (customers["age"] > 100),
    "age"
] = pd.NA

# Replace missing age with median age
median_age = customers["age"].median()

customers["age"] = customers["age"].fillna(median_age)

customers["age"] = customers["age"].round().astype("int64")

# STEP 7: Convert signup_date into date format
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# STEP 8: Check cleaned dataset
print("\nCLEANED DATASET")
print(customers.head())

print("\nMISSING VALUES AFTER CLEANING")
print(customers.isnull().sum())

print("\nDATASET SHAPE AFTER CLEANING")
print(customers.shape)

# STEP 9: Save cleaned dataset
os.makedirs("data/clean", exist_ok=True)

customers.to_csv(
    "data/clean/customers_clean.csv",
    index=False
)

print("\nCustomer data cleaned and saved successfully!")
