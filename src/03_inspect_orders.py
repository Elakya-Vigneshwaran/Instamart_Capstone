
# STEP 1: Import Pandas
import pandas as pd

# STEP 2: Load orders dataset
orders = pd.read_csv("data/raw/orders.csv")

# STEP 3: Display first five records
print("FIRST FIVE ORDERS")
print(orders.head())

# STEP 4: Dataset dimensions
print("\nDATASET SHAPE")
print(orders.shape)

# STEP 5: Column information
print("\nDATASET INFORMATION")
orders.info()

# STEP 6: Missing values
print("\nMISSING VALUES")
print(orders.isnull().sum())

# STEP 7: Check duplicate records
print("\nEXACT DUPLICATE ROWS")
print(orders.duplicated().sum())

# STEP 8: Check order status
print("\nORDER STATUS COUNT")
print(orders["status"].value_counts())

# STEP 9: Check invalid order amounts
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

print("\nINVALID ORDER AMOUNTS")
print(orders["order_amount"].isnull().sum())

# STEP 10: Check negative delivery times
print("\nNEGATIVE DELIVERY TIMES")
print((orders["delivery_minutes"] < 0).sum())

print("\nORDERS INSPECTION COMPLETED")
