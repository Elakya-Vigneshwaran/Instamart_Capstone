
import pandas as pd

# Load cleaned customer data
customers = pd.read_csv("data/clean/customers_clean.csv")

# Load cleaned order data
orders = pd.read_csv("data/clean/orders_clean.csv")

# Display customer columns
print("\nCUSTOMER TABLE COLUMNS")
print(customers.columns.tolist())

# Display order columns
print("\nORDER TABLE COLUMNS")
print(orders.columns.tolist())

# Display first 5 customer records
print("\nCUSTOMER DATA")
print(customers.head())

# Display first 5 order records
print("\nORDER DATA")
print(orders.head())

print("\nDATA INSPECTION COMPLETED")
