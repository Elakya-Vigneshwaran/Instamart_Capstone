# Step 1: Import Pandas
import pandas as pd

# Step 2: Read the customer CSV file
customers = pd.read_csv("data/raw/customers.csv")

# Step 3: Display first 5 rows
print("FIRST FIVE CUSTOMERS")
print(customers.head())

# Step 4: Display number of rows and columns
print("\nDATASET SHAPE")
print(customers.shape)

# Step 5: Display column information
print("\nDATASET INFORMATION")
print(customers.info())

# Step 6: Check missing values
print("\nMISSING VALUES")
print(customers.isnull().sum())