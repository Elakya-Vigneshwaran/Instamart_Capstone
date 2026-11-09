
import pandas as pd

# Load the customer support tickets dataset
tickets = pd.read_csv("data/raw/tickets.csv")

# Display the dataset size
print("Dataset shape:", tickets.shape)

# Display column names
print("\nColumn names:")
print(tickets.columns)

# Display first five records
print("\nFirst five tickets:")
print(tickets.head())

# Check missing values
print("\nMissing values:")
print(tickets.isnull().sum())

# Check data types
print("\nData types:")
print(tickets.dtypes)
