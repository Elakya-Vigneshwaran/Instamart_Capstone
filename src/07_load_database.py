
import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

conn = sqlite3.connect(ROOT / "data" / "instamart.db")

# Load cleaned customer and order data
customers = pd.read_csv(ROOT / "data" / "clean" / "customers_clean.csv")
orders = pd.read_csv(ROOT / "data" / "clean" / "orders_clean.csv")

# Load raw product data
products = pd.read_csv(ROOT / "data" / "raw" / "products.csv")

# Save all three tables into SQLite
customers.to_sql("customers", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)

print("Database tables loaded successfully!")

print("\nTables:")
print(pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table'",
    conn
))

conn.close()
