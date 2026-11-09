
# ==========================================
# INSTAMART CUSTOMER INTELLIGENCE PROJECT
# PART A: DATA FOUNDATION
# STEP 1: SALES ANALYSIS
# ==========================================

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load datasets
orders = pd.read_csv(ROOT / "data" / "clean" / "orders_clean.csv")
products = pd.read_csv(ROOT / "data" / "raw" / "products.csv")
customers = pd.read_csv(ROOT / "data" / "clean" / "customers_clean.csv")

print("ALL DATASETS LOADED SUCCESSFULLY")

# Merge orders with products
sales = pd.merge(orders, products, on="product_id", how="left")

# Merge sales with customers
sales = pd.merge(
    sales,
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left"
)

print("\nMERGED DATASET")
print(sales.head())

# Select completed orders with valid amounts
completed_sales = sales[
    (sales["status"] == "completed") &
    (sales["order_amount"].notna()) &
    (sales["order_amount"] > 0)
].copy()

# Total revenue
total_revenue = completed_sales["order_amount"].sum()

print("\nTOTAL REVENUE")
print("INR", round(total_revenue, 2))

# Revenue by category
category_revenue = completed_sales.groupby(
    "category"
)["order_amount"].sum().sort_values(ascending=False)

print("\nREVENUE BY CATEGORY")
print(category_revenue)

# Revenue by city
city_revenue = completed_sales.groupby(
    "city"
)["order_amount"].sum().sort_values(ascending=False)

print("\nREVENUE BY CITY")
print(city_revenue)

# Top five products
top_products = completed_sales.groupby(
    "product_name"
)["order_amount"].sum().sort_values(ascending=False).head(5)

print("\nTOP 5 PRODUCTS")
print(top_products)

# Save category revenue as a CSV report
output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

category_revenue.to_csv(output_dir / "category_revenue.csv")

print("\nCategory revenue report saved.")
print("SALES ANALYSIS COMPLETED SUCCESSFULLY!")
