
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent

orders = pd.read_csv(ROOT / "data" / "clean" / "orders_clean.csv")
customers = pd.read_csv(ROOT / "data" / "clean" / "customers_clean.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Use completed orders before the analysis date
cutoff = pd.Timestamp("2025-10-01")
completed = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] < cutoff)
].copy()

# Calculate RFM metrics
rfm = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum")
).reset_index()

rfm["recency"] = (cutoff - rfm["last_order"]).dt.days
rfm = rfm.drop(columns=["last_order"])

# Prepare clustering data
features = ["recency", "frequency", "monetary"]
X = rfm[features].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Evaluate K values
print("K-Means evaluation:")
for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K={k}: Silhouette={score:.4f}, Inertia={model.inertia_:.2f}")

# Final clustering: K=4
model = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm["segment_id"] = model.fit_predict(X_scaled)

# Name segments based on their RFM characteristics
summary = rfm.groupby("segment_id")[features].mean()
ranked = summary.sort_values("monetary", ascending=False)

names = [
    "Premium",
    "Loyal",
    "Regular",
    "At Risk"
]

name_map = dict(zip(ranked.index, names))
rfm["segment_name"] = rfm["segment_id"].map(name_map)

# Mark Premium customers
rfm["is_premium"] = (rfm["segment_name"] == "Premium")

# PCA analysis
pca = PCA(n_components=2)
pca.fit(X_scaled)

print("\nPCA explained variance:", pca.explained_variance_ratio_)
print("Total PCA variance:", pca.explained_variance_ratio_.sum())

# Save results
output = ROOT / "outputs"
output.mkdir(exist_ok=True)

rfm.to_csv(output / "customer_segments.csv", index=False)

print("\nCustomer segmentation completed!")
print("Customers segmented:", len(rfm))
print("\nSegment counts:")
print(rfm["segment_name"].value_counts())
print("\nSaved: outputs/customer_segments.csv")
