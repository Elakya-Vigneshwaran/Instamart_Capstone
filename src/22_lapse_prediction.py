
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier
from sklearn.base import clone

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# -------------------------------------------------
# 1. Load datasets
# -------------------------------------------------
customers = pd.read_csv(ROOT / "data/clean/customers_clean.csv")
orders = pd.read_csv(ROOT / "data/clean/orders_clean.csv")
products = pd.read_csv(ROOT / "data/raw/products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

cutoff = pd.Timestamp("2025-09-01")
end_date = pd.Timestamp("2025-10-01")

# -------------------------------------------------
# 2. Create features using only pre-September data
# -------------------------------------------------
historical = orders[
    (orders["order_ts"] < cutoff)
].copy()

completed = historical[
    historical["status"] == "completed"
].copy()

# RFM features
rfm = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum")
).reset_index()

rfm["recency"] = (cutoff - rfm["last_order"]).dt.days
rfm = rfm.drop(columns="last_order")

# Other historical order features
other = historical.groupby("customer_id").agg(
    avg_discount=("discount_pct", "mean"),
    cancellation_rate=(
        "status",
        lambda x: (x == "cancelled").sum() / len(x)
    )
).reset_index()

delivery = completed.groupby("customer_id").agg(
    avg_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

# Most frequently purchased product category
category_orders = completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

category_counts = (
    category_orders.groupby(["customer_id", "category"])
    .size().reset_index(name="count")
)

category_counts = category_counts.sort_values(
    ["customer_id", "count", "category"],
    ascending=[True, False, True]
)

most_common = category_counts.drop_duplicates("customer_id")[
    ["customer_id", "category"]
].rename(columns={"category": "most_frequent_category"})

# Combine features
features = customers.copy()

features = features.merge(rfm, on="customer_id", how="inner")
features = features.merge(other, on="customer_id", how="left")
features = features.merge(delivery, on="customer_id", how="left")
features = features.merge(most_common, on="customer_id", how="left")

features["tenure_days"] = (
    cutoff - features["signup_date"]
).dt.days

features = features.drop(columns="signup_date")

# -------------------------------------------------
# 3. Create lapse target
# Lapsed = no completed order during September
# -------------------------------------------------
september_completed = orders[
    (orders["order_ts"] >= cutoff) &
    (orders["order_ts"] < end_date) &
    (orders["status"] == "completed")
]

active_ids = set(september_completed["customer_id"])

features["lapsed"] = (
    ~features["customer_id"].isin(active_ids)
).astype(int)

X = features.drop(columns=["customer_id", "lapsed"])
y = features["lapsed"]

print("Customers in lapse cohort:", len(features))
print("Lapsed customers:", int(y.sum()))
print("Lapse percentage:", round(y.mean() * 100, 2), "%")

# -------------------------------------------------
# 4. Train/test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_features = [
    "recency", "frequency", "monetary",
    "avg_discount", "cancellation_rate",
    "avg_delivery_minutes", "age", "tenure_days"
]

categorical_features = [
    "city", "gender", "most_frequent_category"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# -------------------------------------------------
# 5. Baseline model
# -------------------------------------------------
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

print("\nBaseline accuracy:", round(
    accuracy_score(y_test, baseline_predictions), 4
))
print("Baseline F1:", round(
    f1_score(y_test, baseline_predictions, zero_division=0), 4
))

# -------------------------------------------------
# 6. Logistic Regression tuning
# -------------------------------------------------
lr_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("model", LogisticRegression(max_iter=1000))
])

lr_search = GridSearchCV(
    lr_pipeline,
    {"model__C": [0.01, 0.1, 1, 10, 100]},
    scoring="f1",
    cv=5,
    n_jobs=-1
)

lr_search.fit(X_train, y_train)

print("\nBest Logistic Regression parameters:", lr_search.best_params_)
print("Logistic Regression CV F1:", round(lr_search.best_score_, 4))

# -------------------------------------------------
# 7. Random Forest tuning
# -------------------------------------------------
rf_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_search = GridSearchCV(
    rf_pipeline,
    {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    },
    scoring="f1",
    cv=5,
    n_jobs=-1
)

rf_search.fit(X_train, y_train)

print("\nBest Random Forest parameters:", rf_search.best_params_)
print("Random Forest CV F1:", round(rf_search.best_score_, 4))

# -------------------------------------------------
# 8. Compare models on test data
# -------------------------------------------------
lr_model = lr_search.best_estimator_
rf_model = rf_search.best_estimator_

lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

def show_metrics(name, actual, predicted):
    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(actual, predicted), 4))
    print("Precision:", round(precision_score(actual, predicted, zero_division=0), 4))
    print("Recall:", round(recall_score(actual, predicted, zero_division=0), 4))
    print("F1:", round(f1_score(actual, predicted, zero_division=0), 4))

show_metrics("Logistic Regression", y_test, lr_predictions)
show_metrics("Random Forest", y_test, rf_predictions)

# -------------------------------------------------
# 9. Oversample training data for Logistic Regression
# -------------------------------------------------
train_data = X_train.copy()
train_data["lapsed"] = y_train.values

majority_count = train_data["lapsed"].value_counts().max()

balanced_parts = []

for class_value in train_data["lapsed"].unique():
    part = train_data[train_data["lapsed"] == class_value]

    balanced_parts.append(
        part.sample(
            n=majority_count,
            replace=True,
            random_state=42
        )
    )

balanced_train = pd.concat(balanced_parts).sample(
    frac=1,
    random_state=42
)

X_balanced = balanced_train.drop(columns="lapsed")
y_balanced = balanced_train["lapsed"]

balanced_lr = clone(lr_model)
balanced_lr.fit(X_balanced, y_balanced)

balanced_predictions = balanced_lr.predict(X_test)

show_metrics(
    "Oversampled Logistic Regression",
    y_test,
    balanced_predictions
)

# -------------------------------------------------
# 10. Select best model using cross-validation F1
# -------------------------------------------------
if rf_search.best_score_ >= lr_search.best_score_:
    best_model = rf_model
    best_name = "Random Forest"
else:
    best_model = lr_model
    best_name = "Logistic Regression"

print("\nSelected model:", best_name)

# -------------------------------------------------
# 11. Feature importance / coefficient analysis
# -------------------------------------------------
fitted_preprocessor = best_model.named_steps["preprocessor"]
feature_names = fitted_preprocessor.get_feature_names_out()
classifier = best_model.named_steps["model"]

if best_name == "Random Forest":
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": classifier.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 5 Random Forest features:")
    print(importance.head(5).to_string(index=False))

else:
    coefficients = pd.DataFrame({
        "feature": feature_names,
        "coefficient": classifier.coef_[0]
    })

    print("\nTop 3 positive lapse indicators:")
    print(coefficients.sort_values(
        "coefficient", ascending=False
    ).head(3).to_string(index=False))

    print("\nTop 3 negative lapse indicators:")
    print(coefficients.sort_values(
        "coefficient"
    ).head(3).to_string(index=False))

# -------------------------------------------------
# 12. Save model and verify after reloading
# -------------------------------------------------
model_path = OUT / "lapse_model.joblib"

joblib.dump({
    "model": best_model,
    "features": list(X.columns),
    "model_name": best_name
}, model_path)

saved = joblib.load(model_path)
reloaded_predictions = saved["model"].predict(X_test)

print("\nReloaded predictions identical:",
      np.array_equal(best_model.predict(X_test), reloaded_predictions))

# -------------------------------------------------
# 13. Top 10 Premium customers in test set
# -------------------------------------------------
probabilities = saved["model"].predict_proba(X_test)[:, 1]

retention = pd.DataFrame({
    "customer_id": features.loc[X_test.index, "customer_id"].values,
    "lapse_probability": probabilities,
    "actually_lapsed": y_test.values
})

premium_ids = set(
    pd.read_csv(OUT / "customer_segments.csv")
    .query("is_premium == True")["customer_id"]
)

retention = retention[
    retention["customer_id"].isin(premium_ids)
].sort_values("lapse_probability", ascending=False).head(10)

retention.to_csv(
    OUT / "premium_retention_list.csv",
    index=False
)

print("\nTop Premium customers for retention:")
print(retention.to_string(index=False))

print("\nSaved:")
print("outputs/lapse_model.joblib")
print("outputs/premium_retention_list.csv")
