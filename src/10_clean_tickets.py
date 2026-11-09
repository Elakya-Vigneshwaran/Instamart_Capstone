
import pandas as pd

# Load original tickets dataset
tickets = pd.read_csv("data/raw/tickets.csv")

# Remove tickets with missing or empty complaint text
tickets_clean = tickets.dropna(subset=["text"]).copy()

tickets_clean = tickets_clean[
    tickets_clean["text"].str.strip() != ""
].copy()

# Remove extra spaces from complaint text
tickets_clean["text"] = tickets_clean["text"].str.strip()

# Save cleaned dataset
tickets_clean.to_csv(
    "data/clean/tickets_clean.csv",
    index=False
)

# Display cleaning results
print("Original ticket count:", len(tickets))
print("Cleaned ticket count:", len(tickets_clean))

print("\nMissing values after cleaning:")
print(tickets_clean.isnull().sum())

print("\nCleaned dataset saved successfully!")
