
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load cleaned tickets
tickets = pd.read_csv(
    ROOT / "data" / "clean" / "tickets_clean.csv"
)

# Count tickets in each category
category_counts = tickets["true_category"].value_counts()

print("CUSTOMER COMPLAINT CATEGORIES:")
print(category_counts)

# Save category counts as a CSV report
output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

category_counts.rename_axis("category").reset_index(
    name="ticket_count"
).to_csv(
    output_dir / "ticket_category_distribution.csv",
    index=False
)

print("\nCategory distribution report saved successfully!")
