
import pandas as pd

# Load tickets dataset
tickets = pd.read_csv("data/raw/tickets.csv")

# Count tickets in each category
print("Ticket category distribution:")
print(tickets["true_category"].value_counts())

# Check duplicate ticket IDs
print("\nDuplicate ticket IDs:")
print(tickets["ticket_id"].duplicated().sum())

# Check missing complaint text
print("\nTickets with missing text:")
print(tickets["text"].isnull().sum())

# Check total number of tickets
print("\nTotal tickets:", len(tickets))
