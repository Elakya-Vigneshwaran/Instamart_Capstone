
import pandas as pd
import re
from pathlib import Path


# 1. Define project folders
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


# 2. Load datasets
scenarios = pd.read_csv(DATA_DIR / "raw" / "agent_scenarios.csv")

orders = pd.read_csv(DATA_DIR / "clean" / "orders_clean.csv")

customers = pd.read_csv(DATA_DIR / "clean" / "customers_clean.csv")


# 3. Customer support decision function
def handle_customer_request(message, order, customer):

    amount = float(order["order_amount"])
    status = str(order["status"]).lower()
    delivery_time = order["delivery_minutes"]

    # Rule 1: Check whether order exists
    if order.empty:
        return "ESCALATE", 0, "Order not found. Human review required."

    # Rule 2: Check order status
    if status != "completed":
        return "DENIED", 0, "Cancelled or incomplete orders are not eligible for refunds."

    # Rule 3: Check delivery time
    if pd.isna(delivery_time):
        return "ESCALATE", 0, "Delivery information is missing. Human review required."

    if delivery_time <= 45:
        return "DENIED", 0, "Delivery was within the promised refund eligibility limit."

    # Rule 4: Check refund amount
    if amount > 1000:
        return "ESCALATE", amount, "Refund above INR 1000 requires human support lead approval."

    return "REFUND", amount, "Eligible for automatic full refund."


# 4. Process all agent scenarios
results = []

for _, scenario in scenarios.iterrows():

    message = scenario["message"]
    customer_id = scenario["customer_id"]

        # Rule 1: Check for LOOP condition
    if "[LOOP]" in message.upper():
        results.append({
            "scenario_id": scenario["scenario_id"],
            "customer_id": customer_id,
            "message": message,
            "predicted_outcome": "FALLBACK",
            "refund_amount_inr": 0,
            "reason": "Repeated processing detected. Fallback activated.",
            "expected_outcome": scenario["expected_outcome"],
            "expected_refund_inr": scenario["expected_refund_inr"],
            "outcome_correct": "FALLBACK" == scenario["expected_outcome"]
        })
        continue

    # Rule 2: Extract order ID
    order_ids = re.findall(r"ORD-\d+", message)
    if not order_ids:
        results.append({
            "scenario_id": scenario["scenario_id"],
            "customer_id": customer_id,
            "message": message,
            "predicted_outcome": "ESCALATE",
            "refund_amount_inr": 0,
            "reason": "Order ID not found in customer message.",
            "expected_outcome": scenario["expected_outcome"],
            "expected_refund_inr": scenario["expected_refund_inr"],
            "outcome_correct": False
        })
        continue

    order_id = order_ids[0]

    # Find matching order and customer
    order_match = orders[orders["order_id"] == order_id]
    customer_match = customers[
        customers["customer_id"] == customer_id
    ]

    if order_match.empty or customer_match.empty:
        predicted_outcome = "NOT_FOUND"
        refund_amount = 0
        reason = "Order or customer record not found."
    
    else:
        order = order_match.iloc[0]
        customer = customer_match.iloc[0]

        predicted_outcome, refund_amount, reason = (
            handle_customer_request(message, order, customer)
        )

    # Compare predicted and expected outcomes
    outcome_correct = (
        predicted_outcome == scenario["expected_outcome"]
    )

    results.append({
        "scenario_id": scenario["scenario_id"],
        "customer_id": customer_id,
        "message": message,
        "predicted_outcome": predicted_outcome,
        "refund_amount_inr": refund_amount,
        "reason": reason,
        "expected_outcome": scenario["expected_outcome"],
        "expected_refund_inr": scenario["expected_refund_inr"],
        "outcome_correct": outcome_correct
    })


# 5. Create results dataframe
results_df = pd.DataFrame(results)


# 6. Calculate performance
total = len(results_df)

correct = results_df["outcome_correct"].sum()

accuracy = (correct / total) * 100 if total > 0 else 0


# 7. Display results
print("\n========== CUSTOMER SUPPORT AGENT RESULTS ==========\n")

print(results_df.to_string(index=False))

print("\nTotal scenarios:", total)

print("Correct decisions:", correct)

print("Agent decision accuracy:", round(accuracy, 2), "%")


# 8. Save results
OUTPUT_DIR.mkdir(exist_ok=True)

output_file = OUTPUT_DIR / "agent_simulation_results.csv"

results_df.to_csv(output_file, index=False)

print("\nAgent simulation results saved to:", output_file)
