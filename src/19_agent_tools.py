
import sqlite3
import json
from pathlib import Path
from datetime import datetime


# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "data" / "instamart.db"

LEDGER_PATH = BASE_DIR / "outputs" / "refund_ledger.jsonl"


# 2. Tool 1: Get order details
def get_order(order_id):

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE order_id = ?
        """,
        (order_id,)
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return {
            "error": "Order not found"
        }

    return dict(row)


# 3. Tool 2: Search policy
def search_policy(query):

    from policy_rag import retrieve

    results = retrieve(query, k=3)

    return {
        "results": results
    }


# 4. Tool 3: Calculate refund
def calculate_refund(order_amount, delivery_minutes, status="completed"):

    if status == "completed" and delivery_minutes is not None and delivery_minutes > 45:

        refund = float(order_amount)

    else:

        refund = 0.0

    return {
        "refund_inr": round(refund, 2)
    }


# 5. Common ledger function
def write_ledger(tool_name, details):

    LEDGER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    record = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        **details
    }

    with open(LEDGER_PATH, "a", encoding="utf-8") as file:

        file.write(json.dumps(record) + "\n")


# 6. Tool 4: Issue refund
def issue_refund(order_id, refund_inr):

    write_ledger(
        "issue_refund",
        {
            "order_id": order_id,
            "refund_inr": refund_inr
        }
    )

    return {
        "order_id": order_id,
        "refund_inr": refund_inr,
        "message": "Refund recorded successfully"
    }


# 7. Tool 5: Escalate to human
def escalate_to_human(order_id, reason, refund_inr=0):

    write_ledger(
        "escalate_to_human",
        {
            "order_id": order_id,
            "reason": reason,
            "refund_inr": refund_inr
        }
    )

    return {
        "order_id": order_id,
        "message": "Escalated to human support lead"
    }


# 8. Tool descriptions
TOOLS_SPEC = [

    {
        "name": "get_order",
        "description": "Retrieve order details using an order ID.",
        "parameters": {
            "order_id": "string"
        }
    },

    {
        "name": "search_policy",
        "description": "Search policy documents for relevant information.",
        "parameters": {
            "query": "string"
        }
    },

    {
        "name": "calculate_refund",
        "description": "Calculate the eligible refund for a completed late order.",
        "parameters": {
            "order_amount": "number",
            "delivery_minutes": "number",
            "status": "string"
        }
    },

    {
        "name": "issue_refund",
        "description": "Record an approved refund in the refund ledger.",
        "parameters": {
            "order_id": "string",
            "refund_inr": "number"
        }
    },

    {
        "name": "escalate_to_human",
        "description": "Send a refund request to a human support lead.",
        "parameters": {
            "order_id": "string",
            "reason": "string",
            "refund_inr": "number"
        }
    }

]


# 9. Tool map
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


# 10. Basic tool test
if __name__ == "__main__":

    print("Available tools:")
    print(list(TOOL_MAP.keys()))

    print("\nTesting refund calculation:")

    print(
        calculate_refund(
            order_amount=128.25,
            delivery_minutes=60,
            status="completed"
        )
    )
