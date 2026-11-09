
import sys
import re
import json
import time
import random
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import mock_llm

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

TOKEN_BUDGET = 100000
DEMO_BUDGET = 6000
MAX_RETRIES = 3
BASE_DELAY = 0.05
MAX_TOKENS_PER_TICKET = 500

random.seed(42)

# -------------------------------------------------
# 1. Load tickets
# -------------------------------------------------
tickets = pd.read_csv(ROOT / "data/raw/tickets.csv")
segments = pd.read_csv(OUT / "customer_segments.csv")

segment_lookup = segments.set_index("customer_id")["segment_name"].to_dict()

print("Total raw tickets:", len(tickets))

# -------------------------------------------------
# 2. PII redaction
# -------------------------------------------------
PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "phone": r"\b[6-9]\d{9}\b",
    "order_id": r"ORD-\d{6}"
}

def redact_pii(text):
    text = str(text)

    counts = {}

    for name, pattern in PATTERNS.items():
        counts[name] = len(re.findall(pattern, text))
        text = re.sub(
            pattern,
            {
                "email": "[EMAIL]",
                "phone": "[PHONE]",
                "order_id": "[ORDER_ID]"
            }[name],
            text
        )

    return text, counts


redacted_texts = []
redaction_counts = {"email": 0, "phone": 0, "order_id": 0}

for value in tickets["text"]:
    if pd.isna(value) or not str(value).strip():
        redacted_texts.append("")
        continue

    cleaned, counts = redact_pii(value)
    redacted_texts.append(cleaned)

    for key in counts:
        redaction_counts[key] += counts[key]

tickets["redacted_text"] = redacted_texts

print("\nPII redaction counts:")
print(redaction_counts)

# Verify second pass
second_pass_count = 0

for value in tickets["redacted_text"]:
    _, counts = redact_pii(value)
    second_pass_count += sum(counts.values())

print("PII matches after second pass:", second_pass_count)

# -------------------------------------------------
# 3. Guardrails
# -------------------------------------------------
def initial_status(row):
    text = row["redacted_text"]

    if not text.strip():
        return "skipped_empty"

    if mock_llm.est_tokens(text) > MAX_TOKENS_PER_TICKET:
        return "flagged_long"

    return "eligible"


tickets["initial_status"] = tickets.apply(initial_status, axis=1)

print("\nGuardrail counts:")
print(tickets["initial_status"].value_counts())

# -------------------------------------------------
# 4. Classification prompts
# -------------------------------------------------
ZERO_SYSTEM = """
CLASSIFY
You are a customer support ticket classifier.
Classify the ticket into exactly one category:
billing, delivery, technical, fraud.
Return JSON only with keys category and reason.
"""

FEW_SYSTEM = """
CLASSIFY
You are a customer support ticket classifier.
Classify the ticket into exactly one category:
billing, delivery, technical, fraud.
Return JSON only with keys category and reason.

EXAMPLES:
Ticket: Unauthorized transaction from my account.
Answer: {"category":"fraud","reason":"unauthorized transaction"}

Ticket: My app crashes whenever I log in.
Answer: {"category":"technical","reason":"application crash"}

Ticket: I was charged twice for my order.
Answer: {"category":"billing","reason":"duplicate charge"}

Ticket: My delivery has not arrived.
Answer: {"category":"delivery","reason":"late delivery"}
"""

# -------------------------------------------------
# 5. Parse model output
# -------------------------------------------------
def parse_response(content):
    if not content:
        return None, "flagged_invalid_output"

    try:
        result = json.loads(content)

        if not isinstance(result, dict):
            return None, "flagged_invalid_output"

        category = result.get("category")

        if category not in CATEGORIES:
            return None, "flagged_invalid_output"

        return category, "classified_json"

    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", content.lower())
        matches = set(words) & CATEGORIES

        if len(matches) == 1:
            return matches.pop(), "classified_keyword"

        return None, "flagged_invalid_output"


# -------------------------------------------------
# 6. Classification with retry and token tracking
# -------------------------------------------------
retry_logs = []

def classify_ticket(ticket_id, text, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(messages)

            content = response.choices[0].message.content

            tokens = response.usage.total_tokens

            category, status = parse_response(content)

            return {
                "ticket_id": ticket_id,
                "category": category,
                "status": status,
                "tokens_used": tokens
            }

        except mock_llm.TransientLLMError:
            if attempt >= MAX_RETRIES:
                retry_logs.append({
                    "ticket_id": ticket_id,
                    "retry_number": attempt + 1,
                    "wait_seconds": 0,
                    "result": "failed_after_retries"
                })
                return {
                    "ticket_id": ticket_id,
                    "category": None,
                    "status": "flagged_invalid_output",
                    "tokens_used": 0
                }

            wait = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)

            retry_logs.append({
                "ticket_id": ticket_id,
                "retry_number": attempt + 1,
                "wait_seconds": wait,
                "result": "retry"
            })

            time.sleep(wait)

    return {
        "ticket_id": ticket_id,
        "category": None,
        "status": "flagged_invalid_output",
        "tokens_used": 0
    }


# -------------------------------------------------
# 7. Run independent zero-shot or few-shot batch
# -------------------------------------------------
def run_batch(system_prompt, label):
    mock_llm.reset()

    results = []

    for _, row in tickets.iterrows():

        if row["initial_status"] != "eligible":
            results.append({
                "ticket_id": row["ticket_id"],
                "category": None,
                "status": row["initial_status"],
                "tokens_used": 0
            })
            continue

        result = classify_ticket(
            row["ticket_id"],
            row["redacted_text"],
            system_prompt
        )

        results.append(result)

    result_df = pd.DataFrame(results)

    eligible = result_df[
        result_df["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ].merge(
        tickets[["ticket_id", "true_category"]],
        on="ticket_id"
    )

    accuracy = (
        (eligible["category"] == eligible["true_category"]).mean()
        if len(eligible) else 0
    )

    total_tokens = int(result_df["tokens_used"].sum())
    classified_count = len(eligible)

    print(f"\n{label} RESULTS")
    print("Classified:", classified_count)
    print("JSON:", (result_df["status"] == "classified_json").sum())
    print("Keyword:", (result_df["status"] == "classified_keyword").sum())
    print("Invalid:", (result_df["status"] == "flagged_invalid_output").sum())
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Total tokens:", total_tokens)
    print(
        "Tokens per classified ticket:",
        round(total_tokens / classified_count, 2)
        if classified_count else 0
    )

    return result_df


zero_results = run_batch(ZERO_SYSTEM, "ZERO-SHOT")
few_results = run_batch(FEW_SYSTEM, "FEW-SHOT")

# Save comparison
comparison = pd.DataFrame([
    {
        "method": "zero_shot",
        "classified": int(zero_results["status"].isin(
            ["classified_json", "classified_keyword"]
        ).sum()),
        "tokens": int(zero_results["tokens_used"].sum())
    },
    {
        "method": "few_shot",
        "classified": int(few_results["status"].isin(
            ["classified_json", "classified_keyword"]
        ).sum()),
        "tokens": int(few_results["tokens_used"].sum())
    }
])

comparison.to_csv(OUT / "triage_comparison.csv", index=False)

# -------------------------------------------------
# 8. Token budget demonstration
# -------------------------------------------------
def run_budget_demo():
    mock_llm.reset()

    results = []
    tokens_spent = 0
    exhausted = False

    for _, row in tickets.iterrows():

        if row["initial_status"] != "eligible":
            results.append({
                "ticket_id": row["ticket_id"],
                "status": row["initial_status"],
                "tokens_used": 0
            })
            continue

        if tokens_spent >= DEMO_BUDGET:
            exhausted = True
            results.append({
                "ticket_id": row["ticket_id"],
                "status": "budget_exhausted",
                "tokens_used": 0
            })
            continue

        result = classify_ticket(
            row["ticket_id"],
            row["redacted_text"],
            ZERO_SYSTEM
        )

        tokens_spent += result["tokens_used"]

        results.append({
            "ticket_id": row["ticket_id"],
            "status": result["status"],
            "tokens_used": result["tokens_used"]
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT / "triage_budget_demo.csv", index=False)

    print("\nBUDGET DEMONSTRATION")
    print("Budget:", DEMO_BUDGET)
    print("Tokens used:", tokens_spent)
    print("Budget exhausted:", exhausted)
    print("Saved outputs/triage_budget_demo.csv")


run_budget_demo()

# -------------------------------------------------
# 9. Final few-shot run and ticket routing
# -------------------------------------------------
# few_results is the final full few-shot run.
final_results = tickets[
    ["ticket_id", "customer_id", "initial_status"]
].merge(
    few_results,
    on="ticket_id"
)

final_results["segment_name"] = (
    final_results["customer_id"].map(segment_lookup)
    .fillna("Non-Premium")
)

def route_ticket(row):
    status = row["status"]
    category = row["category"]
    segment = row["segment_name"]

    if status in [
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    ]:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if segment == "Premium":
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


routing = final_results.apply(
    route_ticket,
    axis=1,
    result_type="expand"
)

final_results["route"] = routing[0]
final_results["priority"] = routing[1]

final_results = final_results.rename(
    columns={"initial_status": "guardrail_status"}
)

final_results = final_results[
    [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used"
    ]
]

final_results.to_csv(
    OUT / "triage_results.csv",
    index=False
)

# Save retry logs
pd.DataFrame(retry_logs).to_csv(
    OUT / "triage_retry_log.csv",
    index=False
)

print("\nFINAL FEW-SHOT ROUTING")
print(final_results["route"].value_counts())
print("\nStatus counts:")
print(final_results["status"].value_counts())

print("\nRetry attempts:", len(retry_logs))
print("\nFinal rows:", len(final_results))

print("\nSaved:")
print("outputs/triage_results.csv")
print("outputs/triage_retry_log.csv")
print("outputs/triage_comparison.csv")
