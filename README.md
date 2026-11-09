
# Instamart Customer Intelligence

## Overview

This project develops a customer intelligence solution for Instamart using Python, SQLite, machine learning, support ticket classification, policy-based retrieval, and an AI refund support agent.

The project is divided into four parts:

- **Part A – Data Foundation:** Data cleaning, exploratory analysis, SQLite database creation, SQL queries, and sales analysis.
- **Part B – Customer Intelligence:** RFM analysis, K-Means customer segmentation, PCA, lapse prediction, and retention recommendations.
- **Part C – Support Ticket Triage:** PII redaction, ticket classification using an offline mock LLM, retry handling, token-budget demonstration, and automated routing.
- **Part D – Policy RAG and Refund Agent:** Policy retrieval, evaluation, agent tools, ReAct execution, safety limits, and checkpoint recovery.

## Requirements

- Python 3.11 or later
- SQLite 3.25 or later
- pandas
- numpy
- scikit-learn
- joblib

## Reproduce

Run all commands from the project root directory.

### 1. Create and activate the environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate project data

```bash
python generate_data.py
```

### 4. Run Part A – Data Foundation

```bash
python src/01_customer_cleaning.py
python src/02_clean_customers.py
python src/03_inspect_orders.py
python src/04_clean_orders.py
python src/05_sales_analysis.py
python src/06_create_database.py
python src/07_load_database.py
python src/08_inspect_tickets.py
python src/09_ticket_quality.py
python src/10_clean_tickets.py
python src/11_ticket_category_analysis.py
python src/12_ticket_text_preprocessing.py
python src/13_train_ticket_model.py
python src/14_test_predictions.py
```

SQL analysis queries are available in `sql/queries.sql`.

### 5. Run Part D – Policy RAG and Refund Agent

```bash
python src/15_policy_evaluation.py
python src/16_agent_simulation.py
python src/18_evaluate_policy_rag.py
python src/20_react_agent.py
```

The agent tools are implemented in `src/19_agent_tools.py` and used by the ReAct agent.

### 6. Run Part B – Customer Intelligence

```bash
python src/21_customer_segmentation.py
python src/22_lapse_prediction.py
```

### 7. Run Part C – Support Ticket Triage

```bash
python src/23_ticket_triage.py
```

The supplied `src/mock_llm.py` is used as an offline deterministic stand-in. No external API key or network access is required.

## Part A – Data Foundation

Includes customer and order cleaning, SQLite database creation, SQL analysis, and sales analysis.

## Part B – Customer Intelligence

- 1,175 customers segmented using K-Means.
- Four customer segments identified: Premium, Loyal, Regular, and At Risk.
- K=4 silhouette score: 0.5257.
- PCA explained variance: approximately 85.22%.
- Lapse prediction evaluated using Logistic Regression, Random Forest, and an oversampled Logistic Regression model.
- Random Forest selected based on cross-validation F1 score.

## Part C – Support Ticket Triage

- 164 raw tickets processed.
- Email, phone number, and order ID redaction implemented.
- 156 eligible tickets; 4 empty and 4 long tickets flagged.
- Zero-shot and few-shot classification evaluated using the offline mock LLM.
- Few-shot accuracy: 100% among classified tickets.
- Invalid outputs, retries, token usage, and automated routing recorded.

## Part D – Policy RAG and Refund Agent

Includes policy chunking, TF-IDF retrieval, policy evaluation, five agent tools, ReAct execution, safety limits, and checkpoint recovery.

Agent scenario tests:

- S1: REFUND – INR 128.25
- S2: REFUND – INR 135.00
- S3: ESCALATE
- S4: DENIED
- S5: DENIED
- S6: NOT_FOUND
- S7: FALLBACK_MAX_ITERATIONS

Checkpoint recovery, token-limit fallback, and timeout fallback tests were also performed.

## Outputs

Generated reports and model artifacts are stored in `outputs/`.

## Limitations and Next Steps

- The mock LLM is an offline deterministic simulation, not a production language model.
- TF-IDF retrieval depends on keyword overlap and may miss semantic relationships.
- Model predictions require further validation before production use.
- The project should be tested from a clean clone before submission.
