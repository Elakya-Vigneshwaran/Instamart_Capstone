results_summary = []
from pathlib import Path
import sys

import pandas as pd
import mock_llm

from policy_rag import retrieve, answer


# 1. Load evaluation questions
BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_FILE = BASE_DIR / "data" / "raw" / "policy_eval.csv"

df = pd.read_csv(EVAL_FILE)


# 2. Evaluate one overlap setting
def evaluate(overlap):

    print("\n" + "=" * 60)
    print("EVALUATION: OVERLAP =", overlap)
    print("=" * 60)

    mock_llm.reset()

    retrieval_correct = 0
    fact_correct = 0
    fallback_correct = 0

    answerable_count = 0
    unanswerable_count = 0

    failures = []

    for _, row in df.iterrows():

        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        # Retrieve top 3 chunks using the selected overlap
        results = retrieve(
            question,
            k=3,
            overlap=overlap
        )

        retrieved_sources = [
            result["source"] for result in results
        ]

        # Check expected document retrieval
        if expected_doc != "NONE":
            answerable_count += 1

            if expected_doc in retrieved_sources:
                retrieval_correct += 1

        else:
            unanswerable_count += 1

        # Generate answer
        answer_text, sources = answer(
            question,
            overlap=overlap
        )

        # Check unanswerable questions
        if expected_doc == "NONE":

            if answer_text.startswith(
                "I could not find"
            ):
                fallback_correct += 1

            else:
                failures.append({
                    "question": question,
                    "expected": "Fallback response",
                    "actual": answer_text
                })

        # Check answerable questions
        else:

            if pd.notna(expected_fact) and str(expected_fact) in answer_text:
                fact_correct += 1

            else:
                failures.append({
                    "question": question,
                    "expected": expected_fact,
                    "actual": answer_text
                })

        results_summary.append({
        "overlap": overlap,
        "total_questions": len(df),
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "retrieval_correct": retrieval_correct,
        "expected_facts_found": fact_correct,
        "fallback_correct": fallback_correct
    })

    # 3. Print results
    print("\nAnswerable questions:", answerable_count)
    print("Unanswerable questions:", unanswerable_count)

    print(
        "\nRetrieval accuracy:",
        retrieval_correct, "/", answerable_count
    )

    print(
        "Expected facts found:",
        fact_correct, "/", answerable_count
    )

    print(
        "Correct fallback responses:",
        fallback_correct, "/", unanswerable_count
    )

    print("\nFailed questions:")

    if not failures:
        print("No failures")

    for failure in failures:
        print("\nQuestion:", failure["question"])
        print("Expected:", failure["expected"])
        print("Actual:", failure["actual"])


# 4. Run both evaluations
if __name__ == "__main__":

    evaluate(overlap=10)

    evaluate(overlap=0)

    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    pd.DataFrame(results_summary).to_csv(
        output_dir / "policy_rag_evaluation.csv",
        index=False
    )

    print("\nEvaluation results saved successfully!")
