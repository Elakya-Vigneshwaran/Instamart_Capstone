
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Set project folder locations
BASE_DIR = Path(__file__).resolve().parent.parent

POLICY_DIR = BASE_DIR / "data" / "policies"
DATA_FILE = BASE_DIR / "data" / "raw" / "policy_eval.csv"


# 2. Load all policy documents
policies = {}

for file in POLICY_DIR.glob("*_policy.txt"):
    policy_name = file.name
    policy_text = file.read_text(encoding="utf-8")
    policies[policy_name] = policy_text

print("Loaded policy documents:", list(policies.keys()))


# 3. Load evaluation questions
df = pd.read_csv(DATA_FILE)

# Fill missing expected facts with empty strings
df["expected_fact"] = df["expected_fact"].fillna("").astype(str)


# 4. Convert policy documents into TF-IDF vectors
policy_names = list(policies.keys())
policy_texts = [policies[name] for name in policy_names]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

policy_vectors = vectorizer.fit_transform(policy_texts)


# 5. Function to identify the correct policy
def find_policy(question):

    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector,
        policy_vectors
    )[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    # If similarity is too low, classify as unanswerable
    if best_score < 0.05:
        return "NONE", best_score

    return policy_names[best_index], best_score


# 6. Evaluate all questions
results = []

for _, row in df.iterrows():

    question = row["question"]
    expected_doc = row["expected_doc"]
    expected_fact = row["expected_fact"]

    predicted_doc, similarity_score = find_policy(question)

    # Check whether the expected fact exists in the selected policy
    if predicted_doc != "NONE" and expected_fact.strip():

        policy_text = policies[predicted_doc].lower()
        fact_found = expected_fact.lower() in policy_text

    else:
        fact_found = False

    # Compare predicted and expected policy
    document_correct = predicted_doc == expected_doc

    # A NONE question is correctly handled when NONE is predicted
    if expected_doc == "NONE":
        fact_correct = predicted_doc == "NONE"
    else:
        fact_correct = document_correct and fact_found

    results.append({
        "question": question,
        "expected_doc": expected_doc,
        "predicted_doc": predicted_doc,
        "similarity": round(float(similarity_score), 3),
        "document_correct": document_correct,
        "expected_fact": expected_fact,
        "fact_found": fact_found,
        "fact_correct": fact_correct
    })


# 7. Create evaluation results
results_df = pd.DataFrame(results)

total_questions = len(results_df)

document_accuracy = (
    results_df["document_correct"].sum() / total_questions
) * 100

fact_accuracy = (
    results_df["fact_correct"].sum() / total_questions
) * 100


# 8. Display results
print("\n========== POLICY EVALUATION RESULTS ==========")

print(results_df.to_string(index=False))

print("\nTotal questions:", total_questions)

print("Correct policy predictions:",
      results_df["document_correct"].sum())

print("Policy identification accuracy:",
      round(document_accuracy, 2), "%")

print("Correct policy and fact results:",
      results_df["fact_correct"].sum())

print("Overall evaluation accuracy:",
      round(fact_accuracy, 2), "%")


# 9. Save results
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

output_file = OUTPUT_DIR / "policy_evaluation_results.csv"

results_df.to_csv(output_file, index=False)

print("\nResults saved successfully to:", output_file)
