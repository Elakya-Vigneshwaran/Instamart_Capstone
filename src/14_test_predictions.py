
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load cleaned tickets
tickets = pd.read_csv("data/clean/tickets_clean.csv")

X = tickets["text"]
y = tickets["true_category"]

# Recreate and train the model using all cleaned tickets
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r'(?u)\b[a-zA-Z]{2,}\b'
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X, y)

# Save trained model
joblib.dump(model, "outputs/ticket_classifier.pkl")

print("Model saved successfully!")

# Test with new complaints
new_complaints = [
    "My order was delivered very late",
    "I was charged twice for my purchase",
    "Someone used my card without permission",
    "The application keeps crashing when I open it"
]

# Predict categories
# Test a technical complaint using different wording
test_complaints = [
    "The application keeps crashing when I open it",
    "The app is not working and shows an error",
    "The application freezes whenever I try to use it",
    "I cannot open the app because of a technical error"
]

# Predict categories and probabilities
predictions = model.predict(test_complaints)
probabilities = model.predict_proba(test_complaints)

print("\nTechnical Complaint Prediction Results:")

for complaint, prediction, probability in zip(
    test_complaints, predictions, probabilities
):
    print("\nComplaint:", complaint)
    print("Predicted category:", prediction)

    for category, score in zip(model.classes_, probability):
        print(f"{category}: {score * 100:.2f}%")