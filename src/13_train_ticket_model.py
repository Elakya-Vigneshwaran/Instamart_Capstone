
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load cleaned tickets
tickets = pd.read_csv("data/clean/tickets_clean.csv")

# Input: customer complaint text
X = tickets["text"]

# Output: actual complaint category
y = tickets["true_category"]

# Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r'(?u)\b[a-zA-Z]{2,}\b'
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Predict complaint categories
y_pred = model.predict(X_test)

# Evaluate model
print("Training completed successfully!")

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nModel Accuracy:")
print(round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
