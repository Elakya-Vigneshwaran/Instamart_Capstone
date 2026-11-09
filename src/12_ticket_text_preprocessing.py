
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned tickets
tickets = pd.read_csv("data/clean/tickets_clean.csv")

# Extract complaint text
X = tickets["text"]

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=1000,
    token_pattern=r'(?u)\b[a-zA-Z]{2,}\b'
)

X_tfidf = vectorizer.fit_transform(X)

# Display results
print("Original dataset shape:", tickets.shape)
print("TF-IDF matrix shape:", X_tfidf.shape)

print("\nFirst 20 text features:")
print(vectorizer.get_feature_names_out()[:20])

print("\nText preprocessing completed successfully!")
