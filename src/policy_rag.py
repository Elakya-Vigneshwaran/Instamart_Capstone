
import mock_llm
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Project folder
BASE_DIR = Path(__file__).resolve().parent.parent
POLICY_DIR = BASE_DIR / "data" / "policies"


# 2. Split policy documents into chunks
def create_chunks(chunk_size=40, overlap=10):

    chunks = []

    for file in sorted(POLICY_DIR.glob("*_policy.txt")):

        text = file.read_text(encoding="utf-8")
        words = text.split()

        step = chunk_size - overlap

        for start in range(0, len(words), step):

            chunk_words = words[start:start + chunk_size]

            chunks.append({
                "source": file.name,
                "text": " ".join(chunk_words)
            })

            if start + chunk_size >= len(words):
                break

    return chunks


# 3. Build a TF-IDF index
def build_index(overlap=10):

    chunks = create_chunks(chunk_size=40, overlap=overlap)

    chunk_texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunk_texts)

    return chunks, vectorizer, chunk_vectors


# Default index: 40 words with 10-word overlap
chunks, vectorizer, chunk_vectors = build_index(overlap=10)


# 4. Retrieve the most relevant chunks
def retrieve(query, k=3, overlap=10):

    if overlap == 10:
        current_chunks = chunks
        current_vectorizer = vectorizer
        current_vectors = chunk_vectors

    else:
        current_chunks, current_vectorizer, current_vectors = build_index(
            overlap=overlap
        )

    query_vector = current_vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        current_vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1][:k]

    results = []

    for index in ranked_indices:

        results.append({
            "source": current_chunks[index]["source"],
            "score": float(similarities[index]),
            "text": current_chunks[index]["text"]
        })

    return results


# 5. Generate an answer using the mock LLM
def answer(question, overlap=10):

    results = retrieve(question, k=3, overlap=overlap)

    context = ""

    for result in results:
        context += f"[{result['source']}]\n"
        context += result["text"] + "\n"

    user_message = (
        f"CONTEXT:\n{context}\n"
        f"QUESTION:\n{question}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT\n"
                "Answer only using the provided context. "
                "If the answer is not available, say: "
                "I could not find that information in the policy documents."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = mock_llm.chat(messages)

    answer_text = response.choices[0].message.content

    sources = list(dict.fromkeys(
        result["source"] for result in results
    ))

    return answer_text, sources


# 6. Test the system
if __name__ == "__main__":

    print("Total chunks:", len(chunks))

    print("\nChunks per policy:")
    print(Counter(chunk["source"] for chunk in chunks))

    results = retrieve("refund policy late delivery", k=3)

    print("\nTop 3 retrieved chunks:")

    for result in results:
        print("\nSource:", result["source"])
        print("Score:", round(result["score"], 4))
        print("Text:", result["text"])

    question = "What happens if my completed order is delivered late?"

    answer_text, sources = answer(question)

    print("\nQuestion:", question)
    print("\nGenerated Answer:", answer_text)
    print("\nSources:", sources)
