from step1_chunk import DOCS_DIR, extract_text, split_into_chunks
import math
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).parent.parent / "my_agent" / ".env")
client = genai.Client()
EMBED_MODEL = "gemini-embedding-2"  

# def embed(texts:list[str]) -> list[list[float]]:
#     result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
#     return [e.values for e in result.embeddings]

def embed(texts: list[str]) -> list[list[float]]:
    vectors=[]
    for text in texts:
        result= client.models.embed_content(model=EMBED_MODEL, contents=text)               
        vectors.append(result.embeddings[0].values)
    return vectors

# vectors = embed(["Employees get 18 paid leaves per year."])
# print(f"Number of vectors: {len(vectors)}")
# print(f"Number in each vector: {len(vectors[0])}")
# print(f"first 5 numbers: {vectors[0][:5]}")

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

if __name__ == "__main__":

    sentences = [
        "When will I get my money?",
        "Employees get 18 paid leaves per year.",
        "Salary is credited on the last day of the month.",
        "The weather in Mohali is sunny today.",
    ]
    vectors = embed(sentences)
    print(f"Sentences: {len(sentences)}, vectors:{len(vectors)}")

    print(f"Similarity to: '{sentences[0]}'\n")
    for sentence, vector in zip(sentences, vectors):
        score = cosine_similarity(vectors[0], vector)
        print(f"  {score:.3f}  {sentence}")

    print("\n\n========== CHUNK SEARCH ==========")

    text = extract_text(DOCS_DIR / "hr_policy.txt")
    chunks = split_into_chunks(text, 300, 50)
    chunk_vectors = embed(chunks)

    question = "Can I work from my house?"
    question_vector = embed([question])[0]

    results = []
    for chunk, vector in zip(chunks, chunk_vectors):
        score = cosine_similarity(question_vector, vector)
        results.append((score, chunk))

    results.sort(reverse=True)

    print(f"Question: {question}\n")
    for score, chunk in results:
        print(f"--- score {score:.3f} ---")
        print(chunk.strip()[:120], "...\n")




