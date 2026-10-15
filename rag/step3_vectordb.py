from pathlib import Path

import chromadb

from step2_embed import embed

DB_DIR = Path(__file__).parent / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name="my_docs",
    metadata={"hnsw:space": "cosine"},
)

texts = [
    "Employees get 18 paid leaves per year.",
    "Employees can work from home 2 days per week.",
]

collection.add(
    ids=["test_0", "test_1"],
    embeddings=embed(texts),
    documents=texts,
    metadatas=[{"source": "manual_test"}, {"source": "manual_test"}],
)

print(f"Records in collection: {collection.count()}")

record = collection.get(ids=["test_0"], include=["documents", "metadatas", "embeddings"])
print(f"id:        {record['ids'][0]}")
print(f"document:  {record['documents'][0]}")
print(f"metadata:  {record['metadatas'][0]}")
print(f"embedding: {len(record['embeddings'][0])} numbers, first 3: {record['embeddings'][0][:3]}")
