from pathlib import Path

import chromadb

DB_DIR = Path(__file__).parent / "chroma_db"
client = chromadb.PersistentClient(path=str(DB_DIR))
print("Collections client=====:", client)
print("Collections client list_collection=====:", client.list_collections())
print("Collections:", [c.name for c in client.list_collections()])

collection = client.get_collection("my_docs")
print(f"Records in 'my_docs': {collection.count()}\n")
print(f"Collection metadata: {collection}")

data = collection.get(include=["documents", "metadatas", "embeddings"])

for i in range(len(data["ids"])):
    print(f"--- {data['ids'][i]} ---")
    print(f"document:  {data['documents'][i][:80]}")
    print(f"metadata:  {data['metadatas'][i]}")
    print(f"embedding: {len(data['embeddings'][i])} numbers, first 5: {data['embeddings'][i][:5].round(4)}\n")
