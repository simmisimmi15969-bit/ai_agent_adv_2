from pathlib import Path
import chromadb
from step1_chunk import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, SUPPORTED_TYPES, load_documents, split_documents
from step2_embed import embed

DB_DIR = Path(__file__).parent / "chroma_db"
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_or_create_collection(name="my_docs", metadata={"hnsw:space": "cosine"})

collection.delete(where={"source":"manual_test"})

current_files = {p.name for p in DOCS_DIR.glob("*") if p.suffix.lower() in SUPPORTED_TYPES}
stored = collection.get(include=["metadatas"])
for source in {m["source"] for m in stored["metadatas"]} - current_files:
    collection.delete(where={"source": source})
    print(f"removed {source}: file no longer in docs/")

for path in DOCS_DIR.glob("*"):
    if path.suffix.lower() not in SUPPORTED_TYPES:
        continue

    docs = load_documents(path)
    chunks = [c for c in split_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP) if c.page_content.strip()]
    if not chunks:
        print(f"skipping empty document: {path.name} (scanned PDF?)")
        continue

    metadatas = []
    for i, chunk in enumerate(chunks):
        meta = {"source": path.name, "chunk": i}
        if "page" in chunk.metadata:
            meta["page"] = chunk.metadata["page"] + 1
        metadatas.append(meta)

    collection.delete(where={"source": path.name})
    collection.add(
        ids=[f"{path.name}_{i}" for i in range(len(chunks))],
        embeddings=embed([c.page_content for c in chunks]),
        documents=[c.page_content for c in chunks],
        metadatas=metadatas,
    )
    print(f"indexed {path.name}: {len(chunks)} chunks")

print(f"\nTotal records in collection: {collection.count()}")
