from langchain_text_splitters import RecursiveCharacterTextSplitter

from step1_chunk import DOCS_DIR, extract_text, split_into_chunks

text = extract_text(DOCS_DIR / "hr_policy.txt")

old_chunks = split_into_chunks(text, 300, 50)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
new_chunks = splitter.split_text(text)

print(f"========== YOUR SPLITTER: {len(old_chunks)} chunks ==========")
for i, chunk in enumerate(old_chunks):
    print(f"\n--- chunk {i} ({len(chunk)} chars) ---")
    print(chunk)

print(f"\n\n========== LANGCHAIN SPLITTER: {len(new_chunks)} chunks ==========")
for i, chunk in enumerate(new_chunks):
    print(f"\n--- chunk {i} ({len(chunk)} chars) ---")
    print(chunk)
