import re
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).parent.parent / "my_agent" / "docs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

SUPPORTED_TYPES = {".pdf", ".docx", ".txt", ".md"}
def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)

def clean_text(text: str) -> str:
    text = re.sub(r"\n \n", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_documents(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    docs = loader.load()
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    return docs


def split_documents(docs: list[Document], chunk_size: int, overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    for path in DOCS_DIR.glob("*"):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue

        text = extract_text(path)
        chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

        print(f"\n========== {path.name}: {len(text)} characters -> {len(chunks)} chunks ==========")
        for i, chunk in enumerate(chunks):
            print(f"\n--- chunk {i} ---")
            print(chunk)
