"""Build the ChromaDB index over the business documents.

Production mode uses OpenAI embeddings. When no API key is configured, the
script builds a deterministic local hashing index so the complete demo (and its
tests) still runs offline. The chosen backend is written next to the index so
queries always use the same vector space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DOCS_DIR,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MARKER,
    embedding_model,
    offline_mode,
    openai_api_key,
)

LOCAL_BACKEND = "local-hash-v1"


class HashingEmbeddings(Embeddings):
    """A small deterministic embedding model for offline demos.

    It combines word, word-bigram, and character-trigram features hashed into a
    fixed-size signed vector. This is not a replacement for a semantic embedding
    model, but it keeps ChromaDB retrieval functional without network access.
    """

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions

    @staticmethod
    def _features(text: str) -> Iterable[tuple[str, float]]:
        normalized = text.lower().replace("$", " ")
        normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
        tokens = re.findall(r"[a-z0-9%]+", normalized)
        for token in tokens:
            yield f"w:{token}", 2.0
        for left, right in zip(tokens, tokens[1:]):
            yield f"b:{left}_{right}", 0.75
        for token in tokens:
            padded = f"#{token}#"
            for size in (3, 4, 5):
                for idx in range(max(1, len(padded) - size + 1)):
                    yield f"c:{padded[idx:idx + size]}", 0.25

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature, weight in self._features(text or ""):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            return [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _backend_for_build(backend: str | None) -> str:
    if backend:
        return backend
    if offline_mode():
        return LOCAL_BACKEND
    return f"openai:{embedding_model()}"


def build_embeddings(backend: str) -> Embeddings:
    if backend == LOCAL_BACKEND:
        return HashingEmbeddings()
    if backend.startswith("openai:"):
        if not openai_api_key():
            raise RuntimeError(
                "OPENAI_API_KEY is required for an OpenAI-backed index. "
                "Unset BI_COPILOT_OFFLINE or rebuild with --local."
            )
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=backend.split(":", 1)[1],
            api_key=openai_api_key(),
        )
    raise ValueError(f"Unknown embedding backend: {backend}")


def read_index_meta() -> dict:
    if not EMBEDDING_MARKER.exists():
        return {}
    try:
        return json.loads(EMBEDDING_MARKER.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def get_vectorstore() -> Chroma:
    meta = read_index_meta()
    backend = meta.get("backend")
    if not backend:
        raise FileNotFoundError(
            "Chroma index metadata not found. Run `python -m src.build_rag` first."
        )
    embeddings = build_embeddings(backend)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine", "embedding_backend": backend},
    )


def _load_documents() -> list:
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Docs directory not found: {DOCS_DIR}")
    paths = sorted(
        path
        for path in DOCS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}
    )
    if not paths:
        raise FileNotFoundError(f"No .txt, .md, or .pdf documents found in {DOCS_DIR}")

    documents = []
    for path in paths:
        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            loaded = TextLoader(str(path), encoding="utf-8").load()
        for document in loaded:
            document.metadata["source"] = path.name
            document.metadata["file_type"] = path.suffix.lower().lstrip(".")
        documents.extend(loaded)
    return documents


def _delete_existing_collection() -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # The collection does not exist on the first run.
        pass


def build_index(force_local: bool = False, reset: bool = True) -> dict:
    if force_local:
        backend = LOCAL_BACKEND
    else:
        backend = _backend_for_build(None)

    documents = _load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise RuntimeError("Document splitter produced zero chunks.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if reset:
        _delete_existing_collection()

    embeddings = build_embeddings(backend)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine", "embedding_backend": backend},
    )
    persist = getattr(vectorstore, "persist", None)
    if callable(persist):
        persist()

    meta = {
        "backend": backend,
        "embedding_model": backend.split(":", 1)[1] if ":" in backend else backend,
        "chunks": len(chunks),
        "documents": len(documents),
        "sources": sorted({document.metadata.get("source", "unknown") for document in documents}),
    }
    EMBEDDING_MARKER.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the BI Copilot Chroma index.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force deterministic local hashing embeddings (no API key needed).",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Append to the existing collection instead of rebuilding it.",
    )
    args = parser.parse_args()

    meta = build_index(force_local=args.local, reset=not args.no_reset)
    print("Chroma index built successfully.")
    print(f"  Backend : {meta['backend']}")
    print(f"  Documents: {meta['documents']} | chunks: {meta['chunks']}")
    print(f"  Sources : {', '.join(meta['sources'])}")
    print(f"  Path    : {CHROMA_DIR}")


if __name__ == "__main__":
    main()
