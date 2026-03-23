"""
vector_store.py
===============
ChromaDB vector store operations — ingest, query, manage.
"""

from pathlib import Path

import chromadb
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    SIMILARITY_THRESHOLD,
)
from document_processor import generate_doc_id, load_file, chunk_documents


@st.cache_resource
def get_embedding_model():
    """Load and cache the sentence-transformers embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )


@st.cache_resource
def get_chroma_collection():
    """Initialize persistent ChromaDB collection."""
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_documents(uploaded_files) -> dict:
    """
    Full pipeline: load → chunk → embed → store.
    Returns summary with counts.
    """
    collection = get_chroma_collection()
    embeddings = get_embedding_model()

    total_ingested = 0
    total_skipped = 0
    errors = []
    processed = []

    for uploaded_file in uploaded_files:
        try:
            docs = load_file(uploaded_file)
            chunks = chunk_documents(docs)

            file_ingested = 0
            file_skipped = 0

            for i, chunk in enumerate(chunks):
                doc_id = generate_doc_id(
                    uploaded_file.name, chunk.page_content, i
                )

                # Duplicate check
                existing = collection.get(ids=[doc_id])
                if len(existing["ids"]) > 0:
                    file_skipped += 1
                    continue

                # Embed and store
                vector = embeddings.embed_documents([chunk.page_content])[0]
                collection.upsert(
                    ids=[doc_id],
                    embeddings=[vector],
                    documents=[chunk.page_content],
                    metadatas=[{
                        "source": uploaded_file.name,
                        "chunk_index": i,
                    }],
                )
                file_ingested += 1

            total_ingested += file_ingested
            total_skipped += file_skipped
            processed.append(uploaded_file.name)

        except Exception as e:
            errors.append(f"{uploaded_file.name}: {str(e)}")

    return {
        "ingested": total_ingested,
        "skipped": total_skipped,
        "errors": errors,
        "files": processed,
    }


def query_vector_store(query: str) -> list[dict]:
    """Retrieve top-k relevant chunks, filtered by similarity threshold."""
    collection = get_chroma_collection()
    embeddings = get_embedding_model()

    query_vector = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=RETRIEVAL_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results["ids"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1 - dist
            if score >= SIMILARITY_THRESHOLD:
                chunks.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "score": round(score, 3),
                })

    return chunks


def get_collection_stats() -> dict:
    """Return corpus summary — total chunks and source files."""
    collection = get_chroma_collection()
    count = collection.count()
    sources = set()
    if count > 0:
        results = collection.get(include=["metadatas"])
        for meta in results["metadatas"]:
            sources.add(meta.get("source", "unknown"))
    return {"total_chunks": count, "sources": sorted(sources)}


def delete_source(source_name: str) -> int:
    """Delete all chunks belonging to a specific source file."""
    collection = get_chroma_collection()
    results = collection.get(
        include=["metadatas"],
        where={"source": source_name},
    )
    if results["ids"]:
        collection.delete(ids=results["ids"])
    return len(results["ids"])


def clear_collection():
    """Delete the entire collection and recreate it."""
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    client.delete_collection(COLLECTION_NAME)
    # Clear the cached resource so it recreates
    get_chroma_collection.clear()
