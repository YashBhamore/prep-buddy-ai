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


_INGEST_BATCH_SIZE = 32  # Number of chunks to embed and store at once


def ingest_documents(uploaded_files) -> dict:
    """
    Full pipeline: load → chunk → embed (batched) → store.
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

            # Build all IDs upfront, then batch-check for duplicates
            all_ids = [
                generate_doc_id(uploaded_file.name, chunk.page_content, i)
                for i, chunk in enumerate(chunks)
            ]

            # Batch duplicate check — much faster than one-by-one
            existing_ids: set[str] = set()
            try:
                existing = collection.get(ids=all_ids)
                existing_ids = set(existing["ids"])
            except Exception:
                pass  # If batch get fails, fall through to per-chunk upsert

            new_chunks = [
                (doc_id, chunk, i)
                for i, (doc_id, chunk) in enumerate(zip(all_ids, chunks))
                if doc_id not in existing_ids
            ]
            total_skipped += len(chunks) - len(new_chunks)

            # Batch embed and store in fixed-size batches
            for batch_start in range(0, len(new_chunks), _INGEST_BATCH_SIZE):
                batch = new_chunks[batch_start: batch_start + _INGEST_BATCH_SIZE]
                batch_ids = [item[0] for item in batch]
                batch_texts = [item[1].page_content for item in batch]
                batch_metas = [
                    {
                        "source": uploaded_file.name,
                        "chunk_index": item[2],
                        "page": str(item[1].metadata.get("page", "")),
                    }
                    for item in batch
                ]

                batch_vectors = embeddings.embed_documents(batch_texts)
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_vectors,
                    documents=batch_texts,
                    metadatas=batch_metas,
                )
                total_ingested += len(batch)

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
                source = meta.get("source", "unknown")
                page = meta.get("page", "")
                source_label = f"{source}, p.{page}" if page else source
                chunks.append({
                    "text": doc,
                    "source": source_label,
                    "page": page,
                    "score": round(score, 3),
                })

    # Sort by score descending so Claude sees the most relevant chunks first
    chunks.sort(key=lambda c: c["score"], reverse=True)
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
