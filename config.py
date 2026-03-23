"""
config.py
=========
Centralized configuration for Prep Buddy AI.
All tuneable parameters live here — no magic numbers in other files.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Anthropic Claude API ─────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# ── Embedding Model ──────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 is ~90MB, downloads once, runs locally on CPU.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "prep_buddy_ai"

# ── Retrieval Tuning ─────────────────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
RETRIEVAL_K = 5
SIMILARITY_THRESHOLD = 0.3  # chunks below this score are discarded

# ── Chat History DB ──────────────────────────────────────────────────────────
CHAT_DB_PATH = "./chat_history.db"

# ── Supported File Types ─────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".csv", ".json", ".docx"]

# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable AI assistant that answers questions \
strictly from uploaded documents.

Rules:
1. Answer ONLY from the provided context. Never use general knowledge.
2. If the context is insufficient, say: "I couldn't find relevant information \
in the uploaded documents. Try rephrasing or uploading more documents."
3. Cite your sources at the end: **Source:** [filename]
4. Be concise, accurate, and well-structured.
5. If multiple sources are relevant, synthesize them and cite all."""
