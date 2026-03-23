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
CHUNK_SIZE = 1000            # Larger chunks preserve more context per passage
CHUNK_OVERLAP = 200          # More overlap prevents losing content at boundaries
RETRIEVAL_K = 8              # Retrieve more candidates for better coverage
SIMILARITY_THRESHOLD = 0.2   # Lower threshold catches more relevant chunks

# ── Chat History DB ──────────────────────────────────────────────────────────
CHAT_DB_PATH = "./chat_history.db"

# ── Supported File Types ─────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".csv", ".json", ".docx"]

# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise document assistant. Your job is to answer questions \
using ONLY the document excerpts provided below.

Rules:
1. Base your answer ENTIRELY on the provided excerpts. Do not use outside knowledge.
2. Read ALL excerpts carefully before answering — the answer may span multiple chunks.
3. Quote or paraphrase directly from the text when relevant.
4. At the end of your answer, cite sources as: **Source:** [filename, page X] or **Source:** [filename].
5. If the excerpts do not contain enough information, say exactly: \
"The uploaded documents don't contain enough information to answer this. \
Try uploading more relevant documents or rephrasing your question."
6. Never guess or infer beyond what the text explicitly states."""
