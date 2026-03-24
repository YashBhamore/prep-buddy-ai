# Prep Buddy AI

Prep Buddy AI is a document-based learning assistant built with **Anthropic Claude**, **LangGraph**, **ChromaDB**, and **Streamlit**. Upload notes, study material, or reference documents, then ask grounded questions and learn directly from your own content.

## Features

- **Claude-Powered Generation** — Uses Anthropic's Claude API with streaming responses
- **Retrieval-Augmented Generation** — LangGraph orchestrates a retrieve → generate pipeline
- **Multi-Format Ingestion** — PDF, TXT, Markdown, CSV, JSON, DOCX
- **Vector Search** — ChromaDB with cosine similarity and relevance scoring
- **Persistent Chat History** — SQLite-backed conversations that survive restarts
- **Hallucination Guard** — Refuses to answer if no relevant context is found
- **Duplicate Detection** — Content-addressed hashing prevents redundant indexing
- **Source Citations** — Every answer includes scored source references

## Working Demo 
**https://drive.google.com/file/d/1J4373pYnotAGapEz_OmxasrbQiGVGcEy/view?usp=sharing**

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                    │
│  ┌──────────┐  ┌────────────┐  ┌─────────────┐  │
│  │ Upload   │  │   Chat     │  │  History    │  │
│  │ Panel    │  │  Interface │  │  Panel      │  │
│  └────┬─────┘  └─────┬──────┘  └──────┬──────┘  │
└───────┼──────────────┼─────────────────┼─────────┘
        │              │                 │
        ▼              ▼                 ▼
   ┌─────────┐   ┌──────────┐     ┌──────────┐
   │ Document │   │ LangGraph│     │  SQLite  │
   │Processor │   │  Agent   │     │ Chat DB  │
   └────┬─────┘   └────┬─────┘     └──────────┘
        │              │
        ▼              ▼
   ┌──────────────────────┐     ┌──────────────┐
   │      ChromaDB        │◄────│  Claude API  │
   │   (Vector Store)     │     │  (Anthropic) │
   └──────────────────────┘     └──────────────┘
```

## Tech Stack

| Layer      | Technology                        |
|------------|-----------------------------------|
| LLM        | Anthropic Claude (Sonnet 4)       |
| Orchestration | LangGraph (state machine)      |
| Embeddings | all-MiniLM-L6-v2 (local, no API)  |
| Vector DB  | ChromaDB (persistent, cosine)     |
| UI         | Streamlit                         |
| Chat Store | SQLite                            |
| Doc Loaders| LangChain, python-docx, pypdf     |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Anthropic API key ([get one here](https://console.anthropic.com/))

### Setup

```bash
# Clone
git clone https://github.com/YashBhamore/prep-buddy-ai.git
cd prep-buddy-ai

# Create .env
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Install dependencies (with uv)
uv sync

# Run
uv run streamlit run app.py
```

**Or with pip:**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # or: pip install .
streamlit run app.py
```

### Usage

1. Open `http://localhost:8501` in your browser
2. Upload documents via the sidebar (PDF, TXT, MD, CSV, JSON, DOCX)
3. Click **Ingest** to index them into the vector store
4. Ask questions in the chat — answers are grounded in your documents
5. Check source citations to verify accuracy

## Project Structure

```
prep-buddy-ai/
├── app.py                  # Streamlit UI — header, sidebar, chat
├── agent.py                # LangGraph agent — retrieve + generate nodes
├── config.py               # All configuration and constants
├── document_processor.py   # Multi-format document loading + chunking
├── vector_store.py         # ChromaDB operations — ingest, query, manage
├── chat_history.py         # SQLite persistence for conversations
├── pyproject.toml          # Dependencies
├── .env.example            # Environment variable template
└── .gitignore
```

## Configuration

All tuneable parameters are in `config.py`:

| Parameter             | Default | Description                            |
|-----------------------|---------|----------------------------------------|
| `CLAUDE_MODEL`        | `claude-sonnet-4-20250514` | Anthropic model       |
| `CHUNK_SIZE`          | 512     | Characters per document chunk          |
| `CHUNK_OVERLAP`       | 50      | Overlap between consecutive chunks     |
| `RETRIEVAL_K`         | 5       | Number of chunks to retrieve           |
| `SIMILARITY_THRESHOLD`| 0.3     | Minimum cosine similarity to keep      |

## How It Works

1. **Ingestion**: Documents are loaded → split into chunks → embedded with MiniLM → stored in ChromaDB with content-addressed IDs
2. **Retrieval**: User query is embedded → cosine similarity search → chunks below threshold are filtered out
3. **Generation**: Retrieved context + chat history → Claude API with streaming → response with source citations
4. **Persistence**: Chat messages are saved to SQLite → conversations can be loaded/deleted across sessions

## Contributors

**Yash Bhamore**
- GitHub: [YashBhamore](https://github.com/YashBhamore)
- LinkedIn: [yash-bhamore](https://linkedin.com/in/yash-bhamore)

**Darsh Joshi**
- Contributor: Prompt Engineering and QA

## Role Ownership

| Role | Contributor | Ownership |
|------|-------------|-----------|
| Corpus Architect | Yash Bhamore | Document ingestion and corpus handling in `document_processor.py` |
| Pipeline Engineer | Yash Bhamore | Retrieval pipeline, vector store, orchestration, and configuration in `agent.py`, `vector_store.py`, and `config.py` |
| UX Lead | Yash Bhamore | Streamlit experience and app flow in `app.py` |
| Prompt Engineer | Darsh Patel | Prompt design, instruction tuning, and response behavior |
| QA Lead | Darsh Patel | Validation, testing support, and demo review |
