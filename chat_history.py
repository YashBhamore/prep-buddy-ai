"""
chat_history.py
===============
SQLite-backed chat history persistence.
Conversations survive app restarts and can be loaded/deleted.
"""

import json
import sqlite3
import uuid
from datetime import datetime

from config import CHAT_DB_PATH


def _get_conn():
    """Get a SQLite connection with WAL mode for better concurrency."""
    conn = sqlite3.connect(CHAT_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()


def create_conversation(title: str = "New Chat") -> str:
    """Create a new conversation and return its ID."""
    conn = _get_conn()
    conv_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    conn.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (conv_id, title, now, now),
    )
    conn.commit()
    conn.close()
    return conv_id


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
):
    """Save a message to a conversation."""
    conn = _get_conn()
    now = datetime.now().isoformat()

    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, sources, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            conversation_id,
            role,
            content,
            json.dumps(sources) if sources else None,
            now,
        ),
    )

    # Update conversation timestamp and title (from first user message)
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )

    # Auto-title from first user message
    if role == "user":
        msg_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND role = 'user'",
            (conversation_id,),
        ).fetchone()[0]

        if msg_count == 1:
            title = content[:50] + ("..." if len(content) > 50 else "")
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id),
            )

    conn.commit()
    conn.close()


def get_conversations() -> list[dict]:
    """List all conversations, most recent first."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations "
        "ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "title": r[1],
            "created_at": r[2],
            "updated_at": r[3],
        }
        for r in rows
    ]


def get_messages(conversation_id: str) -> list[dict]:
    """Get all messages in a conversation."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT role, content, sources, created_at FROM messages "
        "WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,),
    ).fetchall()
    conn.close()

    return [
        {
            "role": r[0],
            "content": r[1],
            "sources": json.loads(r[2]) if r[2] else [],
            "created_at": r[3],
        }
        for r in rows
    ]


def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages."""
    conn = _get_conn()
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        "DELETE FROM conversations WHERE id = ?",
        (conversation_id,),
    )
    # Also delete messages (in case FK cascade doesn't fire)
    conn.execute(
        "DELETE FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    )
    conn.commit()
    conn.close()


def rename_conversation(conversation_id: str, new_title: str):
    """Rename a conversation."""
    conn = _get_conn()
    conn.execute(
        "UPDATE conversations SET title = ? WHERE id = ?",
        (new_title, conversation_id),
    )
    conn.commit()
    conn.close()


# Initialize DB on import
init_db()
