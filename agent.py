"""
agent.py
========
LangGraph RAG agent powered by Anthropic Claude.
Supports streaming responses for real-time UI updates.

Graph: [START] → retrieve → generate → [END]
"""

from typing import Annotated, Generator

import anthropic
import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, SYSTEM_PROMPT
from vector_store import query_vector_store


# ── Agent State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_chunks: list[dict]
    no_context_found: bool


# ── Anthropic Client ─────────────────────────────────────────────────────────

@st.cache_resource
def get_anthropic_client():
    """Initialize and cache the Anthropic client."""
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ── Graph Nodes ──────────────────────────────────────────────────────────────

def retrieve_node(state: AgentState) -> dict:
    """Node 1 — Retrieve relevant chunks from ChromaDB."""
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)

    chunks = query_vector_store(query)

    return {
        "retrieved_chunks": chunks,
        "no_context_found": len(chunks) == 0,
    }


def generate_node(state: AgentState) -> dict:
    """
    Node 2 — Generate response using Claude API.
    Non-streaming version used by LangGraph pipeline.
    """
    if state["no_context_found"]:
        return {
            "messages": [AIMessage(
                content="I couldn't find relevant information in the uploaded "
                        "documents. Try rephrasing your question or upload "
                        "relevant documents first."
            )]
        }

    client = get_anthropic_client()

    # Build context from retrieved chunks
    context_parts = []
    for chunk in state["retrieved_chunks"]:
        context_parts.append(
            f"[Source: {chunk['source']} | Relevance: {chunk['score']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Build message history for Claude
    system = f"{SYSTEM_PROMPT}\n\nRelevant document excerpts:\n\n{context}"

    messages = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=system,
        messages=messages,
    )

    return {"messages": [AIMessage(content=response.content[0].text)]}


# ── Build Graph ──────────────────────────────────────────────────────────────

def build_agent():
    """Assemble the LangGraph state graph."""
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


@st.cache_resource
def get_agent():
    """Cache the compiled agent."""
    return build_agent()


# ── Streaming Chat ───────────────────────────────────────────────────────────

def chat_stream(query: str, history: list[dict]) -> Generator[str, None, None]:
    """
    Stream a response from Claude using retrieved context.
    Yields text chunks for real-time UI rendering.

    Returns a generator — call retrieve separately to get sources.
    """
    # Step 1: Retrieve
    chunks = query_vector_store(query)

    # Store sources in session state for the UI to pick up
    st.session_state._last_sources = chunks

    if not chunks:
        yield ("I couldn't find relevant information in the uploaded "
               "documents. Try rephrasing your question or upload "
               "relevant documents first.")
        return

    # Step 2: Build context
    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[Source: {chunk['source']} | Relevance: {chunk['score']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system = f"{SYSTEM_PROMPT}\n\nRelevant document excerpts:\n\n{context}"

    # Step 3: Build messages
    messages = []
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"],
        })
    messages.append({"role": "user", "content": query})

    # Step 4: Stream from Claude
    client = get_anthropic_client()
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def chat(query: str, history: list[dict]) -> tuple[str, list[dict]]:
    """
    Non-streaming chat — returns full response + sources.
    Used as fallback if streaming isn't desired.
    """
    agent = get_agent()

    lc_messages = []
    for msg in history:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))
    lc_messages.append(HumanMessage(content=query))

    result = agent.invoke({
        "messages": lc_messages,
        "retrieved_chunks": [],
        "no_context_found": False,
    })

    response_text = result["messages"][-1].content
    sources = result.get("retrieved_chunks", [])

    return response_text, sources
