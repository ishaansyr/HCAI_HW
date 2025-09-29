# --- Fix for sqlite3 in Streamlit with ChromaDB ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import glob
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI
import chromadb
import google.generativeai as genai
from mistralai import Mistral

st.title("HW5 - Short-Term Memory Chatbot (SU Orgs)")

def get_api_key(vendor: str) -> str | None:
    try:
        return st.secrets[f"{vendor.upper()}_API_KEY"]
    except Exception:
        return None

provider = st.sidebar.selectbox("Select LLM Vendor", ["OpenAI", "Mistral", "Gemini"])
api_key = get_api_key(provider)
if not api_key:
    st.error(f"Missing API key for {provider}. Add {provider.upper()}_API_KEY to secrets.")
    st.stop()

# OpenAI client (used for embeddings)
if "openai_client" not in st.session_state:
    openai_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=openai_key)


CHROMA_DB_PATH = "./ChromaDB_su_orgs"
COLLECTION_NAME = "SUOrgsCollection"

def semantic_chunking(html_text):
    """
    Semantic chunking: keeps logical sections (p, h1, h2, h3, li).
    This prevents cutting across unrelated topics.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    sections = [
        el.get_text(" ", strip=True)
        for el in soup.find_all(["p", "h1", "h2", "h3", "li"])
    ]
    return [sec for sec in sections if sec]

def load_vectordb():
    """Load an already-built ChromaDB collection."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        if collection.count() == 0:
            st.error("Vector DB is empty. Please build it once locally and commit it.")
        else:
            st.success("Vector DB loaded ✅")
        return collection
    except Exception:
        st.error("Vector DB not found. Build it locally before deploying.")
        return None

collection = load_vectordb()
if not collection:
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about student organizations."}]

def trim_history(history, max_pairs=5):
    """Keep only the last `max_pairs` user+assistant turns."""
    user_idxs = [i for i, m in enumerate(history) if m["role"] == "user"]
    if len(user_idxs) <= max_pairs:
        return history
    cutoff = user_idxs[-max_pairs]
    return history[cutoff:]


def get_relevant_club_info(query: str, top_k: int = 5) -> str:
    """Embed query, retrieve relevant chunks from DB, return concatenated text."""
    openai_client = st.session_state.openai_client
    resp = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = resp.data[0].embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    retrieved_docs = results["documents"][0] if results and results["documents"] else []
    return "\n\n".join(retrieved_docs)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about SU organizations...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve context
    context_text = get_relevant_club_info(user_input)

    # Construct prompt
    system_prompt = (
        "You are a chatbot for Syracuse University student organizations. "
        "Use only the provided context from the knowledge base. "
        "Do not hallucinate beyond it. "
        "If the context is empty, say you don’t know."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {user_input}\n\nContext:\n{context_text}"}
    ]

    reply = ""
    if provider == "OpenAI":
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages
        )
        reply = response.choices[0].message.content

    elif provider == "Mistral":
        client = Mistral(api_key=api_key)
        resp = client.chat.complete(
            model="mistral-small-latest",
            messages=messages
        )
        reply = resp.choices[0].message.content

    elif provider == "Gemini":
        genai.configure(api_key=api_key)
        prompt = "".join(f"{m['role'].upper()}: {m['content']}\n" for m in messages)
        gmodel = genai.GenerativeModel("gemini-2.5-flash-lite")
        resp = gmodel.generate_content(prompt)
        reply = getattr(resp, "text", "") or (
            resp.candidates[0].content.parts[0].text if resp.candidates else ""
        )

    with st.chat_message("assistant"):
        st.markdown(f"**Answer from {provider}:**\n\n{reply}")

    # Update history
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.messages = trim_history(st.session_state.messages, max_pairs=5)
