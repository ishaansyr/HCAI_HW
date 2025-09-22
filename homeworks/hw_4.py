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

st.title("HW 4 - iSchool RAG Chatbot")

provider = st.sidebar.selectbox("Select LLM Vendor", ["OpenAI", "Mistral", "Gemini"])


if "openai_client" not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=api_key)

def get_api_key(vendor: str) -> str | None:
    try:
        return st.secrets[f"{vendor.upper()}_API_KEY"]
    except Exception:
        return None

api_key = get_api_key(provider)
if not api_key:
    st.error(f"Missing API key for {provider}. Add {provider.upper()}_API_KEY to secrets.")
    st.stop()

CHROMA_DB_PATH = "./ChromaDB_su_orgs"
COLLECTION_NAME = "SUOrgsCollection"

def semantic_chunking(html_text):
    """
    I have used semantic chunking as there's a wide range of topics in this database (journalism, business, fraternities etc)
    Fixed-size would have cut off chunks abruptly, which would not be ideal for this exercise.
    If it was possible for each html page to be a chunk, I would have preferred that instead.
    """
    soup = BeautifulSoup(html_text, "html.parser")

    # Collect text from semantic elements
    sections = [
        el.get_text(" ", strip=True)
        for el in soup.find_all(["p", "h1", "h2", "h3", "li"])
    ]

    # Keep only non-empty chunks
    chunks = [sec for sec in sections if sec]

    return chunks


def build_vectordb():
    if os.path.exists(CHROMA_DB_PATH):
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        return chroma_client.get_or_create_collection(COLLECTION_NAME)

    st.write("⚡ Building vector DB from su_orgs HTML files…")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    openai_client = st.session_state.openai_client
    html_files = glob.glob("su_orgs/*.html")
    for filepath in html_files:
        with open(filepath, "r", encoding="utf-8") as f:
            html_text = f.read()
        chunks = semantic_chunking(html_text)
        for i, chunk in enumerate(chunks):
            resp = openai_client.embeddings.create(
                input=chunk,
                model="text-embedding-3-small"
            )
            embedding = resp.data[0].embedding
            doc_id = f"{os.path.basename(filepath)}_chunk{i}"
            collection.add(
                documents=[chunk],
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[{"filename": os.path.basename(filepath), "chunk": i}]
            )
    st.success("Vector DB built ✅")
    return collection

collection = build_vectordb()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about student organizations."}]

def trim_history(history, max_pairs=5):
    user_idxs = [i for i, m in enumerate(history) if m["role"] == "user"]
    if len(user_idxs) <= max_pairs:
        return history
    # Trim older messages
    cutoff = user_idxs[-max_pairs]
    return history[cutoff:]

# ------------------ Chat Interface ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about SU organizations...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve context from vector DB
    openai_client = st.session_state.openai_client
    resp = openai_client.embeddings.create(
        input=user_input,
        model="text-embedding-3-small"
    )
    query_embedding = resp.data[0].embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=2)
    retrieved_docs = results["documents"][0] if results and results["documents"] else []
    context_text = "\n\n".join(retrieved_docs)

    # Prompt construction
    system_prompt = (
        "You are a chatbot for Syracuse University student organizations. "
        "If relevant context is provided, cite it clearly and explain you are using knowledge base (RAG). "
        "If not, answer from your own reasoning. Clearly state if your response is from your own reasoning or the knowledge base"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {user_input}\n\nContext from knowledge base:\n{context_text}"}
    ]

    # ------------------ Call LLM ------------------
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

    # Save reply and trim history
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.messages = trim_history(st.session_state.messages, max_pairs=5)
