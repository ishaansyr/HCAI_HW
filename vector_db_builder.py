# build_db.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os, glob
from bs4 import BeautifulSoup
from openai import OpenAI
import chromadb

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")  # or paste temporarily
client = OpenAI(api_key=OPENAI_KEY)

CHROMA_DB_PATH = "./ChromaDB_su_orgs"
COLLECTION_NAME = "SUOrgsCollection"

def semantic_chunking(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    sections = [el.get_text(" ", strip=True) for el in soup.find_all(["p","h1","h2","h3","li"])]
    return [sec for sec in sections if sec]

def build_vectordb():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    # Only build if empty
    if collection.count() > 0:
        print("Vector DB already has data. Skipping.")
        return

    html_files = glob.glob("su_orgs/*.html")
    for filepath in html_files:
        with open(filepath, "r", encoding="utf-8") as f:
            html_text = f.read()

        chunks = semantic_chunking(html_text)
        for i, chunk in enumerate(chunks):
            resp = client.embeddings.create(input=chunk, model="text-embedding-3-small")
            embedding = resp.data[0].embedding
            doc_id = f"{os.path.basename(filepath)}_chunk{i}"
            collection.add(
                documents=[chunk],
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[{"filename": os.path.basename(filepath), "chunk": i}]
            )

    print("✅ Vector DB built and stored in", CHROMA_DB_PATH)

if __name__ == "__main__":
    build_vectordb()
