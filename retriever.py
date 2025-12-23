from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

INDEX_DIR = "faiss_index"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_index():
    embeddings = get_embeddings()
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def retrieve(query, k=3):
    db = load_index()
    results = db.similarity_search(query, k=k)
    return results
