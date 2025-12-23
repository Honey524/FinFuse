# indexer.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pdf_loader import load_and_chunk_pdf
import dotenv, os as _os
dotenv.load_dotenv()

OPENAI_KEY = _os.getenv("OPENAI_API_KEY")
EMB_MODEL = "text-embedding-3-small"   # cost/accuracy tradeoff
INDEX_DIR = "faiss_index"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_or_update_index(pdf_path, index_dir=INDEX_DIR):
    chunks = load_and_chunk_pdf(pdf_path)
    if not chunks:
        return 0
    embeddings = get_embeddings()
    if os.path.exists(index_dir):
        db = FAISS.load_local(index_dir, embeddings,allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_dir)
    return len(chunks)
