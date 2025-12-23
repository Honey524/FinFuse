# pdf_loader.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(path, chunk_size=800, chunk_overlap=100):
    """
    Returns a list of LangChain Document objects (chunked).
    """
    loader = PyPDFLoader(path)
    docs = loader.load()           # page-level docs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    return chunks
