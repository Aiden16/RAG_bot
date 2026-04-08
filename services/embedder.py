import os
import re
from langchain_community.vectorstores import Chroma
from config import VECTOR_STORE_FOLDER

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


def build_collection_name(index_name):
    """
    Chroma collection names should be simple and stable across runs.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", index_name.strip())
    sanitized = sanitized.strip("_-") or "pdf_collection"
    return sanitized[:63]


def embed_and_store(chunks, index_name):
    """
    Takes list of chunks, embeds them locally, and stores them in ChromaDB.
    Returns Chroma persist path and collection name.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    save_path = os.path.join(VECTOR_STORE_FOLDER, index_name)
    os.makedirs(save_path, exist_ok=True)
    collection_name = build_collection_name(index_name)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=save_path,
        collection_name=collection_name,
    )
    vector_store.persist()

    return {
        "persist_directory": save_path,
        "collection_name": collection_name,
    }

