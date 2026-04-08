import math
import os
import re
from collections import Counter

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from config import VECTOR_STORE_FOLDER
from services.embedder import build_collection_name

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def _tokenize(text):
    return re.findall(r"\w+", (text or "").lower())


def _bm25_scores(query, documents, k1=1.5, b=0.75):
    if not documents:
        return []

    tokenized_docs = [_tokenize(doc.page_content) for doc in documents]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0

    doc_freq = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    total_docs = len(documents)
    idf = {
        term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freq.items()
    }

    query_terms = _tokenize(query)
    query_freq = Counter(query_terms)

    scores = []
    for index, tokens in enumerate(tokenized_docs):
        if not tokens:
            scores.append(0.0)
            continue

        term_freq = Counter(tokens)
        doc_len = doc_lengths[index]
        score = 0.0

        for term, qf in query_freq.items():
            if term not in term_freq:
                continue

            tf = term_freq[term]
            term_idf = idf.get(term, 0.0)
            denominator = tf + k1 * (1 - b + b * (doc_len / (avg_doc_len or 1.0)))
            score += qf * term_idf * ((tf * (k1 + 1)) / (denominator or 1.0))

        scores.append(score)

    return scores


def _document_key(document):
    source = document.metadata.get("source", "")
    page = document.metadata.get("page", "")
    preview = document.page_content[:160]
    return f"{source}|{page}|{preview}"


def list_available_indexes():
    if not os.path.isdir(VECTOR_STORE_FOLDER):
        return []

    def is_chroma_store(path):
        return os.path.isfile(os.path.join(path, "chroma.sqlite3"))

    directories = [
        item
        for item in os.listdir(VECTOR_STORE_FOLDER)
        if os.path.isdir(os.path.join(VECTOR_STORE_FOLDER, item))
        and is_chroma_store(os.path.join(VECTOR_STORE_FOLDER, item))
    ]
    directories.sort(
        key=lambda name: os.path.getmtime(os.path.join(VECTOR_STORE_FOLDER, name)),
        reverse=True,
    )
    return directories


def _load_vector_store(index_name):
    persist_directory = os.path.join(VECTOR_STORE_FOLDER, index_name)
    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(f"Vector store not found for index '{index_name}'.")
    if not os.path.isfile(os.path.join(persist_directory, "chroma.sqlite3")):
        raise FileNotFoundError(
            f"Index '{index_name}' is not a ChromaDB store. Re-upload the PDF to migrate it."
        )

    collection_name = build_collection_name(index_name)
    return Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=_embeddings,
    )


def hybrid_retrieve(query, index_name, dense_k=8, sparse_k=8, final_k=4, alpha=0.6):
    vector_store = _load_vector_store(index_name)

    dense_hits = vector_store.similarity_search_with_score(query, k=dense_k)
    all_payload = vector_store.get(include=["documents", "metadatas"])

    all_documents = []
    for text, metadata in zip(
        all_payload.get("documents", []),
        all_payload.get("metadatas", []),
    ):
        all_documents.append(Document(page_content=text, metadata=metadata or {}))

    sparse_scores = _bm25_scores(query, all_documents)
    sparse_pairs = sorted(
        list(zip(all_documents, sparse_scores)),
        key=lambda item: item[1],
        reverse=True,
    )[:sparse_k]

    max_sparse = max((score for _, score in sparse_pairs), default=0.0)

    merged = {}

    for document, distance in dense_hits:
        key = _document_key(document)
        dense_score = 1.0 / (1.0 + max(distance, 0.0))
        if key not in merged:
            merged[key] = {"document": document, "dense": 0.0, "sparse": 0.0}
        merged[key]["dense"] = max(merged[key]["dense"], dense_score)

    for document, score in sparse_pairs:
        key = _document_key(document)
        normalized_sparse = score / max_sparse if max_sparse > 0 else 0.0
        if key not in merged:
            merged[key] = {"document": document, "dense": 0.0, "sparse": 0.0}
        merged[key]["sparse"] = max(merged[key]["sparse"], normalized_sparse)

    ranked = []
    for item in merged.values():
        combined = alpha * item["dense"] + (1.0 - alpha) * item["sparse"]
        ranked.append(
            {
                "content": item["document"].page_content,
                "metadata": item["document"].metadata,
                "dense_score": item["dense"],
                "sparse_score": item["sparse"],
                "score": combined,
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:final_k]
