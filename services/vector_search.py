from services.hybrid_retriever import hybrid_retrieve


def search(query, index_name, top_k=5):
    """
    Hybrid search wrapper.
    Returns chunk texts only for compatibility with existing callers.
    """
    ranked_results = hybrid_retrieve(
        query=query,
        index_name=index_name,
        final_k=top_k,
    )
    return [item["content"] for item in ranked_results]
