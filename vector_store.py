import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from faq_data import faq_data

# Load embedding model
embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

faq_questions = [f["question"] for f in faq_data]
faq_embeddings = embed_model.encode(faq_questions, convert_to_numpy=True)

dimension = faq_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(faq_embeddings)

def search_faq(query, topic=None, difficulty=None, k=1):
    """Search FAQ by topic & difficulty and return top-k matches."""
    filtered_faqs = [f for f in faq_data if
                     (topic is None or f["topic"] == topic) and
                     (difficulty is None or f["difficulty"] == difficulty)]
    
    if not filtered_faqs:
        return []

    # Embed filtered questions
    filtered_embeddings = embed_model.encode([f["question"] for f in filtered_faqs], convert_to_numpy=True)
    temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    temp_index.add(filtered_embeddings)

    # Embed user query
    query_vec = embed_model.encode([query], convert_to_numpy=True)

    # Search
    D, I = temp_index.search(query_vec, k)

    # Check if index is valid
    results = []
    for idx in I[0]:
        if idx != -1 and idx < len(filtered_faqs):
            results.append(filtered_faqs[idx])

    return results
