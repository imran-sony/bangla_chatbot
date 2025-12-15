import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from faq_data import faq_data

embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

documents = [faq["question"] + " " + faq["answer"] for faq in faq_data]
embeddings = embedding_model.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def search_faq(query, topic, difficulty=None, k=3):
    query_vector = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)

    results = []
    for idx in indices[0]:
        faq = faq_data[idx]
        if faq["topic"] == topic:
            if difficulty is None or faq["difficulty"] == difficulty:
                results.append(faq)

    return results
