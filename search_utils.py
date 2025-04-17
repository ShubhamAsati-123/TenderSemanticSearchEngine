# search_utils.py
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Load resources once
df = pd.read_csv("processed_tenders.csv")
model = SentenceTransformer("intfloat/e5-base")
index = faiss.read_index("faiss_index.index")
embeddings = np.load("tender_embeddings.npy")

def search_tenders(query: str, top_k: int = 5):
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_embedding), top_k)
    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = scores[0]
    return results
