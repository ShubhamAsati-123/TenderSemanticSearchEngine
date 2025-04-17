import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load data
df = pd.read_csv("inference_data.csv")
df = df.fillna("")  # Fill NaNs to avoid issues

# Combine relevant columns to form text
df['text'] = df['summary'].astype(str) + ". " + df['details'].astype(str)

# Load embedding model
model = SentenceTransformer("intfloat/e5-base")
embeddings = model.encode(df['text'].tolist(), normalize_embeddings=True, show_progress_bar=True)

# Save intermediate files
np.save("tender_embeddings.npy", embeddings)
df.to_csv("processed_tenders.csv", index=False)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine-like for normalized)
index.add(np.array(embeddings))
faiss.write_index(index, "faiss_index.index")
