# app/config.py

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "./uploads")
VECTOR_STORE_FOLDER = os.path.join(BASE_DIR, "./vector_store")

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

# Model config
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Hybrid retrieval config
DENSE_TOP_K = 8
SPARSE_TOP_K = 8
FINAL_TOP_K = 4
HYBRID_ALPHA = 0.6
