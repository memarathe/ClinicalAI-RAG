from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Load SentenceTransformer model"""
    return SentenceTransformer(model_name)

def vectorize_user_query(query_text, model):
    """Generate embedding for user's natural language query"""
    # Generate embedding
    query_embedding = model.encode([query_text])
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    return query_embedding
