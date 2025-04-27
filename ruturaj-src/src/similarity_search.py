import faiss
import numpy as np
import pandas as pd
import os

def load_faiss_index(index_path):
    """Load a FAISS index from disk"""
    return faiss.read_index(index_path)

def load_metadata(metadata_path):
    """Load metadata from CSV file"""
    metadata_df = pd.read_csv(metadata_path)
    return metadata_df.to_dict('records')

def search_similar(query_embedding, index, metadata, top_k=5):
    """Search for similar vectors in a FAISS index"""
    # Search the index
    distances, indices = index.search(query_embedding, top_k)
    
    # Get the metadata for the search results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            result = metadata[idx].copy()
            result['score'] = float(distances[0][i])
            results.append(result)
    
    return results

def search_context(query_embedding, vector_db_dir='vector_db', top_k=5):
    """Search for relevant context from schema and training data"""
    # Load schema index and metadata
    schema_index = load_faiss_index(os.path.join(vector_db_dir, 'schema_index.faiss'))
    schema_metadata = load_metadata(os.path.join(vector_db_dir, 'schema_metadata.csv'))
    
    # Load training index and metadata
    train_index = load_faiss_index(os.path.join(vector_db_dir, 'train_index.faiss'))
    train_metadata = load_metadata(os.path.join(vector_db_dir, 'train_metadata.csv'))
    
    # Search both indices
    schema_results = search_similar(query_embedding, schema_index, schema_metadata, top_k)
    train_results = search_similar(query_embedding, train_index, train_metadata, top_k)
    
    return {
        'schema_results': schema_results,
        'train_results': train_results
    }
