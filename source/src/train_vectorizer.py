import pandas as pd
import numpy as np
import faiss
import os
import json
from sentence_transformers import SentenceTransformer

def process_train_csv(csv_path):
    """Process training data from CSV file with question-query pairs"""
    df = pd.read_csv(csv_path)
    df = df[:4765][:]  # Limit to 4765 rows for testing;
    records = []
    for _, row in df.iterrows():
        question = str(row['question'])
        query = str(row['query'])
        
        # Handle template and val_dict if available
        template = str(row.get('template', ''))
        val_dict_str = str(row.get('val_dict', '{}'))
        
        # Try to parse val_dict if it's a string representation of a dict
        try:
            val_dict = json.loads(val_dict_str.replace("'", '"'))
        except:
            val_dict = {}
        
        # Create text representation for embedding
        text_representation = f"""
        Question: {question}
        SQL Query: {query}
        """
        
        record = {
            'id': str(row.get('id', '')),
            'question': question,
            'query': query,
            'template': template,
            'val_dict': val_dict_str,
            'text_for_embedding': text_representation.strip()
        }
        
        records.append(record)
    
    return records

def vectorize_training_data(csv_path, output_dir='vector_db', model_name='all-MiniLM-L6-v2'):
    """Vectorize training data and save to FAISS index"""
    # Process training data
    records = process_train_csv(csv_path)
    
    # Generate embeddings
    model = SentenceTransformer(model_name)
    texts = [r['text_for_embedding'] for r in records]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    os.makedirs(output_dir, exist_ok=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save index and metadata
    faiss.write_index(index, os.path.join(output_dir, "train_index.faiss"))
    metadata = pd.DataFrame(records)
    metadata.to_csv(os.path.join(output_dir, "train_metadata.csv"), index=False)
    
    print(f"Training data vectorization complete: {len(records)} examples vectorized")
    return records, embeddings
