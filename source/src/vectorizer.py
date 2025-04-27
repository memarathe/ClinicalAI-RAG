import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

def read_schema_excel(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df = df.reset_index(drop=True)
    return df

def preprocess_text(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records = []
    for _, row in df.iterrows():
        column_name = str(row.get('ColumnName', '')).strip().lower()
        friendly_name = str(row.get('User-Friendly Name', '')).strip()
        labels = str(row.get('IUIE Labels', '')).strip()
        description = row.get('Description & Valid Values', '')
        description = "" if pd.isna(description) else str(description).strip()
        
        # Create text representation for embedding
        if description:
            text_representation = f"""
            Column: {column_name}
            Friendly Name: {friendly_name}
            Labels: {labels}
            Description: {description}
            """
        else:
            text_representation = f"""
            Column: {column_name}
            Friendly Name: {friendly_name}
            Labels: {labels}
            """
        
        record = {
            'column_name': column_name,
            'friendly_name': friendly_name,
            'labels': labels,
            'description': description,
            'text_for_embedding': text_representation.strip()
        }
        records.append(record)
    return records

def generate_embeddings(records, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    texts = [record['text_for_embedding'] for record in records]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def build_faiss_index(embeddings, records, output_dir='vector_db'):
    os.makedirs(output_dir, exist_ok=True)
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(output_dir, "schema_index.faiss"))
    metadata = pd.DataFrame(records)
    metadata.to_csv(os.path.join(output_dir, "schema_metadata.csv"), index=False)
    print(f"FAISS index and metadata saved to {output_dir}")
    print(f"Index contains {index.ntotal} vectors of dimension {dimension}")

def vectorize_schema(excel_file, output_dir='vector_db', model_name='all-MiniLM-L6-v2'):
    print("Reading schema from Excel...")
    schema_df = read_schema_excel(excel_file)
    if schema_df.empty:
        print("Failed to read schema data. Exiting.")
        return
    
    print("Preprocessing schema text...")
    processed_records = preprocess_text(schema_df)
    
    print("Generating embeddings...")
    embeddings = generate_embeddings(processed_records, model_name)
    
    print("Building FAISS index...")
    build_faiss_index(embeddings, processed_records, output_dir)
    print("Schema vectorization complete!")
