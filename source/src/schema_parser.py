
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os

def parse_schema_sql(sql_path):
    """Parse SQL schema file into structured records for embedding with improved regex"""
    with open(sql_path, 'r') as f:
        sql_content = f.read()
    
    # Extract table definitions with improved regex
    # Case-insensitive matching, better handling of whitespace and quoted identifiers
    tables = re.findall(r'CREATE\s+TABLE\s+[`"\[]?(\w+)[`"\]]?\s*\((.*?)\)\s*;', sql_content, re.DOTALL | re.IGNORECASE)
    
    records = []
    for table_name, columns_block in tables:
        # Split columns by line
        lines = [line.strip() for line in columns_block.split('\n') if line.strip()]
        
        for line in lines:
            # Skip comments-only lines
            if line.strip().startswith('--'):
                continue
            
            # Extract column definition and comment with better handling
            parts = line.split('--', 1)
            column_def = parts[0].strip().rstrip(',')
            comment = parts[1].strip() if len(parts) > 1 else ""
            
            # Improved regex for column name and type
            col_match = re.match(r'[`"\[]?(\w+)[`"\]]?\s+([\w\(\)\s]+)', column_def)
            if not col_match:
                continue
                
            column_name = col_match.group(1)
            column_type = col_match.group(2).strip()
            
            # Create text representation for embedding
            text_representation = f"""
            Table: {table_name}
            Column: {column_name}
            Type: {column_type}
            Description: {comment}
            """
            
            record = {
                'table_name': table_name,
                'column_name': column_name,
                'column_type': column_type,
                'description': comment,
                'text_for_embedding': text_representation.strip()
            }
            records.append(record)
    
    # Extract join relationships with improved regex
    joins = re.findall(r'--\s*([\w\.\s]+?)\s+can\s+be\s+joined\s+with\s+([\w\.\s]+)', sql_content, re.IGNORECASE)
    
    for left, right in joins:
        text_representation = f"""
        Relationship: {left.strip()} can be joined with {right.strip()}
        """
        
        record = {
            'relationship': f"{left.strip()} can be joined with {right.strip()}",
            'text_for_embedding': text_representation.strip()
        }
        records.append(record)
    
    if records:
        print(f"DEBUG: First table name found: {records[0].get('table_name', 'No tables found')}")
    
    return records

def vectorize_schema_from_sql(sql_path, output_dir='vector_db', model_name='all-MiniLM-L6-v2'):
    """Vectorize schema from SQL file and save to FAISS index"""
    records = parse_schema_sql(sql_path)
    
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
    faiss.write_index(index, os.path.join(output_dir, "schema_index.faiss"))
    metadata = pd.DataFrame(records)
    metadata.to_csv(os.path.join(output_dir, "schema_metadata.csv"), index=False)
    
    # Print detailed debugging information
    table_names = set([r['table_name'] for r in records if 'table_name' in r])
    print(f"Extracted table names: {', '.join(sorted(table_names))}")
    print(f"Schema vectorization complete: {len(records)} elements vectorized")
    
    # Count columns per table for verification
    table_column_counts = {}
    for r in records:
        if 'table_name' in r and 'column_name' in r:
            table_name = r['table_name']
            if table_name not in table_column_counts:
                table_column_counts[table_name] = 0
            table_column_counts[table_name] += 1
    
    print("Column counts per table:")
    for table, count in sorted(table_column_counts.items()):
        print(f"  {table}: {count} columns")

    return records, embeddings
