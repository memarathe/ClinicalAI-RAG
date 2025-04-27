import os
import argparse
from schema_parser import vectorize_schema_from_sql
from train_vectorizer import vectorize_training_data
from query_processor import load_embedding_model, vectorize_user_query
from similarity_search import search_context
from sql_generator import format_context, generate_sql_query
from db_executor import execute_sql_query, format_results

def setup_vectors(schema_path, train_path, vector_db_dir='vector_db', force_rebuild=False):
    """Set up the vector database by vectorizing schema and training data"""
    os.makedirs(vector_db_dir, exist_ok=True)
    
    # Check if indices already exist
    schema_index_exists = os.path.exists(os.path.join(vector_db_dir, 'schema_index.faiss'))
    train_index_exists = os.path.exists(os.path.join(vector_db_dir, 'train_index.faiss'))
    
    # Vectorize schema if needed
    if not schema_index_exists or force_rebuild:
        print("Vectorizing schema from SQL file...")
        vectorize_schema_from_sql(schema_path, vector_db_dir)
    else:
        print("Schema index already exists, skipping vectorization.")
    
    # Vectorize training data if needed
    if not train_index_exists or force_rebuild:
        print("Vectorizing training data...")
        vectorize_training_data(train_path, vector_db_dir)
    else:
        print("Training index already exists, skipping vectorization.")

def clean_sql(sql):
    """Remove markdown and formatting from SQL response"""
    import re
    # Remove opening code fence and optional 'sql' language tag
    sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
    # Remove opening code fence without language tag
    sql = re.sub(r"^```\s*", "", sql)
    # Remove closing code fence at the end
    sql = re.sub(r"\s*```\s*$", "", sql)
    # Remove any leading/trailing backticks or whitespace
    sql = sql.strip("` \n")
    # Remove any leading 'sql ' or 'SQL '
    sql = re.sub(r"^\s*sql\s+", "", sql, flags=re.IGNORECASE)
    # Remove inline SQL comments
    sql = re.sub(r"--.*", "", sql)
    # Collapse multiple spaces and remove trailing semicolons
    sql = re.sub(r"\s+", " ", sql).strip().rstrip(';')
    return sql


def process_user_query(query_text, model, vector_db_dir='vector_db', db_path='mimic_iv.sqlite', api_key=None):
    """Process a user query through the entire pipeline"""
    # 1. Vectorize user query
    query_embedding = vectorize_user_query(query_text, model)
    # 2. Perform similarity search
    search_results = search_context(query_embedding, vector_db_dir)
    # 3. Format context for language model
    formatted_context = format_context(search_results)
    # 4. Generate SQL query
    sql_query = generate_sql_query(query_text, formatted_context)
    #5. Clean SQL query
    sql_query = clean_sql(sql_query)
    # 6. Execute SQL query
    execution_results = execute_sql_query(sql_query, db_path)
    # 7. Format results
    results = format_results(execution_results)
    
    return {
        "user_query": query_text,
        "sql_query": sql_query,
        "execution_success": execution_results["success"],
        "results": results
    }

def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description='RAG-based Text-to-SQL System')
    parser.add_argument('--setup', action='store_true', help='Set up vector database')
    parser.add_argument('--schema', default='./schema/schema.sql', help='Path to SQL schema file')
    parser.add_argument('--train', default='./schema/train.csv', help='Path to training data CSV file')
    parser.add_argument('--db', default='./schema/mimic_iv.sqlite', help='Path to SQLite database')
    parser.add_argument('--query', help='Run a single query and exit')
    
    args = parser.parse_args()
    
    # Set up vectors if requested
    if args.setup:
        setup_vectors(args.schema, args.train)
    
    # Load embedding model
    model = load_embedding_model()
    
    # Process a single query if provided
    if args.query:
        results = process_user_query(args.query, model, db_path=args.db)
        print(f"SQL query: {results['sql_query']}")
        print("Results:")
        print(results['results'])
        return
    
    # Interactive mode
    print("RAG-based Text-to-SQL System")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break
        
        results = process_user_query(query, model, db_path=args.db)
        print(f"\nGenerated SQL: {results['sql_query']}")
        print("\nResults:")
        print(results['results'])

if __name__ == "__main__":
    main()
