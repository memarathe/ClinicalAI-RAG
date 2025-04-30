import os
import argparse
import pandas as pd
import json
import sqlite3
from tqdm import tqdm
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

def compare_results(gold_df, pred_df):
    """Compare the results of gold and predicted SQL queries"""
    if gold_df is None or pred_df is None:
        return False, "One or both query executions failed"
    
    # Exact match
    if gold_df.equals(pred_df):
        return True, "Exact match"
    
    # Sort both dataframes if they have the same columns
    if set(gold_df.columns) == set(pred_df.columns):
        gold_sorted = gold_df.sort_values(by=list(gold_df.columns)).reset_index(drop=True)
        pred_sorted = pred_df.sort_values(by=list(pred_df.columns)).reset_index(drop=True)
        if gold_sorted.equals(pred_sorted):
            return True, "Match after sorting"
    
    # Check if pred is subset of gold
    if len(gold_df) >= len(pred_df):
        try:
            gold_set = set(map(tuple, gold_df.values))
            pred_set = set(map(tuple, pred_df.values))
            if pred_set.issubset(gold_set):
                return True, "Subset match"
        except:
            pass
    
    return False, "Results do not match"

def execute_test_sql(sql, db_path):
    """Execute SQL query and return results as DataFrame"""
    try:
        conn = sqlite3.connect(db_path)
        result = pd.read_sql_query(sql, conn)
        return result, None
    except Exception as e:
        return None, str(e)
    finally:
        if 'conn' in locals():
            conn.close()

def evaluate_model(test_csv_path, model, db_path, vector_db_dir='vector_db', output_dir='evaluation_results'):
    """Evaluate model performance on test set"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Read test data
    test_df = pd.read_csv(test_csv_path)
    
    results = []
    correct = 0
    syntactic_correct = 0
    total = len(test_df.head(10))
    
    for idx, row in tqdm(test_df.iterrows(), total=total, desc="Evaluating"):
        question = row['question']
        gold_sql = clean_sql(row['query'])
        
        # Generate SQL from model
        try:
            # Use the existing pipeline to generate SQL
            query_embedding = vectorize_user_query(question, model)
            search_results = search_context(query_embedding, vector_db_dir)
            formatted_context = format_context(search_results)
            pred_sql = generate_sql_query(question, formatted_context)
            pred_sql = clean_sql(pred_sql)
            
            # Check if model abstained
            if pred_sql.startswith("ABSTAIN:"):
                result = {
                    'id': row.get('id', idx),
                    'question': question,
                    'gold_sql': gold_sql,
                    'pred_sql': pred_sql,
                    'abstained': True,
                    'is_correct': False,
                    'syntactically_valid': False,
                    'error': "Model abstained"
                }
                results.append(result)
                continue
        except Exception as e:
            result = {
                'id': row.get('id', idx),
                'question': question,
                'gold_sql': gold_sql,
                'pred_sql': None,
                'abstained': False,
                'is_correct': False,
                'syntactically_valid': False,
                'error': f"Generation error: {str(e)}"
            }
            results.append(result)
            continue
        
        # Execute gold SQL
        gold_result, gold_error = execute_test_sql(gold_sql, db_path)
        
        # Execute predicted SQL
        pred_result, pred_error = execute_test_sql(pred_sql, db_path)
        
        # Check if query is syntactically valid
        is_syntactically_valid = pred_error is None
        if is_syntactically_valid:
            syntactic_correct += 1
        
        # Compare results
        is_correct = False
        comparison_note = ""
        
        if gold_result is not None and pred_result is not None:
            is_correct, comparison_note = compare_results(gold_result, pred_result)
            if is_correct:
                correct += 1
        
        # Store result
        result = {
            'id': row.get('id', idx),
            'question': question,
            'gold_sql': gold_sql,
            'pred_sql': pred_sql,
            'abstained': False,
            'is_correct': is_correct,
            'syntactically_valid': is_syntactically_valid,
            'gold_error': gold_error,
            'pred_error': pred_error,
            'comparison_note': comparison_note
        }
        results.append(result)
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    syntactic_accuracy = syntactic_correct / total if total > 0 else 0
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    
    # Generate summary
    summary = {
        'total_examples': total,
        'correct': correct,
        'accuracy': accuracy,
        'syntactically_valid': syntactic_correct,
        'syntactic_accuracy': syntactic_accuracy,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"Evaluation complete: {correct}/{total} correct ({accuracy:.2%})")
    print(f"Syntactically valid: {syntactic_correct}/{total} ({syntactic_accuracy:.2%})")
    
    return summary, results

def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description='RAG-based Text-to-SQL System')
    parser.add_argument('--setup', action='store_true', help='Set up vector database')
    parser.add_argument('--schema', default='./schema/medical_schema.sql', help='Path to SQL schema file')
    parser.add_argument('--train', default='./schema/train.csv', help='Path to training data CSV file')
    parser.add_argument('--db', default='./schema/mimic_iv.sqlite', help='Path to SQLite database')
    parser.add_argument('--query', help='Run a single query and exit')
    parser.add_argument('--evaluate', help='Path to test CSV file for evaluation')
    parser.add_argument('--output-dir', default='./output', help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Set up vectors if requested
    if args.setup:
        setup_vectors(args.schema, args.train)
    
    # Load embedding model
    model = load_embedding_model()
    
    # Run evaluation if requested
    if args.evaluate:
        print(f"Evaluating model on {args.evaluate}...")
        summary, _ = evaluate_model(args.evaluate, model, args.db, output_dir=args.output_dir)
        print("Evaluation summary:")
        print(f"Accuracy: {summary['accuracy']:.2%}")
        print(f"Syntactic accuracy: {summary['syntactic_accuracy']:.2%}")
        return
    
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
