import pandas as pd
import numpy as np
import time

from main_v1 import process_user_query
from query_processor import load_embedding_model
from sql_generator import clean_sql_query
from db_executor import execute_sql_query

class SQLEvaluator:
    def __init__(self, model, db_path="mimic_iv.sqlite"):
        self.model = model
        self.db_path = db_path
        self.metrics = {
            'total': 0,
            'syntax_errors': 0,
            'execution_errors': 0,
            'exact_match': 0,
            'subset_match': 0,
            'latency': []
        }

    def _compare_results(self, generated, ground_truth):
        try:
            # Exact match
            if generated.equals(ground_truth):
                return 'exact'
            # Subset match
            merged = pd.merge(generated, ground_truth, how='inner')
            if not merged.empty and len(merged) == len(ground_truth):
                return 'subset'
            return 'mismatch'
        except Exception:
            return 'error'

    def _execute_query(self, query):
        result = execute_sql_query(query, self.db_path)
        if result['success']:
            return result['data']
        return None

    def evaluate_row(self, row):
        start_time = time.time()
        # Generate SQL from model
        gen_result = process_user_query(row['question'], self.model)
        generated_sql = clean_sql_query(gen_result['sql_query'])
        ground_truth_sql = clean_sql_query(row['query'])
        # Execute queries
        gen_df = self._execute_query(generated_sql)
        truth_df = self._execute_query(ground_truth_sql)
        # Latency
        self.metrics['latency'].append(time.time() - start_time)
        result = {
            'id': row['id'],
            'question': row['question'],
            'generated_sql': generated_sql,
            'ground_truth_sql': ground_truth_sql,
            'execution_success': gen_df is not None,
            'result_match': 'error'
        }
        if gen_df is None:
            self.metrics['execution_errors'] += 1
        elif truth_df is None:
            result['result_match'] = 'ground_truth_error'
        else:
            match_result = self._compare_results(gen_df, truth_df)
            result['result_match'] = match_result
            if match_result == 'exact':
                self.metrics['exact_match'] += 1
            elif match_result == 'subset':
                self.metrics['subset_match'] += 1
        return result

    def evaluate_dataset(self, test_path):
        test_df = pd.read_csv(test_path)
        # Filter to first 50 entries with both question and query present
        test_df = test_df.dropna(subset=['question', 'query']).head(10)
        results = []
        for _, row in test_df.iterrows():
            self.metrics['total'] += 1
            results.append(self.evaluate_row(row))
        metrics = self.metrics.copy()
        metrics.update({
            'exact_match_accuracy': metrics['exact_match'] / metrics['total'] if metrics['total'] else 0,
            'subset_accuracy': (metrics['exact_match'] + metrics['subset_match']) / metrics['total'] if metrics['total'] else 0,
            'avg_latency': np.mean(metrics['latency']) if metrics['latency'] else 0,
            'error_rate': (metrics['syntax_errors'] + metrics['execution_errors']) / metrics['total'] if metrics['total'] else 0
        })
        return pd.DataFrame(results), metrics

def main():
    model = load_embedding_model()
    evaluator = SQLEvaluator(model)
    results_df, metrics = evaluator.evaluate_dataset('./schema/test.csv')
    results_df.to_csv('evaluation_results.csv', index=False)
    print(f"Evaluation Metrics ({metrics['total']} test cases):")
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
    print(f"Subset Accuracy: {metrics['subset_accuracy']:.2%}")
    print(f"Error Rate: {metrics['error_rate']:.2%}")
    print(f"Average Latency: {metrics['avg_latency']:.2f}s")

if __name__ == "__main__":
    main()
