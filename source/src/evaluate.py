import pandas as pd
import time
import logging
import sys
from tqdm import tqdm
from main_v1 import process_user_query, clean_sql, evaluate_sql_match
from query_processor import load_embedding_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class StructuralSQLEvaluator:
    def __init__(self, model):
        self.model = model
        self.metrics = {
            'total': 0,
            'perfect_structural': 0,
            'avg_select_f1': 0,
            'avg_from_f1': 0,
            'avg_where_f1': 0,
            'latency': []
        }

    def evaluate_row(self, row):
        start_time = time.time()
        try:
            gen_result = process_user_query(row['question'], self.model)
            generated_sql = clean_sql(gen_result['sql_query'])
            ground_truth_sql = clean_sql(row['query'])
            scores = evaluate_sql_match(generated_sql, ground_truth_sql)
            elapsed = time.time() - start_time
            self.metrics['latency'].append(elapsed)
            perfect = all(scores[k]['f1'] == 1 for k in scores)
            if perfect:
                self.metrics['perfect_structural'] += 1
            self.metrics['avg_select_f1'] += scores['select']['f1']
            self.metrics['avg_from_f1'] += scores['from']['f1']
            self.metrics['avg_where_f1'] += scores['where']['f1']
            return {
                'id': row.get('id', 'unknown'),
                'question': row['question'],
                'generated_sql': generated_sql,
                'ground_truth_sql': ground_truth_sql,
                'select_f1': scores['select']['f1'],
                'from_f1': scores['from']['f1'],
                'where_f1': scores['where']['f1'],
                'structural_match': perfect,
                'latency': elapsed
            }
        except Exception as e:
            logger.error(f"Error evaluating row: {e}")
            return {
                'id': row.get('id', 'unknown'),
                'question': row['question'],
                'generated_sql': "ERROR",
                'ground_truth_sql': row.get('query', ""),
                'select_f1': 0,
                'from_f1': 0,
                'where_f1': 0,
                'structural_match': False,
                'latency': 0,
                'error': str(e)
            }

    def evaluate_dataset(self, test_path):
        test_df = pd.read_csv(test_path)
        test_df = test_df.dropna(subset=['question', 'query']).head(10)
        results = []
        for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating (structural)"):
            self.metrics['total'] += 1
            results.append(self.evaluate_row(row))
        n = self.metrics['total']
        self.metrics['avg_select_f1'] /= n
        self.metrics['avg_from_f1'] /= n
        self.metrics['avg_where_f1'] /= n
        self.metrics['avg_latency'] = sum(self.metrics['latency']) / n if n else 0
        self.metrics['structural_accuracy'] = self.metrics['perfect_structural'] / n if n else 0
        return pd.DataFrame(results), self.metrics

def main():
    model = load_embedding_model()
    evaluator = StructuralSQLEvaluator(model)
    try:
        results_df, metrics = evaluator.evaluate_dataset('./schema/test.csv')
        results_df.to_csv('structural_evaluation_results.csv', index=False)
        print(f"\nStructural Evaluation Metrics ({metrics['total']} test cases):")
        print(f"Perfect Structural Match: {metrics['structural_accuracy']:.2%}")
        print(f"Avg SELECT F1: {metrics['avg_select_f1']:.2f}")
        print(f"Avg FROM F1: {metrics['avg_from_f1']:.2f}")
        print(f"Avg WHERE F1: {metrics['avg_where_f1']:.2f}")
        print(f"Average Latency: {metrics['avg_latency']:.2f}s")
    except Exception as e:
        logger.critical(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
