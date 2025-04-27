import sqlite3
import pandas as pd

def execute_sql_query(sql_query, db_path="mimic_iv.sqlite"):
    """Execute SQL query on the database and return results"""
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        
        # Execute the query
        result_df = pd.read_sql_query(sql_query, conn)
        
        # Close the connection
        conn.close()
        
        return {
            "success": True,
            "data": result_df
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def format_results(results):
    """Format query results for display"""
    if results["success"]:
        df = results["data"]
        if len(df) == 0:
            return "Query executed successfully, but returned no results."
        else:
            return df
    else:
        return f"Error executing query: {results['error']}"
