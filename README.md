# ClinicalAI-RAG
Natural Language to SQL generation solution for medical Practitioners.

Usage Instructions
Set up the system:

$ python ./src/main.py --setup --schema ./schema/schema.sql --train ./schema/train.csv
Run a single query:
python main.py --query "What are the side effects of ampicillin sodium?"

Use in interactive mode:
python main.py
This implementation creates a complete RAG-based text-to-SQL system that:

1. Parses the schema directly from SQL file
2. Vectorizes both schema elements and training examples
3. Uses similarity search to find relevant context
4. Generates SQL queries using retrieved context
5. Executes the queries against MIMIC-IV database
