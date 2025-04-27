import re

def parse_schema_sql(sql_path):
    with open(sql_path, 'r') as f:
        sql = f.read()

    tables = re.findall(r'CREATE TABLE (\w+)\s*\((.*?)\);', sql, re.DOTALL)
    records = []
    for table_name, columns_block in tables:
        # Split columns by line, ignore empty lines
        for line in columns_block.split('\n'):
            line = line.strip()
            if not line or line.startswith('--'):
                continue
            # Extract column name, type, and optional comment
            m = re.match(r'(\w+)\s+([\w\(\)]+)[^,]*?(?:--\s*(.*))?', line)
            if m:
                col, typ, comment = m.groups()
                comment = comment or ""
                records.append({
                    'table_name': table_name,
                    'column_name': col,
                    'type': typ,
                    'description': comment,
                    'text_for_embedding': f"Table: {table_name}\nColumn: {col}\nType: {typ}\nDescription: {comment}"
                })
    return records
