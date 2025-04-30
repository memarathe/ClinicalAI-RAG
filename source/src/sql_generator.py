import os
import google.generativeai as genai
import re
from dotenv import load_dotenv
from abstain import is_medical_query
# Load environment variables
load_dotenv()

def format_context(search_results):
    """Format search results into context for the language model"""
    # Format schema results
    schema_context = ""
    for result in search_results['schema_results']:
        if 'table_name' in result and 'column_name' in result:
            schema_context += f"Table: {result['table_name']}, Column: {result['column_name']}\n"
            schema_context += f"Type: {result.get('column_type', '')}\n"
            schema_context += f"Description: {result.get('description', '')}\n\n"
        elif 'relationship' in result:
            schema_context += f"{result['relationship']}\n"
    
    # Format training examples
    examples_context = ""
    for result in search_results['train_results']:
        examples_context += f"Question: {result['question']}\n"
        examples_context += f"SQL: {result['query']}\n\n"
    
    return {
        "schema_context": schema_context.strip(),
        "examples_context": examples_context.strip()
    }


def clean_sql_query(sql):
    """Cleans SQL code block and ensures it starts with SELECT"""
    # Remove code fences
    sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^```\s*", "", sql)
    sql = re.sub(r"\s*```\s*$", "", sql)

    # Strip backticks and whitespace
    sql = sql.strip("` \n")

    # Remove leading non-SQL junk before SELECT
    match = re.search(r"(SELECT\s+.*)", sql, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("No SELECT statement found in input.")
    sql = match.group(1)

    # Remove inline comments
    sql = re.sub(r"--.*", "", sql)

    # Normalize whitespace and remove trailing semicolon
    sql = re.sub(r"\s+", " ", sql).strip().rstrip(';')

    return sql

# def is_medical_query(user_query):
#     """Determine if a query is related to medical database content"""
#     import re
    
#     # Expanded list with normalized terms (spaces instead of underscores)
#     medical_terms = [
#         'patient', 'hospital', 'admission', 'diagnosis', 'procedure',
#         'lab', 'test', 'medication', 'prescription', 'drug', 'dose',
#         'icu', 'subject id', 'hadm id', 'stay id', 'icd code', 'chart',
#         'microbiology', 'treatment', 'cost', 'insurance', 'gender',
#         'age', 'date of birth', 'date of death', 'admission time',
#         'discharge time', 'care', 'medical event', 'surgery', 
#         'operation', 'repair', 'drainage', 'catheter', 'bypass',
#         'valve repair', 'hepatic', 'heart', 'cardiac', 'imaging',
#         'x-ray', 'ct scan', 'mri', 'blood pressure', 'heart rate',
#         'oxygen level', 'respiratory rate', 'temperature', 'weight',
#         'height', 'bmi', 'glucose', 'cholesterol', 'allergy',
#         'side effect', 'pharmacy', 'diagnostic', 'therapy', 'recovery',
#         'medical history', 'symptom', 'diagnosis code', 'procedure code',
#         'insurance', 'billing', 'clinical trial'
#     ]

#     # Normalize query text
#     query_lower = re.sub(r'[^\w\s]', '', user_query.lower())  # Remove punctuation
#     query_words = set(query_lower.split())
    
#     # 1. Check for presence of core medical terms
#     core_terms = {'patient', 'hospital', 'medical', 'diagnosis', 'treatment'}
#     if core_terms & query_words:
#         return True

#     # 2. Check for presence of any medical terms
#     for term in medical_terms:
#         if term in query_lower:  # Substring match for multi-word terms
#             return True

#     # 3. Enhanced regex patterns
#     patterns = [
#         r'\b(patient|pt)\b.*\b\d{4,}\b',  # Patient IDs
#         r'\b(age|dob|gender)\b.*\b(patient|subject)\b',
#         r'\b(medication|prescription|drug)\b.*\b(history|dose)\b',
#         r'\b(diagnos|treatment)\b.*\b(heart|lung|kidney)\b',
#         r'\b(admission|discharge)\b.*\b(date|time)\b',
#         r'\b(lab|test)\b.*\b(result|value)\b',
#         r'\b(surgery|operation)\b.*\b(complication)\b'
#     ]
    
#     return any(re.search(pattern, query_lower) for pattern in patterns)

def generate_sql_query(user_query, formatted_context):
    """Generate SQL query using Gemini API with abstention capability"""
    # First check if query is related to medical domain
    if not is_medical_query(user_query):
        return "ABSTAIN: This question is not related to medical data available in this database."
    
    # Read API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "-- Gemini API key not found in environment variables. Please set GEMINI_API_KEY."

    # Create prompt for the language model
    prompt = f"""
    You are an expert in SQL query generation and a medical practitioner. Using the database schema information and similar 
    examples below,your task is to convert the user question into a valid SQL query for a medical database.
    
    Database schema:
    {formatted_context['schema_context']}
    
    Similar examples:
    {formatted_context['examples_context']}
    
    User question: {user_query}
    
    IMPORTANT: 
    Generate only the SQL query without any explanation.
    Strictly follow the SQLite syntax.
    """
    
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize the Gemini Pro model
        model = genai.GenerativeModel('gemini-2.0-flash') #Gemini 1.5 Flash
        
        # Set generation parameters for better SQL output
        generation_config = {
            "temperature": 0.2,  # Lower temperature for more deterministic outputs
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        # Generate the SQL query
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Extract the response text
        sql_query = response.text.strip()
        
        # Check if response indicates abstention
        if sql_query.startswith("ABSTAIN:"):
            return sql_query
        
        # Clean SQL query by removing markdown and formatting
        sql_query = clean_sql_query(sql_query)
        
        return sql_query
    except Exception as e:
        print(f"Gemini API call failed: {str(e)}")
        
        # Fallback: Pattern-based query generation
        if "patient" in user_query.lower() and any(str(i) in user_query for i in range(10)):
            # Extract patient ID using regex
            patient_id_match = re.search(r'\b\d+\b', user_query)
            patient_id = patient_id_match.group(0) if patient_id_match else "10000"
            
            return f"SELECT * FROM patients WHERE subject_id = {patient_id}"
        else:
            return "-- Gemini API call failed. Unable to generate SQL query."