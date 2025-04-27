import os
import google.generativeai as genai
import re
from dotenv import load_dotenv

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

def is_medical_query(user_query):
    """Determine if a query is related to medical database content"""
    # List of medical domain terms from our schema
    medical_terms = [
    'patient', 'hospital', 'admission', 'diagnosis', 'procedure', 
    'lab', 'test', 'medication', 'prescription', 'drug', 'dose',
    'icu', 'subject_id', 'hadm_id', 'stay_id', 'icd', 'chart', 
    'microbiology', 'treatment', 'cost', 'insurance', 'gender',
    'age', 'dob', 'dod', 'admittime', 'dischtime', 'care', 'event',
    'surgery', 'operation', 'repair', 'drainage', 'catheterization', 
    'bypass', 'valvuloplasty', 'hepat', 'coronary', 'cardiac',
    'angiography', 'arteriography', 'pleural', 'pericardial', 
    'vascular', 'esophagus', 'esophagectomy', 'colonoscopy',
    'endoscopy', 'fluoroscopy', 'intracardiac', 'graft', 
    'nutritional', 'gastrointestinal', 'mammary', 'hepatectomy',
    'device', 'prosthesis', 'fracture', 'fixation', 'balloon',
    'ventricle', 'atrial', 'septal', 'pancreatic', 'artery', 
    'vein', 'open heart', 'internal fixation', 'procedure_name',
    'chronic', 'acute', 'severe', 'remission', 'complication',
    'diagnostic', 'treatment_plan', 'risk_factor', 'comorbidity', 
    'emergency', 'therapy', 'recovery', 'medical_history', 
    'symptom', 'syndrome', 'hospitalization', 'adverse_event', 
    'discharge', 'mortality', 'outcome', 'follow_up', 'immunization', 
    'biopsy', 'radiology', 'ultrasound', 'ct_scan', 'mri', 'x_ray', 
    'blood_pressure', 'heart_rate', 'oxygen_level', 'respiration_rate', 
    'temperature', 'weight', 'height', 'bmi', 'glucose', 'cholesterol', 
    'medication_type', 'dosage', 'administered', 'allergy', 'contraindication',
    'side_effect', 'pharmacy', 'drug_interaction', 'prescribing', 
    'surgical_team', 'anesthesia', 'incision', 'post_op', 'recovery_time',
    'complication_rate', 'infection_rate', 'blood_loss', 'hospital_readmission',
    'clinical_trial', 'preoperative', 'postoperative', 'medical_device', 
    'implant', 'procedure_code', 'diagnostic_code', 'treatment_code', 
    'hospital_cost', 'billing', 'insurance_coverage', 'reimbursement'
]
 
    # Check if query contains medical terms
    query_lower = user_query.lower()
    for term in medical_terms:
        if term in query_lower:
            return True
            
    # Check for question patterns related to medical data
    medical_patterns = [
        r'patient.*\d+',
        r'hospital.*admission',
        r'medication',
        r'diagnosis',
        r'treatment',
        r'prescription',
        r'lab.*result',
        r'icu',
        r'cost.*treatment'
    ]
    
    for pattern in medical_patterns:
        if re.search(pattern, query_lower):
            return True
            
    return False

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
    You are an expert in SQL query generation. Using the database schema information and similar 
    examples below, convert the user question into a valid SQL query for a medical database.
    
    Database schema:
    {formatted_context['schema_context']}
    
    Similar examples:
    {formatted_context['examples_context']}
    
    User question: {user_query}
    
    IMPORTANT: If you determine the question is not related to medical data, patients, 
    treatments, medications, or hospital stays, respond ONLY with:
    "ABSTAIN: This question is not answerable with this medical database."
    
    Otherwise, generate only the SQL query without any explanation.
    
    Please ensure your SQL syntax follows SQLite format.
    """
    
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize the Gemini Pro model
        model = genai.GenerativeModel('gemini-1.5-pro')
        
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