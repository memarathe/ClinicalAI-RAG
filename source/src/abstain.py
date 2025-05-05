def is_medical_query(user_query):
    """
    Determine if a query is related to medical database content with improved accuracy.
    
    The function uses multiple approaches to identify medical queries:
    1. Medical terminology detection
    2. Database-specific term recognition
    3. Intent classification
    4. Context-aware pattern matching
    5. Negative filtering (to exclude non-medical queries)
    
    Args:
        user_query (str): The user's input query to analyze
        
    Returns:
        bool: True if the query is medical-related, False otherwise
    """
    import re
    from collections import Counter
    
    # Normalize query for consistent processing
    query_lower = user_query.lower()
    # Remove punctuation but preserve spaces between words
    normalized_query = re.sub(r'[^\w\s]', ' ', query_lower)
    query_words = normalized_query.split()
    query_word_set = set(query_words)
    
    # 1. MEDICAL DOMAIN TERMINOLOGY - Categorized and weighted
    medical_categories = {
        'high_relevance': {  # Core medical terms - strong indicators
            'patient', 'hospital', 'diagnosis', 'medication', 'treatment',
            'prescription', 'symptom', 'disease', 'doctor', 'nurse', 'physician',
            'surgeon', 'clinical', 'medical', 'health', 'healthcare', 'drug', 'medicine'
        },
        'patient_data': {  # Patient-specific information
            'admission', 'discharge', 'stay', 'subject id', 'hadm id', 'stay id',
            'date of birth', 'date of death', 'gender', 'age', 'weight', 'height',
            'bmi', 'vitals', 'chart', 'record'
        },
        'clinical_procedures': {  # Medical procedures and interventions
            'surgery', 'operation', 'procedure', 'catheter', 'bypass', 'repair',
            'transplant', 'implant', 'therapy', 'treatment', 'intervention', 
            'administration', 'consumption', 'method', 'delivery'
        },
        'diagnostics': {  # Tests and diagnostics
            'lab', 'test', 'x-ray', 'ct scan', 'mri', 'imaging', 'ultrasound',
            'biopsy', 'screening', 'analysis', 'examination', 'blood test'
        },
        'measurements': {  # Medical measurements and vitals
            'blood pressure', 'heart rate', 'pulse', 'temperature', 'oxygen', 'spo2',
            'respiratory rate', 'glucose', 'cholesterol', 'bmi', 'level'
        },
        'medications': {  # Medicine-related terms
            'drug', 'dose', 'medication', 'prescription', 'antibiotic', 'vaccine',
            'pharmacy', 'pharmaceutical', 'pill', 'capsule', 'injection', 'tablet',
            'solution', 'suspension', 'oral', 'intravenous', 'topical', 'inhaler',
            'patch', 'suppository', 'cream', 'ointment', 'drops', 'syrup',
            'ampicillin', 'penicillin', 'aspirin', 'paracetamol', 'ibuprofen',
            'consumption', 'route', 'administration', 'dosage'
        },
        'medical_specialties': {  # Medical specialties and departments
            'cardiology', 'neurology', 'pediatrics', 'oncology', 'radiology',
            'orthopedic', 'psychiatric', 'icu', 'emergency', 'intensive care'
        },
        'anatomy': {  # Body parts and systems
            'heart', 'lung', 'liver', 'kidney', 'brain', 'cardiac', 'pulmonary',
            'renal', 'hepatic', 'neural', 'vascular', 'respiratory', 'digestive'
        },
        'conditions': {  # Medical conditions
            'infection', 'disease', 'syndrome', 'disorder', 'failure', 'injury',
            'inflammation', 'fracture', 'cancer', 'diabetes', 'hypertension'
        },
        'administrative': {  # Healthcare administration
            'insurance', 'billing', 'cost', 'payment', 'reimbursement', 'claim',
            'coverage', 'provider', 'icd code', 'cpt code', 'drg'
        }
    }
    
    # 2. DATABASE-SPECIFIC TERMS - MIMIC database tables/fields
    mimic_specific_terms = {
        'tables': {
            'patients', 'admissions', 'icustays', 'chartevents', 'labevents',
            'prescriptions', 'procedures', 'diagnoses', 'noteevents', 'transfers'
        },
        'ids': {
            'subject_id', 'hadm_id', 'stay_id', 'caregiver_id', 'transfer_id', 
            'itemid', 'charttime', 'storetime'
        },
        'fields': {
            'los', 'dod', 'dob', 'deathtime', 'intime', 'outtime', 'admittime',
            'dischtime', 'icd9_code', 'icd10_code', 'ndc', 'valuenum'
        }
    }
    
    # 3. CONTEXTUAL PATTERNS - Common query patterns in medical contexts
    medical_patterns = [
        # Patient identifiers
        r'\b(patient|pt|subject)(?:\s+id)?\s+\d+\b',
        r'\b(admission|visit)\s+(?:for|of|by)\b',
        
        # Clinical questions
        r'\b(how many|average|count)\s+(?:of\s+)?(patient|admission)',
        r'\b(patient|admission|stay).+\b(demographics|characteristic)',
        r'\bpatient.+\bage\b',
        r'\b(patient|admission).+\bgender\b',
        
        # Medication patterns
        r'\b(medication|drug|prescription|medicine).+\b(given|prescribed|administered|used|taken)\b',
        r'\b(dose|dosage|frequency).+\b(medication|drug|medicine)\b',
        r'\b(consumption|administration|route).+\b(method|drug|medication)\b',
        r'\bhow\s+(?:is|are)\s+\w+\s+(?:administered|given|taken|used)\b',
        r'\bwhat\s+(?:are|is)(?:\s+the)?\s+(?:consumption|administration)\s+methods?\b',
        r'\bhow\s+to\s+(?:take|administer|use|consume)\s+\w+\b',
        
        # Drug-specific patterns
        r'\b\w+(?:\s+\w+)?\s+(?:tablet|pill|injection|solution|suspension|syrup)\b',
        r'\b(?:oral|intravenous|topical|iv|im|sc|subcutaneous)\s+(?:administration|route)\b',
        r'\bampicillin\b',  # Specific drug mention
        r'\bpenicillin\b',  # Specific drug mention
        r'\b\w+\s+sodium\b',  # Common drug formulation pattern
        
        # Diagnostic patterns
        r'\b(test|lab).+\b(result|finding|value)\b',
        r'\b(abnormal|elevated|high|low).+\b(lab|test|value)\b',
        
        # Treatment patterns
        r'\b(treatment|procedure).+\b(performed|conducted|completed)\b',
        r'\b(surgery|operation).+\b(time|duration|outcome|complication)\b',
        
        # Outcome patterns
        r'\b(mortality|survival|death|outcome).+\b(rate|percentage|risk)\b',
        r'\b(length of stay|los).+\b(days|average|median)\b',
        
        # Temporal patterns
        r'\b(between|from).+\b(date|time|year|month|day)\b.+\b(admission|stay)\b',
        r'\b(before|after|during).+\b(admission|procedure|treatment)\b',
        
        # Administrative patterns
        r'\b(cost|charge|payment|bill).+\b(admission|procedure|treatment)\b',
        r'\b(insurance|coverage).+\b(patient|admission|treatment)\b',
        
        # MIMIC-specific patterns
        r'\bicu\s+stay\b',
        r'\bchartevents\b',
        r'\blabevents\b'
    ]
    
    # 4. NEGATIVE PATTERNS - Things that indicate non-medical queries
    non_medical_patterns = [
        # Common non-medical SQL practice queries
        r'\b(customer|order|product|employee|sales|store|inventory)\b',
        r'\b(select|retrieve|display).+\b(all|everything)\b',
        r'\b(test|sample|example).+\b(query|database)\b',
        
        # Educational/tutorial requests
        r'\bshow\s+me\s+how\s+to\b',
        r'\btutorial\b',
        r'\bexplain\s+sql\b',
        
        # Generic data requests
        r'\b(most|popular|best|top|largest|smallest)\s+\d+\b',
        
        # Entertainment/retail queries
        r'\b(movie|film|book|game|product|item)\b',
        r'\b(actor|director|author|artist|singer)\b',
        
        # Business queries
        r'\b(company|business|market|industry|stock|share|profit|revenue)\b',
        
        # Social media terms
        r'\b(user|post|comment|like|share|follow|friend)\b',
        
        # Travel queries
        r'\b(flight|hotel|trip|travel|booking|reservation)\b'
    ]

    # 5. SCORING SYSTEM - Calculate relevance score
    score = 0
    
    # a. Category scoring - check for medical terms by category
    for category, terms in medical_categories.items():
        category_matches = sum(1 for term in terms if term in normalized_query)
        
        # Weight certain categories higher than others
        if category == 'high_relevance':
            score += category_matches * 3
        else:
            score += category_matches * 1
            
    # b. Database-specific scoring
    for category, terms in mimic_specific_terms.items():
        # Table names are strong indicators
        if category == 'tables':
            score += sum(1 for term in terms if term in normalized_query) * 3
        else:
            score += sum(1 for term in terms if term in normalized_query) * 1.5
    
    # c. Pattern matching
    for pattern in medical_patterns:
        if re.search(pattern, normalized_query):
            score += 2
    
    # d. Density calculation - what percentage of words are medical terms?
    all_medical_terms = set()
    for category_terms in medical_categories.values():
        all_medical_terms.update(category_terms)
    for category_terms in mimic_specific_terms.values():
        all_medical_terms.update(category_terms)
    
    medical_word_count = sum(1 for word in query_word_set if word in all_medical_terms)
    if len(query_words) > 0:
        medical_density = medical_word_count / len(query_words)
        # Boost score if high density of medical terms
        if medical_density > 0.3:
            score += 2
    
    # e. Negative scoring - reduce score for non-medical patterns
    for pattern in non_medical_patterns:
        if re.search(pattern, normalized_query):
            score -= 2
    
    # f. N-gram matching for multi-word medical terms
    for category, terms in medical_categories.items():
        for term in terms:
            if ' ' in term and term in normalized_query:
                # Multi-word matches are stronger indicators
                score += 1.5
    
    # g. Query intent analysis
    intent_indicators = {
        'find': 0.5, 'get': 0.5, 'show': 0.5, 'list': 0.5, 'select': 0.5,
        'count': 0.5, 'average': 0.5, 'sum': 0.5, 'max': 0.5, 'min': 0.5,
        'compare': 0.5, 'analyze': 0.5, 'calculate': 0.5, 'determine': 0.5
    }
    
    for word, value in intent_indicators.items():
        if word in query_word_set:
            score += value
    
    # 6. Check for direct or highly indicative mentions
    if any(term in normalized_query for term in ['mimic', 'medical database', 'hospital database', 'clinical data']):
        score += 5
        
    # Medication-specific check - critically important for drug-related queries
    medication_related_terms = ['consumption method', 'administration route', 'how to take', 
                               'how to use', 'drug delivery', 'medication use', 'dosage form']
    
    # Check for medication-related phrases and boost score if found
    if any(term in normalized_query for term in medication_related_terms):
        score += 4
    
    # 7. Query length consideration - very short queries need stronger indicators
    if len(query_words) < 4:
        threshold = 4  # Higher threshold for very short queries
    else:
        threshold = 3  # Standard threshold
        
    # Special case for specific drug names - these are highly indicative of medical queries
    common_drugs = ['ampicillin', 'penicillin', 'aspirin', 'ibuprofen', 'paracetamol', 
                    'acetaminophen', 'amoxicillin', 'morphine', 'codeine', 'warfarin',
                    'insulin', 'metformin', 'atorvastatin', 'lisinopril', 'metoprolol',
                    'levothyroxine', 'albuterol', 'losartan', 'simvastatin']
    
    # Specific drug formulation patterns
    drug_formulations = [r'\b\w+\s+sodium\b', r'\b\w+\s+hydrochloride\b', 
                         r'\b\w+\s+sulfate\b', r'\b\w+\s+citrate\b',
                         r'\b\w+\s+phosphate\b', r'\b\w+\s+tartrate\b']
    
    # If a specific drug name is mentioned, strongly boost the score
    if any(drug in normalized_query for drug in common_drugs):
        score += 5
    
    # If a drug formulation pattern is matched, boost the score
    if any(re.search(pattern, normalized_query) for pattern in drug_formulations):
        score += 3
       
    # Return final decision based on score
    return score >= threshold