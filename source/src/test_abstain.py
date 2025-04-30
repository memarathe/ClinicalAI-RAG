import re
from collections import Counter
"""
Test script to verify the improved is_medical_query function
"""

# Import the improved function (assumes it's in a file called medical_query_detector.py)
# In your actual implementation, adjust the import statement as needed
from abstain import is_medical_query

# Test cases
test_queries = [
    # Should be identified as medical
    "What are the consumption methods of ampicillin sodium?",
    "How is aspirin administered?",
    "List all patients who received ampicillin",
    "How many patients received medication for heart disease?",
    "What are the average lab values for ICU patients?",
    "Show me patient admission data from 2019",
    "Calculate mortality rate for patients over 65",
    "What are the most prescribed antibiotics?",
    
    # Ambiguous cases - should still be identified based on context clues
    "Find information about penicillin",
    "Show data about drug administration",
    "What's the dosage information?",
    "List methods of consumption",
    
    # Non-medical cases - should be correctly identified as non-medical
    "Show me all data in the database",
    "How to query SQL database",
    "List top 10 products by sales",
    "Calculate average price of items",
    "Show employee information"
]

# Run tests
print("Testing improved medical query detection")
print("-" * 50)
for query in test_queries:
    result = is_medical_query(query)
    print(f"Query: '{query}'")
    print(f"Result: {'MEDICAL' if result else 'NON-MEDICAL'}")
    print("-" * 50)