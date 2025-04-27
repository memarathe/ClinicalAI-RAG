import os
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

class QueryClassifier:
    def __init__(self, vector_db_dir='vector_db', model_name='all-MiniLM-L6-v2', threshold=0.3):
        """Initialize the query classifier with model and threshold."""
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)
        
        # Load schema metadata for domain verification
        self.schema_metadata = pd.read_csv(os.path.join(vector_db_dir, 'schema_metadata.csv'))
        
        # Extract medical domain terms from schema
        self.domain_terms = self._extract_domain_terms()
        
    def _extract_domain_terms(self):
        """Extract domain-specific terms from the schema metadata."""
        terms = set()
        # Add table names
        if 'table_name' in self.schema_metadata.columns:
            terms.update(self.schema_metadata['table_name'].dropna().unique())
        
        # Add column names
        if 'column_name' in self.schema_metadata.columns:
            terms.update(self.schema_metadata['column_name'].dropna().unique())
            
        # Add terms from descriptions
        if 'description' in self.schema_metadata.columns:
            for desc in self.schema_metadata['description'].dropna():
                # Add medical terms from descriptions
                words = desc.lower().split()
                terms.update(words)
        
        return terms
        
    def is_relevant_to_medical_db(self, query_text):
        """Determine if the query is relevant to the medical database."""
        # Use embedding similarity to determine relevance
        query_embedding = self.model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        # Get schema topics to compare against
        medical_topics = [
            "patient medical history",
            "hospital admission data",
            "medical diagnoses",
            "laboratory tests",
            "medications and prescriptions",
            "medical procedures",
            "vital signs",
            "patient demographics",
            "hospital stay information"
        ]
        
        topic_embeddings = self.model.encode(medical_topics)
        faiss.normalize_L2(topic_embeddings)
        
        # Create a temporary index for these topics
        dimension = topic_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(topic_embeddings)
        
        # Search for similar topics
        D, I = index.search(query_embedding, 1)
        max_similarity = float(D[0][0])
        
        # Simple keyword matching as backup
        query_lower = query_text.lower()
        keyword_match = any(term in query_lower for term in self.domain_terms)
        
        # Combined decision
        if max_similarity > self.threshold or keyword_match:
            return True, max_similarity
        
        return False, max_similarity
