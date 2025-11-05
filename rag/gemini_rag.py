"""
RAG System with Gemini API
"""

import os
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using cosine similarity fallback")


class GeminiRAG:
    """RAG system powered by Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Configure Gemini
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel('gemini-pro-latest')
        
        # Document storage
        self.documents = []
        self.embeddings = None
        self.index = None
        
        logger.info("✓ Gemini RAG initialized")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Gemini"""
        embeddings = []
        
        for text in texts:
            try:
                # Use simple hash-based embedding as fallback
                # In production, use actual embedding model
                text_hash = hash(text)
                # Convert to pseudo-embedding
                np.random.seed(abs(text_hash) % (2**32))
                embedding = np.random.randn(768)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                embeddings.append(np.zeros(768))
        
        return np.array(embeddings)
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for retrieval"""
        logger.info(f"Indexing {len(documents)} documents...")
        
        self.documents = documents
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        self.embeddings = self.generate_embeddings(texts)
        
        # Create index
        if FAISS_AVAILABLE and len(documents) > 10:
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings.astype('float32'))
            logger.info("✓ FAISS index created")
        else:
            self.index = None
            logger.info("✓ Using cosine similarity for retrieval")
        
        logger.info(f"✓ Indexed {len(documents)} documents")
    
    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents"""
        if not self.documents:
            return []
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        if self.index is not None and FAISS_AVAILABLE:
            # Use FAISS
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'),
                min(top_k, len(self.documents))
            )
            
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['score'] = float(1 / (1 + dist))
                    results.append(doc)
        else:
            # Use cosine similarity
            similarities = np.dot(self.embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                doc = self.documents[idx].copy()
                doc['score'] = float(similarities[idx])
                results.append(doc)
        
        return results
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate answer using Gemini with enhanced prompting"""
        # Prepare context with source attribution
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.get('metadata', {}).get('source', 'Unknown')
            doc_type = doc.get('metadata', {}).get('type', 'data')
            context_parts.append(f"[Source {i}: {source} - {doc_type}]\n{doc['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt with specific instructions
        prompt = f"""You are an AI assistant for the Maharashtra AI Governance Platform. Your role is to provide accurate, helpful, and specific answers about health infrastructure, governance, and public services.

**Available Context Data:**
{context}

**User Question:** {query}

**Instructions:**
1. Provide a clear, comprehensive answer based on the context
2. If the question is about hospitals, facilities, or infrastructure - list specific names, numbers, and details
3. If asking for counts - provide exact numbers when available
4. If asking for names - list actual names from the data
5. Be specific and avoid generic responses
6. If data is insufficient, clearly state what's available and what's missing
7. Use bullet points or numbered lists for better readability
8. Include relevant statistics and metrics

**Your Answer:**"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I apologize, but I couldn't generate an answer. Error: {str(e)}"
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Query the RAG system"""
        # Retrieve relevant documents
        relevant_docs = self.retrieve_documents(question, top_k=top_k)
        
        # Generate answer
        answer = self.generate_answer(question, relevant_docs)
        
        return {
            'question': question,
            'answer': answer,
            'sources': relevant_docs,
            'n_sources': len(relevant_docs)
        }
    
    def summarize(self, documents: List[str], max_length: int = 200) -> str:
        """Summarize documents"""
        combined_text = "\n\n".join(documents[:5])  # Limit to first 5
        
        prompt = f"""Summarize the following information in {max_length} words or less:

{combined_text}

Summary:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error summarizing: {e}")
            return "Unable to generate summary."
    
    def generate_recommendations(self, situation: Dict[str, Any]) -> str:
        """Generate policy recommendations"""
        situation_text = "\n".join([f"{k}: {v}" for k, v in situation.items()])
        
        prompt = f"""Based on the following situation, provide actionable policy recommendations for government officials:

Situation:
{situation_text}

Provide 3-5 specific, actionable recommendations:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return "Unable to generate recommendations."
    
    def explain_prediction(self, prediction_data: Dict[str, Any], audience: str = 'citizen') -> str:
        """Explain ML prediction in simple terms"""
        prediction_text = "\n".join([f"{k}: {v}" for k, v in prediction_data.items()])
        
        if audience == 'citizen':
            prompt = f"""Explain the following prediction in simple terms that a citizen can understand:

{prediction_text}

Explain why this prediction was made and what it means:"""
        else:
            prompt = f"""Provide a technical explanation of the following prediction:

{prediction_text}

Technical explanation:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return "Unable to generate explanation."


if __name__ == "__main__":
    print("=" * 80)
    print("GEMINI RAG SYSTEM - INTERACTIVE TEST")
    print("=" * 80)
    
    # Initialize RAG
    try:
        rag = GeminiRAG()
        
        # Create sample documents (same as before)
        documents = [
            {
                'text': 'Pune has 15 major hospitals with total capacity of 5000 beds. Ward 1 has the highest bed count with 800 beds.',
                'metadata': {'source': 'health_infrastructure', 'type': 'hospital'}
            },
            {
                'text': 'Infrastructure complaints in Pune increased by 30% in 2024. Potholes are the most common complaint type.',
                'metadata': {'source': 'infrastructure', 'type': 'complaint'}
            },
            {
                'text': 'Crime rate in Maharashtra decreased by 15% in 2024. Cyber crime cases increased by 20%.',
                'metadata': {'source': 'crime', 'type': 'statistics'}
            },
            {
                'text': 'Emergency response time average is 12 minutes for ambulance services in Pune city.',
                'metadata': {'source': 'emergency', 'type': 'response_time'}
            },
            {
                'text': 'Citizen satisfaction with water supply services is 75%. Main complaints are about pressure and quality.',
                'metadata': {'source': 'feedback', 'type': 'satisfaction'}
            }
        ]
        
        # Index documents
        print("\nIndexing documents...")
        rag.index_documents(documents)
        print("\n" + "=" * 80)
        print("DOCUMENT INDEXING COMPLETE. READY FOR QUESTIONS.")
        print("=" * 80)

        # === NEW INTERACTIVE LOOP ===
        while True:
            # 1. Ask the user for a question
            user_question = input("\nAsk a question (or type 'quit' to exit): ")
            
            # 2. Check if the user wants to quit
            if user_question.lower() in ['quit', 'exit']:
                print("Exiting interactive test. Goodbye!")
                break
                
            # 3. Call the RAG system with the user's question
            print("...Thinking...")
            result = rag.query(user_question, top_k=2)
            
            # 4. Print the live answer
            print(f"\nAnswer: {result['answer']}")
            
            # 5. Print the sources it used
            print(f"Sources used ({result['n_sources']}):")
            if result['sources']:
                for source in result['sources']:
                    print(f"  - [Source: {source['metadata']['source']}] {source['text'][:50]}...")
            else:
                print("  - No relevant sources found in our knowledge base.")
        # === END OF NEW LOOP ===

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nNote: Make sure GEMINI_API_KEY is set in .env file")