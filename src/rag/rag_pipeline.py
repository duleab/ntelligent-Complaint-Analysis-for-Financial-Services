import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Any

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Class for the Retrieval-Augmented Generation (RAG) pipeline."""
    
    def __init__(self, retriever, generator):
        """
        Initialize the RAG pipeline with the specified retriever and generator.
        
        Args:
            retriever: The retriever component
            generator: The generator component
        """
        self.retriever = retriever
        self.generator = generator
        logger.info("RAG pipeline initialized")
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the RAG pipeline for the given query.
        
        Args:
            query (str): The query to process
            
        Returns:
            dict: A dictionary containing the answer and source documents
        """
        try:
            # Step 1: Retrieve relevant documents
            logger.info(f"Processing query: {query}")
            documents = self.retriever.retrieve(query)
            
            if not documents:
                logger.warning("No relevant documents found")
                return {
                    "answer": "I couldn't find any relevant information to answer your question. Please try rephrasing or asking about a different topic.",
                    "source_documents": []
                }
            
            # Step 2: Generate answer
            answer = self.generator.generate(query, documents)
            
            # Step 3: Return result
            return {
                "answer": answer,
                "source_documents": documents
            }
        
        except Exception as e:
            logger.error(f"Error running RAG pipeline: {str(e)}")
            raise
    
    def run_with_feedback(self, query: str, feedback_loop: bool = False) -> Dict[str, Any]:
        """Run the RAG pipeline with an optional feedback loop for query refinement.
        
        Args:
            query (str): The query to process
            feedback_loop (bool): Whether to use a feedback loop for query refinement
            
        Returns:
            dict: A dictionary containing the answer and source documents
        """
        try:
            if not feedback_loop:
                return self.run(query)
            
            # Step 1: Initial retrieval
            logger.info(f"Processing query with feedback loop: {query}")
            initial_documents = self.retriever.retrieve(query)
            
            if not initial_documents:
                logger.warning("No relevant documents found in initial retrieval")
                return {
                    "answer": "I couldn't find any relevant information to answer your question. Please try rephrasing or asking about a different topic.",
                    "source_documents": []
                }
            
            # Step 2: Query refinement based on initial retrieval
            # This is a simplified version; in a real system, you might use the LLM to refine the query
            refined_query = self._refine_query(query, initial_documents)
            logger.info(f"Refined query: {refined_query}")
            
            # Step 3: Retrieval with refined query
            refined_documents = self.retriever.retrieve(refined_query)
            
            # Step 4: Generate answer
            answer = self.generator.generate(query, refined_documents)  # Note: we use the original query here
            
            # Step 5: Return result
            return {
                "answer": answer,
                "source_documents": refined_documents,
                "refined_query": refined_query
            }
        
        except Exception as e:
            logger.error(f"Error running RAG pipeline with feedback: {str(e)}")
            raise
    
    def _refine_query(self, original_query: str, documents: List) -> str:
        """Refine the query based on the retrieved documents.
        
        Args:
            original_query (str): The original query
            documents (list): The retrieved documents
            
        Returns:
            str: The refined query
        """
        # This is a placeholder for query refinement logic
        # In a real system, you might use the LLM to refine the query based on the retrieved documents
        
        # Extract key terms from documents
        key_terms = set()
        for doc in documents:
            # Extract product and issue from metadata
            product = doc.metadata.get("product", "")
            issue = doc.metadata.get("issue", "")
            
            if product:
                key_terms.add(product.lower())
            if issue:
                key_terms.add(issue.lower())
        
        # Add key terms to the query if they're not already present
        refined_query = original_query
        for term in key_terms:
            if term and term not in original_query.lower():
                refined_query += f" {term}"
        
        return refined_query
    
    def evaluate(self, evaluation_questions: List[str]) -> List[Dict[str, Any]]:
        """Evaluate the RAG pipeline on a set of questions.
        
        Args:
            evaluation_questions (list): List of questions to evaluate
            
        Returns:
            list: List of evaluation results
        """
        evaluation_results = []
        
        for question in evaluation_questions:
            try:
                result = self.run(question)
                evaluation_results.append({
                    "question": question,
                    "answer": result["answer"],
                    "source_documents": result["source_documents"]
                })
            except Exception as e:
                logger.error(f"Error evaluating question '{question}': {str(e)}")
                evaluation_results.append({
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "source_documents": []
                })
        
        return evaluation_results