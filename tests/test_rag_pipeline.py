import os
import sys
import unittest
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.retriever import VectorStoreRetriever
from src.rag.generator import LLMGenerator
from src.rag.rag_pipeline import RAGPipeline

class TestRAGPipeline(unittest.TestCase):
    """Test cases for the RAG pipeline components."""
    
    def setUp(self):
        """Set up test environment."""
        # Skip tests if vector store doesn't exist
        self.vector_store_path = Path("../vector_store")
        if not self.vector_store_path.exists():
            self.skipTest("Vector store not found. Run data processing pipeline first.")
    
    def test_retriever_initialization(self):
        """Test that the retriever can be initialized."""
        try:
            retriever = VectorStoreRetriever(
                vector_store_type="FAISS",
                vector_store_path="../vector_store",
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.assertIsNotNone(retriever)
            self.assertIsNotNone(retriever.get_retriever())
        except Exception as e:
            self.fail(f"Retriever initialization failed with error: {str(e)}")
    
    def test_retriever_search(self):
        """Test that the retriever can search for documents."""
        retriever = VectorStoreRetriever(
            vector_store_type="FAISS",
            vector_store_path="../vector_store",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            top_k=3
        )
        
        # Test search functionality
        query = "credit card fees"
        docs = retriever.retrieve(query)
        
        # Check that we got results
        self.assertIsNotNone(docs)
        self.assertGreater(len(docs), 0)
        self.assertLessEqual(len(docs), 3)
        
        # Check that the documents have the expected structure
        for doc in docs:
            self.assertTrue(hasattr(doc, 'page_content'))
            self.assertTrue(hasattr(doc, 'metadata'))
            self.assertIn('product', doc.metadata)
    
    def test_generator_initialization(self):
        """Test that the generator can be initialized."""
        try:
            generator = LLMGenerator(
                model_name="google/flan-t5-base",  # Use a smaller model for testing
                temperature=0.5,
                max_tokens=256
            )
            self.assertIsNotNone(generator)
            self.assertIsNotNone(generator.get_llm())
            self.assertIsNotNone(generator.get_prompt_template())
        except Exception as e:
            self.fail(f"Generator initialization failed with error: {str(e)}")
    
    def test_rag_pipeline_initialization(self):
        """Test that the RAG pipeline can be initialized."""
        try:
            # Initialize components
            retriever = VectorStoreRetriever(
                vector_store_type="FAISS",
                vector_store_path="../vector_store",
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                top_k=3
            )
            
            generator = LLMGenerator(
                model_name="google/flan-t5-base",  # Use a smaller model for testing
                temperature=0.5,
                max_tokens=256
            )
            
            # Initialize pipeline
            pipeline = RAGPipeline(
                retriever=retriever,
                generator=generator
            )
            
            self.assertIsNotNone(pipeline)
            
            # Test basic pipeline functionality if vector store is available
            if retriever.vector_store is not None:
                result = pipeline.run("What are common issues with credit cards?")
                self.assertIsNotNone(result)
                self.assertIn("answer", result)
                self.assertIn("source_documents", result)
        except Exception as e:
            self.fail(f"RAG pipeline initialization failed with error: {str(e)}")

if __name__ == "__main__":
    unittest.main()