import os
import sys
from pathlib import Path
import pandas as pd
import logging
import argparse

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import project modules
from src.data.data_processor import DataProcessor
from src.data.text_processor import TextProcessor
from langchain_community.vectorstores import FAISS, Chroma

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_vector_store(raw_data_path, processed_data_path, vector_store_path, 
                       vector_store_type="FAISS", embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                       products=None, sample_size=None):
    """Create a vector store from raw complaint data.
    
    Args:
        raw_data_path (str): Path to the raw data file
        processed_data_path (str): Path to save the processed data
        vector_store_path (str): Path to save the vector store
        vector_store_type (str): Type of vector store to create ("FAISS" or "ChromaDB")
        embedding_model (str): Name of the embedding model to use
        products (list): List of products to include (if None, include all)
        sample_size (int): Number of records to sample (if None, use all)
    """
    try:
        # Step 1: Process the raw data
        logger.info("Step 1: Processing raw data")
        data_processor = DataProcessor(raw_data_path, processed_data_path)
        
        # Define products of interest if not provided
        if products is None:
            products = [
                "Credit card",
                "Mortgage",
                "Debt collection",
                "Credit reporting, credit repair services, or other personal consumer reports",
                "Checking or savings account",
                "Money transfer, virtual currency, or money service",
                "Personal loan"
            ]
        
        # Process the data
        processed_data = data_processor.process_pipeline(products=products, min_narrative_length=100)
        logger.info(f"Processed {len(processed_data)} records")
        
        # Sample the data if requested
        if sample_size is not None and sample_size < len(processed_data):
            processed_data = processed_data.sample(sample_size, random_state=42)
            logger.info(f"Sampled {len(processed_data)} records")
        
        # Step 2: Create text chunks and embeddings
        logger.info("Step 2: Creating text chunks and embeddings")
        text_processor = TextProcessor(embedding_model_name=embedding_model)
        
        # Process the dataframe
        chunks, metadatas = text_processor.process_dataframe(
            df=processed_data, 
            text_column='cleaned_narrative',
            metadata_columns=['product', 'issue', 'company']
        )
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3: Create and save the vector store
        logger.info(f"Step 3: Creating {vector_store_type} vector store")
        vector_store = text_processor.create_vector_store(
            chunks=chunks,
            metadatas=metadatas,
            vector_store_type=vector_store_type,
            persist_directory=vector_store_path
        )
        logger.info(f"Vector store created and saved to {vector_store_path}")
        
        # Step 4: Test the vector store
        logger.info("Step 4: Testing the vector store")
        test_queries = [
            "What are common issues with credit cards?",
            "Why are customers unhappy with mortgage services?",
            "What problems do people face with debt collection?"
        ]
        
        for query in test_queries:
            results = vector_store.similarity_search(query, k=2)
            logger.info(f"\nQuery: {query}")
            for i, doc in enumerate(results):
                logger.info(f"Result {i+1}:")
                logger.info(f"Product: {doc.metadata.get('product', 'Unknown')}")
                logger.info(f"Issue: {doc.metadata.get('issue', 'Unknown')}")
                logger.info(f"Content: {doc.page_content[:100]}...")
        
        logger.info("Vector store creation completed successfully")
        return vector_store
    
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create a vector store from complaint data")
    parser.add_argument("--raw_data", type=str, default="../data/raw/complaints.csv",
                        help="Path to the raw data file")
    parser.add_argument("--processed_data", type=str, default="../data/processed/processed_complaints.csv",
                        help="Path to save the processed data")
    parser.add_argument("--vector_store", type=str, default="../vector_store",
                        help="Path to save the vector store")
    parser.add_argument("--vector_store_type", type=str, default="FAISS", choices=["FAISS", "ChromaDB"],
                        help="Type of vector store to create")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Name of the embedding model to use")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of records to sample (if None, use all)")
    
    args = parser.parse_args()
    
    # Convert paths to absolute paths
    raw_data_path = Path(args.raw_data).resolve()
    processed_data_path = Path(args.processed_data).resolve()
    vector_store_path = Path(args.vector_store).resolve()
    
    # Create the vector store
    create_vector_store(
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        vector_store_path=vector_store_path,
        vector_store_type=args.vector_store_type,
        embedding_model=args.embedding_model,
        sample_size=args.sample_size
    )

if __name__ == "__main__":
    main()