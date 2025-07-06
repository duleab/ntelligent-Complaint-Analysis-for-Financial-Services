import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(sample_size=None, vector_store_type="FAISS"):
    """Run the complete data processing pipeline.
    
    Args:
        sample_size (int, optional): Number of rows to sample from the dataset.
        vector_store_type (str): Type of vector store to create ("FAISS" or "ChromaDB").
    """
    try:
        # Step 1: Download the data
        logger.info("Step 1: Downloading CFPB complaint data")
        from src.data.download_data import download_cfpb_data
        
        raw_data_path = download_cfpb_data("data/raw", sample_size)
        logger.info(f"Data downloaded to {raw_data_path}")
        
        # Step 2: Process the data
        logger.info("Step 2: Processing the data")
        from src.data.data_processor import DataProcessor
        
        processed_data_path = "data/processed/processed_complaints.csv"
        processor = DataProcessor(raw_data_path=raw_data_path, processed_data_path=processed_data_path)
        
        processor.process_pipeline(
            products=[
                "Credit card",
                "Mortgage",
                "Debt collection",
                "Credit reporting, credit repair services, or other personal consumer reports",
                "Bank account or service",
                "Student loan",
                "Money transfer, virtual currency, or money service",
                "Payday loan, title loan, personal loan, or advance loan",
                "Vehicle loan or lease",
                "Prepaid card"
            ],
            min_narrative_length=50
        )
        logger.info(f"Data processed and saved to {processed_data_path}")
        
        # Step 3: Create the vector store
        logger.info("Step 3: Creating the vector store")
        from src.data.create_vector_store import create_vector_store
        
        vector_store_path = create_vector_store(
            raw_data_path=raw_data_path,
            processed_data_path=processed_data_path,
            vector_store_path="vector_store",
            vector_store_type=vector_store_type.upper(),  # Convert to uppercase
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info(f"Vector store created at {vector_store_path}")
        
        logger.info("Pipeline completed successfully!")
        logger.info("You can now run the Streamlit app with: python run_app.py")
        
        return True
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the complete data processing pipeline")
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=10000,
        help="Number of rows to sample from the dataset. Default is 10,000."
    )
    parser.add_argument(
        "--vector-store-type", 
        type=str, 
        choices=["faiss", "chroma"],
        default="faiss",
        help="Type of vector store to create. Default is 'faiss'."
    )
    args = parser.parse_args()
    
    success = run_pipeline(args.sample_size, args.vector_store_type)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())