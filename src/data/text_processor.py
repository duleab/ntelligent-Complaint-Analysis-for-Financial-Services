import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple, Optional, Union

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextProcessor:
    """Class for processing text data for the RAG pipeline, including chunking and embedding."""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the text processor.
        
        Args:
            embedding_model_name (str): Name of the embedding model to use
        """
        self.embedding_model_name = embedding_model_name
        self.embeddings = None
        self.text_splitter = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize the text splitter and embeddings."""
        try:
            # Import necessary modules
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            # Initialize text splitter
            logger.info("Initializing text splitter")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # Initialize embeddings
            logger.info(f"Initializing embeddings with model: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            
            logger.info("Components initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text (str): The text to split
            
        Returns:
            list: List of text chunks
        """
        if not self.text_splitter:
            logger.warning("Text splitter not initialized. Initializing components.")
            self._initialize_components()
        
        try:
            logger.info("Chunking text")
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a text.
        
        Args:
            text (str): The text to embed
            
        Returns:
            list: The embedding vector
        """
        if not self.embeddings:
            logger.warning("Embeddings not initialized. Initializing components.")
            self._initialize_components()
        
        try:
            logger.info("Generating embedding")
            embedding = self.embeddings.embed_query(text)
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.
        
        Args:
            documents (list): List of documents to embed
            
        Returns:
            list: List of embedding vectors
        """
        if not self.embeddings:
            logger.warning("Embeddings not initialized. Initializing components.")
            self._initialize_components()
        
        try:
            logger.info(f"Generating embeddings for {len(documents)} documents")
            embeddings = self.embeddings.embed_documents(documents)
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'cleaned_narrative', 
                          metadata_columns: List[str] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process a dataframe to create chunks and metadata.
        
        Args:
            df (pd.DataFrame): The dataframe to process
            text_column (str): The column containing the text to process
            metadata_columns (list): List of columns to include in metadata
            
        Returns:
            tuple: (chunks, metadatas) - Lists of text chunks and their corresponding metadata
        """
        if metadata_columns is None:
            metadata_columns = ['product', 'issue', 'company']
        
        try:
            logger.info(f"Processing dataframe with {len(df)} rows")
            
            all_chunks = []
            all_metadatas = []
            
            for i, row in df.iterrows():
                text = row[text_column]
                
                # Skip if text is missing
                if pd.isna(text) or text == "":
                    continue
                
                # Create chunks
                chunks = self.chunk_text(text)
                
                # Create metadata for each chunk
                metadata = {col: row[col] for col in metadata_columns if col in row}
                metadatas = [metadata.copy() for _ in chunks]
                
                # Add to lists
                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(df)} documents")
            return all_chunks, all_metadatas
        
        except Exception as e:
            logger.error(f"Error processing dataframe: {str(e)}")
            raise
    
    def create_vector_store(self, chunks: List[str], metadatas: List[Dict[str, Any]], 
                            vector_store_type: str = "FAISS", persist_directory: str = None) -> Any:
        """Create a vector store from chunks and metadata.
        
        Args:
            chunks (list): List of text chunks
            metadatas (list): List of metadata dictionaries
            vector_store_type (str): Type of vector store to create ("FAISS" or "ChromaDB")
            persist_directory (str): Directory to persist the vector store (required for ChromaDB)
            
        Returns:
            The created vector store
        """
        if not self.embeddings:
            logger.warning("Embeddings not initialized. Initializing components.")
            self._initialize_components()
        
        try:
            logger.info(f"Creating {vector_store_type} vector store with {len(chunks)} chunks")
            
            if vector_store_type == "FAISS":
                from langchain_community.vectorstores import FAISS
                
                # Create FAISS vector store
                vector_store = FAISS.from_texts(chunks, self.embeddings, metadatas=metadatas)
                
                # Save if persist_directory is provided
                if persist_directory:
                    persist_path = Path(persist_directory)
                    persist_path.mkdir(parents=True, exist_ok=True)
                    vector_store.save_local(str(persist_path))
                    logger.info(f"Saved FAISS vector store to {persist_directory}")
            
            elif vector_store_type == "ChromaDB":
                from langchain_community.vectorstores import Chroma
                
                # ChromaDB requires a persist_directory
                if not persist_directory:
                    logger.error("persist_directory is required for ChromaDB")
                    raise ValueError("persist_directory is required for ChromaDB")
                
                # Create ChromaDB vector store
                persist_path = Path(persist_directory)
                persist_path.mkdir(parents=True, exist_ok=True)
                
                vector_store = Chroma.from_texts(
                    texts=chunks,
                    embedding=self.embeddings,
                    metadatas=metadatas,
                    persist_directory=str(persist_path)
                )
                
                # Persist the vector store
                vector_store.persist()
                logger.info(f"Saved ChromaDB vector store to {persist_directory}")
            
            else:
                logger.error(f"Unsupported vector store type: {vector_store_type}")
                raise ValueError(f"Unsupported vector store type: {vector_store_type}")
            
            logger.info("Vector store created successfully")
            return vector_store
        
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    processed_data_path = Path("../../data/processed/processed_complaints.csv")
    vector_store_path = Path("../../vector_store")
    
    # Load processed data
    df = pd.read_csv(processed_data_path)
    print(f"Loaded {len(df)} processed records")
    
    # Create text processor
    processor = TextProcessor()
    
    # Process a sample of the data
    sample_size = min(1000, len(df))
    sample_df = df.sample(sample_size, random_state=42)
    
    # Process the sample
    chunks, metadatas = processor.process_dataframe(sample_df, text_column='cleaned_narrative')
    print(f"Created {len(chunks)} chunks from {sample_size} documents")
    
    # Create vector store
    vector_store = processor.create_vector_store(
        chunks=chunks,
        metadatas=metadatas,
        vector_store_type="FAISS",
        persist_directory=str(vector_store_path)
    )
    
    # Test a query
    query = "What are common issues with credit cards?"
    results = vector_store.similarity_search(query, k=3)
    
    print("\nSample query results:")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Product: {doc.metadata.get('product', 'Unknown')}")
        print(f"Issue: {doc.metadata.get('issue', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")