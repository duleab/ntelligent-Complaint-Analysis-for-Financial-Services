import os
import sys
from pathlib import Path
import logging

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreRetriever:
    """Class for retrieving relevant documents from a vector store."""
    
    def __init__(self, vector_store_type="FAISS", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 vector_store_path=None, top_k=5):
        """
        Initialize the retriever with the specified vector store and embedding model.
        
        Args:
            vector_store_type (str): Type of vector store to use ("FAISS" or "ChromaDB")
            embedding_model_name (str): Name of the embedding model to use
            vector_store_path (Path): Path to the vector store
            top_k (int): Number of documents to retrieve
        """
        self.vector_store_type = vector_store_type
        self.embedding_model_name = embedding_model_name
        self.vector_store_path = vector_store_path
        self.top_k = top_k
        self.vector_store = None
        self.embeddings = None
        
        # Initialize the retriever
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize the vector store and embeddings."""
        try:
            # Import necessary modules
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            # Initialize embeddings
            logger.info(f"Initializing embeddings with model: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            
            # Load the vector store
            if self.vector_store_type == "FAISS":
                from langchain_community.vectorstores import FAISS
                
                if self.vector_store_path and self.vector_store_path.exists():
                    logger.info(f"Loading FAISS vector store from: {self.vector_store_path}")
                    self.vector_store = FAISS.load_local(
                        str(self.vector_store_path), 
                        self.embeddings, 
                        allow_dangerous_deserialization=True  # Allow deserialization of the FAISS index
                    )
                else:
                    logger.warning(f"Vector store path does not exist: {self.vector_store_path}")
                    self.vector_store = None
            
            elif self.vector_store_type == "ChromaDB":
                from langchain_community.vectorstores import Chroma
                
                if self.vector_store_path and self.vector_store_path.exists():
                    logger.info(f"Loading ChromaDB vector store from: {self.vector_store_path}")
                    self.vector_store = Chroma(persist_directory=str(self.vector_store_path), embedding_function=self.embeddings)
                else:
                    logger.warning(f"Vector store path does not exist: {self.vector_store_path}")
                    self.vector_store = None
            
            else:
                logger.error(f"Unsupported vector store type: {self.vector_store_type}")
                raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
            
            if self.vector_store:
                logger.info("Vector store loaded successfully")
        
        except Exception as e:
            logger.error(f"Error initializing retriever: {str(e)}")
            raise
    
    def retrieve(self, query):
        """Retrieve relevant documents for the given query.
        
        Args:
            query (str): The query to retrieve documents for
            
        Returns:
            list: List of retrieved documents
        """
        if not self.vector_store:
            logger.error("Vector store not initialized")
            raise ValueError("Vector store not initialized")
        
        try:
            logger.info(f"Retrieving documents for query: {query}")
            retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
            documents = retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
        
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    def similarity_search(self, query):
        """Perform a similarity search for the given query.
        
        Args:
            query (str): The query to search for
            
        Returns:
            list: List of (document, score) tuples
        """
        if not self.vector_store:
            logger.error("Vector store not initialized")
            raise ValueError("Vector store not initialized")
        
        try:
            logger.info(f"Performing similarity search for query: {query}")
            results = self.vector_store.similarity_search_with_score(query, k=self.top_k)
            logger.info(f"Found {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def get_retriever(self):
        """Get the retriever object.
        
        Returns:
            Retriever: The retriever object
        """
        if not self.vector_store:
            logger.error("Vector store not initialized")
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.as_retriever(search_kwargs={"k": self.top_k})