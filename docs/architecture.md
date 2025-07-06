# System Architecture

This document provides an overview of the Intelligent Complaint Analysis System architecture, detailing the various components and how they interact with each other.

## High-Level Architecture

The system follows a modular architecture with the following main components:

1. **Data Processing Pipeline**: Responsible for downloading, cleaning, and processing the raw complaint data
2. **Vector Store**: Stores and indexes the embedded text chunks for efficient retrieval
3. **RAG Pipeline**: Implements the Retrieval-Augmented Generation logic
4. **User Interface**: Provides an interactive interface for users to interact with the system

## Architecture Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Raw Complaint  │────▶│  Data Processing│────▶│  Vector Store   │
│     Data        │     │     Pipeline    │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  User Interface │◀────│  RAG Pipeline   │◀────│    Retriever    │
│                 │     │                 │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │                 │
                        │    Generator    │
                        │                 │
                        │                 │
                        └─────────────────┘
```

## Component Details

### Data Processing Pipeline

The data processing pipeline is responsible for:

1. **Data Acquisition**: Downloading the raw complaint data from the CFPB database
2. **Data Cleaning**: Cleaning and preprocessing the raw data
3. **Text Chunking**: Splitting the complaint narratives into manageable chunks
4. **Embedding Generation**: Converting text chunks into vector embeddings

Key components:
- `src/data/download_data.py`: Downloads the raw data
- `src/data/data_processor.py`: Cleans and processes the data
- `src/data/text_processor.py`: Handles text chunking and embedding
- `src/data/create_vector_store.py`: Creates the vector store

### Vector Store

The vector store indexes and stores the embedded text chunks for efficient retrieval. The system supports two types of vector stores:

1. **FAISS**: Facebook AI Similarity Search, an efficient similarity search library
2. **ChromaDB**: A database for storing and querying embeddings

Key components:
- `vector_store/`: Directory containing the vector store files

### RAG Pipeline

The RAG (Retrieval-Augmented Generation) pipeline consists of two main components:

1. **Retriever**: Retrieves relevant documents from the vector store based on the user's query
2. **Generator**: Generates a response based on the retrieved documents and the user's query

Key components:
- `src/rag/retriever.py`: Implements the retrieval functionality
- `src/rag/generator.py`: Implements the generation functionality
- `src/rag/rag_pipeline.py`: Orchestrates the retrieval and generation process

### User Interface

The user interface provides an interactive way for users to interact with the system. It is implemented using Streamlit, a Python library for creating web applications.

Key components:
- `src/app.py`: Main application entry point
- `src/streamlit_app.py`: Streamlit UI implementation

## Data Flow

1. **User Input**: The user enters a query through the user interface
2. **Query Processing**: The query is processed and sent to the RAG pipeline
3. **Document Retrieval**: The retriever component retrieves relevant documents from the vector store
4. **Response Generation**: The generator component generates a response based on the retrieved documents and the user's query
5. **Response Display**: The response is displayed to the user through the user interface

## Technology Stack

- **Programming Language**: Python
- **Data Processing**: Pandas, NumPy
- **Embedding Models**: Sentence Transformers, Hugging Face Transformers
- **Vector Stores**: FAISS, ChromaDB
- **Language Models**: Hugging Face Models, OpenAI
- **Web Framework**: Streamlit
- **Testing**: Pytest
- **CI/CD**: GitHub Actions

## Deployment Architecture

The system is deployed using Streamlit Cloud, which provides a managed hosting environment for Streamlit applications. The deployment is automated through the CI/CD pipeline, which is implemented using GitHub Actions.

Key components:
- `.github/workflows/ci_cd.yml`: Defines the CI/CD pipeline

## Future Architecture Enhancements

- **Scalability**: Implement a more scalable architecture for handling larger datasets
- **Performance**: Optimize the retrieval and generation process for better performance
- **Security**: Enhance security measures for protecting sensitive data
- **Monitoring**: Implement monitoring and alerting for system health and performance
- **Feedback Loop**: Implement a feedback loop for continuous improvement of the system