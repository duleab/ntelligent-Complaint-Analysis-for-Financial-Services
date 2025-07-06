# RAG Pipeline

This document provides detailed information about the Retrieval-Augmented Generation (RAG) pipeline in the Intelligent Complaint Analysis System.

## Overview

The RAG pipeline combines retrieval-based and generation-based approaches to provide accurate and contextually relevant answers to user queries. It consists of two main components:

1. **Retriever**: Retrieves relevant documents from the vector store based on the user's query
2. **Generator**: Generates a response based on the retrieved documents and the user's query

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  User Query     │────▶│    Retriever    │────▶│    Generator    │
│                 │     │                 │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └────────┬────────┘
                                 │                        │
                                 ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │                 │     │                 │
                        │  Vector Store   │     │    Response     │
                        │                 │     │                 │
                        │                 │     │                 │
                        └─────────────────┘     └─────────────────┘
```

## Components

### Retriever

The retriever component is responsible for retrieving relevant documents from the vector store based on the user's query.

**Implementation**: `src/rag/retriever.py`

```python
# Example usage
from src.rag.retriever import VectorStoreRetriever

retriever = VectorStoreRetriever(
    vector_store_path="vector_store",
    vector_store_type="faiss",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    top_k=5
)

documents = retriever.retrieve("How do I dispute a charge on my credit card?")
```

**Parameters**:
- `vector_store_path`: Path to the vector store
- `vector_store_type`: Type of vector store ("faiss" or "chroma")
- `embedding_model`: Name of the pre-trained model to use for embedding
- `top_k`: Number of documents to retrieve

**Methods**:
- `retrieve(query)`: Retrieves relevant documents based on the query
- `similarity_search(query, top_k)`: Performs a similarity search in the vector store
- `get_retriever()`: Returns the retriever object

### Generator

The generator component is responsible for generating a response based on the retrieved documents and the user's query.

**Implementation**: `src/rag/generator.py`

```python
# Example usage
from src.rag.generator import LLMGenerator

generator = LLMGenerator(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    max_tokens=500
)

response = generator.generate(
    query="How do I dispute a charge on my credit card?",
    context="Document 1: ... Document 2: ..."
)
```

**Parameters**:
- `model_name`: Name of the language model to use
- `temperature`: Temperature parameter for controlling randomness
- `max_tokens`: Maximum number of tokens to generate

**Methods**:
- `generate(query, context)`: Generates a response based on the query and context

### RAG Pipeline

The RAG pipeline orchestrates the retrieval and generation process.

**Implementation**: `src/rag/rag_pipeline.py`

```python
# Example usage
from src.rag.rag_pipeline import RAGPipeline
from src.rag.retriever import VectorStoreRetriever
from src.rag.generator import LLMGenerator

retriever = VectorStoreRetriever(
    vector_store_path="vector_store",
    vector_store_type="faiss",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    top_k=5
)

generator = LLMGenerator(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    max_tokens=500
)

pipeline = RAGPipeline(retriever=retriever, generator=generator)

response, documents = pipeline.run("How do I dispute a charge on my credit card?")
```

**Parameters**:
- `retriever`: Retriever component
- `generator`: Generator component

**Methods**:
- `run(query)`: Runs the RAG pipeline and returns the response and retrieved documents
- `run_with_feedback(query, feedback)`: Runs the RAG pipeline with user feedback
- `_refine_query(query, documents)`: Refines the query based on the retrieved documents

## Prompt Templates

The RAG pipeline uses prompt templates to structure the input to the language model. Here's an example of a prompt template used in the generator component:

```python
PROMPT_TEMPLATE = """
You are a helpful assistant for CrediTrust Financial, answering questions about customer complaints.

Context information is below.
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the question: {question}

If the answer cannot be determined from the context, say "I don't have enough information to answer this question."
"""
```

## Configuration

The RAG pipeline can be configured through various parameters:

### Retriever Configuration

- **Vector Store Type**: Type of vector store to use ("faiss" or "chroma")
- **Embedding Model**: Pre-trained model to use for embedding
- **Top K**: Number of documents to retrieve

### Generator Configuration

- **Model Name**: Name of the language model to use
- **Temperature**: Temperature parameter for controlling randomness
- **Max Tokens**: Maximum number of tokens to generate

## Usage Examples

### Basic Usage

```python
from src.rag.rag_pipeline import RAGPipeline
from src.rag.retriever import VectorStoreRetriever
from src.rag.generator import LLMGenerator

# Initialize the retriever
retriever = VectorStoreRetriever(
    vector_store_path="vector_store",
    vector_store_type="faiss",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    top_k=5
)

# Initialize the generator
generator = LLMGenerator(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    max_tokens=500
)

# Initialize the RAG pipeline
pipeline = RAGPipeline(retriever=retriever, generator=generator)

# Run the pipeline
response, documents = pipeline.run("How do I dispute a charge on my credit card?")

print("Response:", response)
print("\nRetrieved Documents:")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: {doc.page_content[:100]}...")
```

### Advanced Usage with Query Refinement

```python
from src.rag.rag_pipeline import RAGPipeline
from src.rag.retriever import VectorStoreRetriever
from src.rag.generator import LLMGenerator

# Initialize the retriever
retriever = VectorStoreRetriever(
    vector_store_path="vector_store",
    vector_store_type="faiss",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    top_k=5
)

# Initialize the generator
generator = LLMGenerator(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    max_tokens=500
)

# Initialize the RAG pipeline with query refinement
pipeline = RAGPipeline(retriever=retriever, generator=generator, refine_query=True)

# Run the pipeline
response, documents = pipeline.run("How do I dispute a charge on my credit card?")

print("Response:", response)
print("\nRetrieved Documents:")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: {doc.page_content[:100]}...")
```

## Evaluation

The RAG pipeline can be evaluated using various metrics:

### Retrieval Metrics

- **Precision@K**: Proportion of relevant documents among the top K retrieved documents
- **Recall@K**: Proportion of relevant documents retrieved among all relevant documents
- **Mean Reciprocal Rank (MRR)**: Average of the reciprocal ranks of the first relevant document

### Generation Metrics

- **BLEU**: Measures the similarity between the generated response and reference responses
- **ROUGE**: Measures the overlap between the generated response and reference responses
- **Human Evaluation**: Manual evaluation of the generated responses

## Best Practices

1. **Retriever Configuration**: Experiment with different embedding models and top K values to find the optimal configuration
2. **Generator Configuration**: Adjust the temperature and max tokens parameters based on the specific requirements
3. **Prompt Engineering**: Refine the prompt templates to improve the quality of the generated responses
4. **Query Refinement**: Enable query refinement for complex queries that may benefit from additional context
5. **Evaluation**: Regularly evaluate the RAG pipeline using appropriate metrics to ensure quality

## Troubleshooting

### Common Issues

1. **Irrelevant Documents**: If the retrieved documents are irrelevant, try adjusting the embedding model or top K value
2. **Low-Quality Responses**: If the generated responses are of low quality, try adjusting the prompt template or generator parameters
3. **Slow Performance**: If the pipeline is slow, consider optimizing the retrieval process or using a more efficient vector store

### Debugging Tips

1. **Logging**: Enable detailed logging to track the progress and identify issues
2. **Incremental Testing**: Test each component separately to isolate issues
3. **Validation**: Validate the retrieved documents and generated responses at each stage

## Future Improvements

1. **Multi-Stage Retrieval**: Implement a multi-stage retrieval process for improved accuracy
2. **Query Expansion**: Explore query expansion techniques to improve retrieval performance
3. **Hybrid Search**: Implement hybrid search combining dense and sparse retrieval
4. **Feedback Loop**: Implement a feedback loop for continuous improvement
5. **Model Fine-Tuning**: Fine-tune the language model on domain-specific data