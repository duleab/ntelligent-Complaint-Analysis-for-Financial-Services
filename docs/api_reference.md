# API Reference

This document provides a comprehensive reference for the APIs in the Intelligent Complaint Analysis System.

## Data Processing APIs

### `DataProcessor` Class

**File**: `src/data/data_processor.py`

**Description**: Handles data cleaning, filtering, and processing.

**Constructor**:
```python
DataProcessor(raw_data_path, processed_data_path)
```

**Parameters**:
- `raw_data_path` (str): Path to the raw data file
- `processed_data_path` (str): Path to save the processed data

**Methods**:

#### `load_data()`

**Description**: Loads the raw data from the specified path.

**Returns**: None

#### `clean_narratives()`

**Description**: Cleans the narrative text in the data.

**Returns**: None

#### `filter_data()`

**Description**: Filters the data based on specific criteria.

**Returns**: None

#### `save_data()`

**Description**: Saves the processed data to the specified path.

**Returns**: None

### `download_cfpb_data` Function

**File**: `src/data/download_data.py`

**Description**: Downloads complaint data from the CFPB database.

```python
download_cfpb_data(output_path, sample_size=None)
```

**Parameters**:
- `output_path` (str): Path to save the downloaded data
- `sample_size` (int, optional): Number of complaints to sample. If None, all complaints are downloaded.

**Returns**: None

### `create_vector_store` Function

**File**: `src/data/create_vector_store.py`

**Description**: Creates a vector store from the processed data.

```python
create_vector_store(data_path, vector_store_path, vector_store_type="faiss", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
```

**Parameters**:
- `data_path` (str): Path to the processed data
- `vector_store_path` (str): Path to save the vector store
- `vector_store_type` (str, optional): Type of vector store to create ("faiss" or "chroma")
- `embedding_model` (str, optional): Name of the pre-trained model to use for embedding

**Returns**: The created vector store

## RAG Pipeline APIs

### `VectorStoreRetriever` Class

**File**: `src/rag/retriever.py`

**Description**: Retrieves relevant documents from the vector store.

**Constructor**:
```python
VectorStoreRetriever(vector_store_path, vector_store_type="faiss", embedding_model="sentence-transformers/all-MiniLM-L6-v2", top_k=5)
```

**Parameters**:
- `vector_store_path` (str): Path to the vector store
- `vector_store_type` (str, optional): Type of vector store ("faiss" or "chroma")
- `embedding_model` (str, optional): Name of the pre-trained model to use for embedding
- `top_k` (int, optional): Number of documents to retrieve

**Methods**:

#### `retrieve(query)`

**Description**: Retrieves relevant documents based on the query.

**Parameters**:
- `query` (str): The query to search for

**Returns**: List of retrieved documents

#### `similarity_search(query, top_k=None)`

**Description**: Performs a similarity search in the vector store.

**Parameters**:
- `query` (str): The query to search for
- `top_k` (int, optional): Number of documents to retrieve. If None, uses the value from the constructor.

**Returns**: List of retrieved documents

#### `get_retriever()`

**Description**: Returns the retriever object.

**Returns**: The retriever object

### `LLMGenerator` Class

**File**: `src/rag/generator.py`

**Description**: Generates responses based on retrieved documents and user queries.

**Constructor**:
```python
LLMGenerator(model_name="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.7, max_tokens=500)
```

**Parameters**:
- `model_name` (str, optional): Name of the language model to use
- `temperature` (float, optional): Temperature parameter for controlling randomness
- `max_tokens` (int, optional): Maximum number of tokens to generate

**Methods**:

#### `generate(query, context)`

**Description**: Generates a response based on the query and context.

**Parameters**:
- `query` (str): The user's query
- `context` (str): The context from retrieved documents

**Returns**: The generated response

### `RAGPipeline` Class

**File**: `src/rag/rag_pipeline.py`

**Description**: Orchestrates the retrieval and generation process.

**Constructor**:
```python
RAGPipeline(retriever, generator, refine_query=False)
```

**Parameters**:
- `retriever` (VectorStoreRetriever): The retriever component
- `generator` (LLMGenerator): The generator component
- `refine_query` (bool, optional): Whether to refine the query based on retrieved documents

**Methods**:

#### `run(query)`

**Description**: Runs the RAG pipeline and returns the response and retrieved documents.

**Parameters**:
- `query` (str): The user's query

**Returns**: Tuple of (response, documents)

#### `run_with_feedback(query, feedback)`

**Description**: Runs the RAG pipeline with user feedback.

**Parameters**:
- `query` (str): The user's query
- `feedback` (str): User feedback on previous response

**Returns**: Tuple of (response, documents)

#### `_refine_query(query, documents)`

**Description**: Refines the query based on the retrieved documents.

**Parameters**:
- `query` (str): The original query
- `documents` (list): The retrieved documents

**Returns**: The refined query

## Visualization APIs

### `visualize_product_distribution` Function

**File**: `src/utils/visualization.py`

**Description**: Visualizes the distribution of complaints by product category.

```python
visualize_product_distribution(data, top_n=10)
```

**Parameters**:
- `data` (pandas.DataFrame): The complaint data
- `top_n` (int, optional): Number of top products to show

**Returns**: matplotlib.pyplot.Figure

### `visualize_issue_distribution` Function

**File**: `src/utils/visualization.py`

**Description**: Visualizes the distribution of complaints by issue category.

```python
visualize_issue_distribution(data, top_n=10)
```

**Parameters**:
- `data` (pandas.DataFrame): The complaint data
- `top_n` (int, optional): Number of top issues to show

**Returns**: matplotlib.pyplot.Figure

### `visualize_product_issue_heatmap` Function

**File**: `src/utils/visualization.py`

**Description**: Visualizes the relationship between products and issues.

```python
visualize_product_issue_heatmap(data, top_products=5, top_issues=5)
```

**Parameters**:
- `data` (pandas.DataFrame): The complaint data
- `top_products` (int, optional): Number of top products to show
- `top_issues` (int, optional): Number of top issues to show

**Returns**: matplotlib.pyplot.Figure

### `visualize_time_series` Function

**File**: `src/utils/visualization.py`

**Description**: Visualizes the trend of complaints over time.

```python
visualize_time_series(data, time_column="date_received", resample_freq="M")
```

**Parameters**:
- `data` (pandas.DataFrame): The complaint data
- `time_column` (str, optional): Name of the time column
- `resample_freq` (str, optional): Frequency for resampling ("D" for daily, "W" for weekly, "M" for monthly, etc.)

**Returns**: matplotlib.pyplot.Figure

### `visualize_narrative_length_distribution` Function

**File**: `src/utils/visualization.py`

**Description**: Visualizes the distribution of complaint narrative lengths.

```python
visualize_narrative_length_distribution(data, narrative_column="narrative")
```

**Parameters**:
- `data` (pandas.DataFrame): The complaint data
- `narrative_column` (str, optional): Name of the narrative column

**Returns**: matplotlib.pyplot.Figure

## Web Application APIs

### Streamlit App

**File**: `src/app.py`

**Description**: Main application entry point for the Streamlit web interface.

**Functions**:

#### `initialize_session_state()`

**Description**: Initializes the Streamlit session state.

**Returns**: None

#### `initialize_rag_pipeline(retriever_type, embedding_model, language_model, top_k)`

**Description**: Initializes the RAG pipeline with the specified configuration.

**Parameters**:
- `retriever_type` (str): Type of vector store to use ("faiss" or "chroma")
- `embedding_model` (str): Name of the pre-trained model to use for embedding
- `language_model` (str): Name of the language model to use
- `top_k` (int): Number of documents to retrieve

**Returns**: The initialized RAG pipeline

#### `display_chat_history()`

**Description**: Displays the chat history in the Streamlit interface.

**Returns**: None

#### `process_user_input(user_input)`

**Description**: Processes the user's input and generates a response.

**Parameters**:
- `user_input` (str): The user's input

**Returns**: None

#### `display_data_insights(data)`

**Description**: Displays data insights in the Streamlit interface.

**Parameters**:
- `data` (pandas.DataFrame): The complaint data

**Returns**: None

#### `display_system_information()`

**Description**: Displays system information in the Streamlit interface.

**Returns**: None

## Utility APIs

### `chunk_text` Function

**File**: `src/data/text_processor.py`

**Description**: Splits text into manageable chunks.

```python
chunk_text(text, chunk_size=500, chunk_overlap=50)
```

**Parameters**:
- `text` (str): The text to chunk
- `chunk_size` (int, optional): The size of each chunk in characters
- `chunk_overlap` (int, optional): The overlap between consecutive chunks in characters

**Returns**: List of text chunks

### `embed_text` Function

**File**: `src/data/text_processor.py`

**Description**: Converts text chunks into vector embeddings.

```python
embed_text(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2")
```

**Parameters**:
- `chunks` (list): The text chunks to embed
- `model_name` (str, optional): The name of the pre-trained model to use for embedding

**Returns**: List of embeddings

## Command-Line Interfaces

### `run_pipeline.py`

**Description**: Script to run the complete data processing pipeline.

**Usage**:
```bash
python run_pipeline.py --sample-size 10000 --vector-store-type faiss
```

**Arguments**:
- `--sample-size` (int, optional): Number of complaints to sample (default: 10000)
- `--vector-store-type` (str, optional): Type of vector store to create ("faiss" or "chroma", default: "faiss")

### `run_app.py`

**Description**: Script to run the Streamlit application.

**Usage**:
```bash
python run_app.py
```

### `run_tests.py`

**Description**: Script to run all tests.

**Usage**:
```bash
python run_tests.py
```