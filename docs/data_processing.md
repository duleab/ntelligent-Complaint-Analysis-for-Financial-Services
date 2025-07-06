# Data Processing

This document provides detailed information about the data processing pipeline in the Intelligent Complaint Analysis System.

## Overview

The data processing pipeline is responsible for acquiring, cleaning, and preparing the complaint data for use in the RAG system. It includes several stages, from downloading the raw data to creating the vector store.

## Pipeline Stages

### 1. Data Acquisition

The first stage of the pipeline involves downloading the raw complaint data from the Consumer Financial Protection Bureau (CFPB) database.

**Implementation**: `src/data/download_data.py`

```python
# Example usage
from src.data.download_data import download_cfpb_data

download_cfpb_data(output_path="data/raw/complaints.csv", sample_size=10000)
```

**Parameters**:
- `output_path`: Path where the downloaded data will be saved
- `sample_size`: Number of complaints to sample (use `None` for all complaints)

### 2. Data Cleaning and Preprocessing

The second stage involves cleaning and preprocessing the raw data. This includes:

- Removing duplicate complaints
- Handling missing values
- Cleaning the narrative text
- Filtering complaints based on specific criteria

**Implementation**: `src/data/data_processor.py`

```python
# Example usage
from src.data.data_processor import DataProcessor

processor = DataProcessor(raw_data_path="data/raw/complaints.csv", 
                         processed_data_path="data/processed/cleaned_complaints.csv")
processor.load_data()
processor.clean_narratives()
processor.filter_data()
processor.save_data()
```

**Key Methods**:
- `load_data()`: Loads the raw data from the specified path
- `clean_narratives()`: Cleans the narrative text
- `filter_data()`: Filters complaints based on specific criteria
- `save_data()`: Saves the processed data to the specified path

### 3. Text Chunking

The third stage involves splitting the complaint narratives into manageable chunks for embedding and retrieval.

**Implementation**: `src/data/text_processor.py`

```python
# Example usage
from src.data.text_processor import chunk_text

chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
```

**Parameters**:
- `text`: The text to chunk
- `chunk_size`: The size of each chunk in characters
- `chunk_overlap`: The overlap between consecutive chunks in characters

### 4. Embedding Generation

The fourth stage involves converting the text chunks into vector embeddings using a pre-trained model.

**Implementation**: `src/data/text_processor.py`

```python
# Example usage
from src.data.text_processor import embed_text

embeddings = embed_text(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2")
```

**Parameters**:
- `chunks`: The text chunks to embed
- `model_name`: The name of the pre-trained model to use for embedding

### 5. Vector Store Creation

The final stage involves creating a vector store from the embeddings for efficient retrieval.

**Implementation**: `src/data/create_vector_store.py`

```python
# Example usage
from src.data.create_vector_store import create_vector_store

vector_store = create_vector_store(
    data_path="data/processed/cleaned_complaints.csv",
    vector_store_path="vector_store",
    vector_store_type="faiss",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
```

**Parameters**:
- `data_path`: Path to the processed data
- `vector_store_path`: Path where the vector store will be saved
- `vector_store_type`: Type of vector store to create ("faiss" or "chroma")
- `embedding_model`: Name of the pre-trained model to use for embedding

## Running the Complete Pipeline

The complete pipeline can be run using the `run_pipeline.py` script:

```bash
python run_pipeline.py --sample-size 10000 --vector-store-type faiss
```

**Parameters**:
- `--sample-size`: Number of complaints to sample (default: 10000)
- `--vector-store-type`: Type of vector store to create ("faiss" or "chroma", default: "faiss")

## Data Schema

### Raw Data Schema

The raw data from the CFPB database has the following schema:

| Column Name | Description |
|-------------|-------------|
| `complaint_id` | Unique identifier for the complaint |
| `date_received` | Date the complaint was received |
| `product` | Financial product category |
| `sub_product` | Specific product within the category |
| `issue` | Issue category |
| `sub_issue` | Specific issue within the category |
| `company` | Company the complaint is about |
| `state` | State where the complaint originated |
| `zip_code` | ZIP code where the complaint originated |
| `consumer_consent_provided` | Whether the consumer consented to publish the narrative |
| `submitted_via` | How the complaint was submitted |
| `date_sent_to_company` | Date the complaint was sent to the company |
| `company_response_to_consumer` | How the company responded to the consumer |
| `timely_response` | Whether the company responded in a timely manner |
| `consumer_disputed` | Whether the consumer disputed the company's response |
| `complaint_what_happened` | The narrative text of the complaint |

### Processed Data Schema

The processed data has the following schema:

| Column Name | Description |
|-------------|-------------|
| `complaint_id` | Unique identifier for the complaint |
| `date_received` | Date the complaint was received |
| `product` | Financial product category |
| `sub_product` | Specific product within the category |
| `issue` | Issue category |
| `sub_issue` | Specific issue within the category |
| `company` | Company the complaint is about |
| `state` | State where the complaint originated |
| `narrative` | The original narrative text of the complaint |
| `cleaned_narrative` | The cleaned narrative text |
| `chunk_id` | Identifier for the text chunk |
| `chunk_text` | The text chunk |
| `metadata` | Metadata associated with the chunk |

## Best Practices

1. **Sample Size**: Start with a small sample size for development and testing, then increase for production
2. **Chunk Size**: Experiment with different chunk sizes to find the optimal balance between context and retrieval efficiency
3. **Embedding Model**: Choose an embedding model that balances performance and resource requirements
4. **Vector Store Type**: Choose a vector store type based on your specific requirements (FAISS for speed, ChromaDB for additional features)
5. **Data Quality**: Regularly check the quality of the processed data to ensure it meets the requirements of the RAG system

## Troubleshooting

### Common Issues

1. **Missing Data**: If the raw data is missing important fields, check the download process and source
2. **Low-Quality Narratives**: If the narratives are of low quality, adjust the cleaning and filtering criteria
3. **Embedding Errors**: If there are errors during embedding, check the model compatibility and resource requirements
4. **Vector Store Creation Failures**: If the vector store creation fails, check the embedding format and storage requirements

### Debugging Tips

1. **Logging**: Enable detailed logging to track the progress and identify issues
2. **Incremental Processing**: Process the data in smaller batches to isolate issues
3. **Validation**: Validate the data at each stage of the pipeline to ensure quality
4. **Error Handling**: Implement robust error handling to gracefully handle failures

## Future Improvements

1. **Incremental Updates**: Implement incremental updates to the vector store
2. **Data Augmentation**: Explore data augmentation techniques to improve retrieval performance
3. **Advanced Filtering**: Implement more advanced filtering criteria based on specific requirements
4. **Parallel Processing**: Implement parallel processing for improved performance
5. **Data Versioning**: Implement data versioning for better tracking and reproducibility