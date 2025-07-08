# CrediTrust Financial - Intelligent Complaint Analysis System

## Overview

This Streamlit application provides an interactive interface to query and analyze customer complaints from the Consumer Financial Protection Bureau (CFPB) database using a Retrieval-Augmented Generation (RAG) system. The application allows users to ask natural language questions about customer complaints across various financial products and receive AI-generated responses based on relevant complaint data.

## Features

- **Natural Language Querying**: Ask questions about customer complaints in plain English
- **Semantic Search**: Find relevant complaints based on meaning, not just keywords
- **Source Transparency**: View the actual complaint narratives that informed each answer
- **Configurable Model Settings**: Choose different embedding and language models
- **Interactive Chat Interface**: Maintain conversation history for context

## How to Use

1. **Initialize the RAG Pipeline**:
   - Select your preferred embedding model and language model in the sidebar
   - Click "Initialize RAG Pipeline" to set up the system

2. **Ask Questions**:
   - Type your question in the input field
   - Click "Submit" to get an AI-generated response
   - View source documents that informed the answer

3. **Sample Questions**:
   - Use the provided sample questions as inspiration
   - Try asking about specific financial products, issues, or trends

## Technical Details

The application uses:

- **LangChain**: For building the RAG pipeline
- **FAISS**: For efficient vector similarity search
- **HuggingFace Models**: For embeddings and text generation
- **Streamlit**: For the web interface

## Requirements

All dependencies are listed in the project's `requirements.txt` file. Make sure to install them before running the application.

## Running the Application

From the project root directory, run:

```bash
python run_app.py
```

Or directly with Streamlit:

```bash
streamlit run src/streamlit_app.py
```

The application will be available at http://localhost:8501 in your web browser.

## Troubleshooting

- **Vector Store Not Found**: Make sure you've created the vector store by running the data processing pipeline
- **Model Loading Errors**: Check your internet connection and HuggingFace token if using gated models
- **Slow Responses**: Consider using smaller models or reducing the number of retrieved documents

## Future Improvements

- Add data visualization for complaint trends
- Implement user feedback mechanism for responses
- Add support for filtering by date range or product type
- Improve response generation with more advanced prompting