# User Interface

This document provides detailed information about the user interface of the Intelligent Complaint Analysis System.

## Overview

The user interface is implemented using Streamlit, a Python library for creating web applications. It provides an interactive way for users to interact with the RAG system, ask questions about customer complaints, and view the results.

## Components

The user interface consists of several components:

### 1. Sidebar

The sidebar contains configuration options for the RAG system:

- **Retriever Type**: Type of vector store to use ("faiss" or "chroma")
- **Embedding Model**: Pre-trained model to use for embedding
- **Language Model**: Language model to use for generation
- **Number of Documents**: Number of documents to retrieve

### 2. Chat Interface

The chat interface allows users to interact with the RAG system:

- **Chat History**: Displays the conversation history
- **User Input**: Allows users to enter questions
- **Submit Button**: Submits the question to the RAG system
- **Clear Button**: Clears the conversation history

### 3. Data Insights

The data insights tab provides visualizations of the complaint data:

- **Product Distribution**: Distribution of complaints by product category
- **Issue Distribution**: Distribution of complaints by issue category
- **Product-Issue Heatmap**: Heatmap showing the relationship between products and issues
- **Time Series**: Trend of complaints over time
- **Narrative Length Distribution**: Distribution of complaint narrative lengths

### 4. System Information

The system information tab provides information about the RAG system:

- **Model Information**: Information about the embedding and language models
- **Vector Store Information**: Information about the vector store
- **System Performance**: Performance metrics of the RAG system

## Implementation

The user interface is implemented in two files:

1. `src/app.py`: Main application entry point
2. `src/streamlit_app.py`: Streamlit UI implementation

### Main Application Entry Point

```python
# src/app.py
import streamlit as st
from src.rag.retriever import VectorStoreRetriever
from src.rag.generator import LLMGenerator
from src.rag.rag_pipeline import RAGPipeline
from src.utils.visualization import (
    visualize_product_distribution,
    visualize_issue_distribution,
    visualize_product_issue_heatmap,
    visualize_time_series,
    visualize_narrative_length_distribution
)

# ... (rest of the code)
```

### Streamlit UI Implementation

```python
# src/streamlit_app.py
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# ... (rest of the code)
```

## Usage

### Starting the Application

To start the application, run the following command:

```bash
python run_app.py
```

This will start the Streamlit web interface, which can be accessed at http://localhost:8501.

### Interacting with the System

1. **Configure the System**:
   - Select the retriever type, embedding model, language model, and number of documents in the sidebar

2. **Ask Questions**:
   - Enter a question in the input field and click "Submit"
   - The system will retrieve relevant documents and generate a response
   - The response and retrieved documents will be displayed in the chat history

3. **View Data Insights**:
   - Click on the "Data Insights" tab to view visualizations of the complaint data

4. **View System Information**:
   - Click on the "System Information" tab to view information about the RAG system

## Customization

### Changing the Theme

The Streamlit theme can be customized by creating a `.streamlit/config.toml` file with the following content:

```toml
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Adding Custom CSS

Custom CSS can be added to the application using the `st.markdown` function:

```python
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}
.sidebar .sidebar-content {
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)
```

### Adding Custom JavaScript

Custom JavaScript can be added to the application using the `st.components.v1.html` function:

```python
st.components.v1.html("""
<script>
// Custom JavaScript code
</script>
""", height=0)
```

## Best Practices

1. **Responsive Design**: Ensure the UI is responsive and works well on different screen sizes
2. **Clear Instructions**: Provide clear instructions on how to use the system
3. **Error Handling**: Implement robust error handling to gracefully handle failures
4. **Loading States**: Show loading states during long-running operations
5. **Feedback Mechanism**: Provide a way for users to give feedback on the system

## Troubleshooting

### Common Issues

1. **Slow Loading**: If the UI is slow to load, check the size of the vector store and consider optimizing it
2. **Memory Issues**: If the application runs out of memory, consider reducing the number of documents or using a more efficient vector store
3. **UI Glitches**: If there are UI glitches, check the Streamlit version and update if needed

### Debugging Tips

1. **Logging**: Enable detailed logging to track the progress and identify issues
2. **Incremental Testing**: Test each component separately to isolate issues
3. **Browser Console**: Check the browser console for JavaScript errors

## Future Improvements

1. **User Authentication**: Implement user authentication for secure access
2. **Advanced Visualizations**: Add more advanced visualizations for better insights
3. **Export Functionality**: Allow users to export conversation history and insights
4. **Mobile App**: Develop a mobile app for on-the-go access
5. **Integration with Other Systems**: Integrate with other systems for seamless workflow

## Screenshots

### Chat Interface

```
+---------------------------------------------------------------+
|                                                               |
|  [Sidebar]                 [Chat Interface]                   |
|                                                               |
|  Retriever Type:           User: How do I dispute a charge?   |
|  [FAISS]                                                      |
|                            Assistant: To dispute a charge on  |
|  Embedding Model:          your credit card, you should:      |
|  [all-MiniLM-L6-v2]        1. Contact the merchant first     |
|                            2. If unresolved, contact your     |
|  Language Model:           credit card issuer                 |
|  [Mistral-7B]              3. Provide documentation           |
|                            4. Follow up as needed             |
|  Number of Documents:                                         |
|  [5]                       User:                              |
|                            [                    ] [Submit]    |
|                                                               |
|                            [Clear Chat]                       |
|                                                               |
+---------------------------------------------------------------+
```

### Data Insights

```
+---------------------------------------------------------------+
|                                                               |
|  [Sidebar]                 [Data Insights]                    |
|                                                               |
|  Retriever Type:           Product Distribution               |
|  [FAISS]                   [Bar Chart]                        |
|                                                               |
|  Embedding Model:          Issue Distribution                 |
|  [all-MiniLM-L6-v2]        [Bar Chart]                        |
|                                                               |
|  Language Model:           Product-Issue Heatmap              |
|  [Mistral-7B]              [Heatmap]                          |
|                                                               |
|  Number of Documents:      Time Series                        |
|  [5]                       [Line Chart]                       |
|                                                               |
|                            Narrative Length Distribution      |
|                            [Histogram]                        |
|                                                               |
+---------------------------------------------------------------+
```

### System Information

```
+---------------------------------------------------------------+
|                                                               |
|  [Sidebar]                 [System Information]               |
|                                                               |
|  Retriever Type:           Model Information                  |
|  [FAISS]                   Embedding Model: all-MiniLM-L6-v2  |
|                            Language Model: Mistral-7B         |
|  Embedding Model:                                             |
|  [all-MiniLM-L6-v2]        Vector Store Information           |
|                            Type: FAISS                        |
|  Language Model:           Size: 10000 documents              |
|  [Mistral-7B]                                                 |
|                            System Performance                 |
|  Number of Documents:      Average Response Time: 2.5s        |
|  [5]                       Memory Usage: 1.2GB                |
|                                                               |
|                                                               |
|                                                               |
+---------------------------------------------------------------+
```