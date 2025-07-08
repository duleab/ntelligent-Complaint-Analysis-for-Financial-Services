import streamlit as st
import pandas as pd
import os
import sys
import time
from pathlib import Path
from langchain.llms.fake import FakeListLLM  # Import FakeListLLM for testing

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import RAG components
from src.rag.retriever import VectorStoreRetriever
from src.rag.generator import LLMGenerator
from src.rag.rag_pipeline import RAGPipeline
from src.utils.visualization import visualize_product_distribution, visualize_issue_distribution

# Page configuration
st.set_page_config(
    page_title="CrediTrust Financial - Complaint Analysis",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #4B5563;
    }
    .highlight {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #E0F2FE;
        border-left: 4px solid #0EA5E9;
    }
    .assistant-message {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
    }
    .source-document {
        background-color: #F9FAFB;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        margin-top: 0.5rem;
        border: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Sidebar
with st.sidebar:
    st.markdown('<div class="sub-header">CrediTrust Financial</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-text">Intelligent Complaint Analysis System</div>', unsafe_allow_html=True)
    st.divider()
    
    # Model settings
    st.markdown("### Model Settings")
    
    retriever_type = st.selectbox(
        "Retriever Type",
        options=["FAISS", "ChromaDB"],
        index=0
    )
    
    embedding_model = st.selectbox(
        "Embedding Model",
        options=["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
        index=0
    )
    
    llm_model = st.selectbox(
        "Language Model",
        options=["mistralai/Mistral-7B-Instruct-v0.2", "google/flan-t5-base", "meta-llama/Llama-2-7b-chat-hf"],
        index=0
    )
    
    top_k = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=5)
    
    # Initialize or reset RAG pipeline
    if st.button("Initialize/Reset RAG Pipeline"):
        with st.spinner("Initializing RAG pipeline..."):
            # Load vector store
            vector_store_path = Path("vector_store")
            retriever = VectorStoreRetriever(
                vector_store_type=retriever_type,
                embedding_model_name=embedding_model,
                vector_store_path=vector_store_path,
                top_k=top_k
            )
            
            # Initialize LLM
            generator = LLMGenerator(
                model_name=llm_model,
                temperature=0.7,
                max_tokens=512
            )
            
            # Create RAG pipeline
            st.session_state.rag_pipeline = RAGPipeline(
                retriever=retriever,
                generator=generator
            )
            
            # Load processed data for visualization
            try:
                st.session_state.processed_data = pd.read_csv("../data/processed/processed_complaints.csv")
                st.success("RAG pipeline initialized successfully!")
            except Exception as e:
                st.error(f"Error loading processed data: {str(e)}")
    
    st.divider()
    
    # About section
    st.markdown("### About")
    st.markdown("""
    This application uses Retrieval-Augmented Generation (RAG) to analyze customer complaints 
    from the Consumer Financial Protection Bureau (CFPB) database.
    
    **Features:**
    - Natural language querying of complaint data
    - Semantic search across multiple financial products
    - AI-powered insights and trend identification
    - Source document transparency
    """)

# Main content
st.markdown('<div class="main-header">Intelligent Complaint Analysis System</div>', unsafe_allow_html=True)

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Chat Interface", "Data Insights", "System Information"])

# Tab 1: Chat Interface
with tab1:
    st.markdown('<div class="sub-header">Ask questions about customer complaints</div>', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            
            # Display source documents if available
            if "sources" in message and message["sources"]:
                st.markdown("<strong>Sources:</strong>", unsafe_allow_html=True)
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"""
                    <div class="source-document">
                        <strong>Source {i+1}:</strong> {source['product']} - {source['issue']}<br>
                        <em>{source['content'][:200]}...</em>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Input for new question
    user_question = st.text_input("Ask a question about customer complaints:", key="user_question")
    
    # Process the question
    if user_question and st.button("Submit"):
        if st.session_state.rag_pipeline is None:
            st.warning("Please initialize the RAG pipeline first!")
        else:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Get response from RAG pipeline
            with st.spinner("Generating response..."):
                try:
                    result = st.session_state.rag_pipeline.run(user_question)
                    
                    # Extract sources for display
                    sources = []
                    for doc in result["source_documents"]:
                        sources.append({
                            "product": doc.metadata.get("product", "Unknown"),
                            "issue": doc.metadata.get("issue", "Unknown"),
                            "content": doc.page_content
                        })
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": result["answer"],
                        "sources": sources
                    })
                    
                    # Rerun to update the display
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

# Tab 2: Data Insights
with tab2:
    st.markdown('<div class="sub-header">Complaint Data Insights</div>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is not None:
        # Create two columns for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Product Distribution")
            fig1 = visualize_product_distribution(st.session_state.processed_data)
            st.pyplot(fig1)
        
        with col2:
            st.markdown("### Issue Distribution")
            fig2 = visualize_issue_distribution(st.session_state.processed_data)
            st.pyplot(fig2)
        
        # Show sample complaints
        st.markdown("### Sample Complaints")
        sample_size = min(5, len(st.session_state.processed_data))
        samples = st.session_state.processed_data.sample(sample_size)
        
        for i, (_, row) in enumerate(samples.iterrows()):
            with st.expander(f"Complaint {i+1}: {row['product']} - {row['issue']}"):
                st.markdown(f"**Product:** {row['product']}")
                st.markdown(f"**Issue:** {row['issue']}")
                st.markdown(f"**Company:** {row['company']}")
                st.markdown(f"**Narrative:**")
                st.markdown(f"<div class='highlight'>{row['narrative'][:500]}...</div>", unsafe_allow_html=True)
    else:
        st.info("Please initialize the RAG pipeline to load the processed data.")

# Tab 3: System Information
with tab3:
    st.markdown('<div class="sub-header">System Information</div>', unsafe_allow_html=True)
    
    st.markdown("### RAG Pipeline Architecture")
    st.markdown("""
    <div class="highlight">
    The RAG (Retrieval-Augmented Generation) pipeline consists of three main components:
    
    1. **Retriever**: Searches the vector database to find relevant complaint narratives based on the user's question.
    2. **Generator**: Uses a language model to generate a comprehensive answer based on the retrieved documents.
    3. **Pipeline**: Coordinates the retrieval and generation processes, ensuring a smooth flow of information.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Data Processing Workflow")
    st.markdown("""
    <div class="highlight">
    The complaint data undergoes several processing steps:
    
    1. **Filtering**: Select relevant financial products and complaints with narratives.
    2. **Cleaning**: Remove special characters, extra whitespace, and boilerplate text.
    3. **Chunking**: Split long narratives into smaller, semantically meaningful chunks.
    4. **Embedding**: Convert text chunks into vector representations using a pre-trained model.
    5. **Indexing**: Store the vectors in a searchable database for efficient retrieval.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### System Requirements")
    st.markdown("""
    - Python 3.8+
    - Required libraries: streamlit, pandas, numpy, matplotlib, seaborn, langchain, transformers, faiss-cpu/chromadb
    - Recommended: GPU for faster inference with larger language models
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
    CrediTrust Financial - Intelligent Complaint Analysis System | Developed for 10 Academy Week 6 Project
</div>
""", unsafe_allow_html=True)