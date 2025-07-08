import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import RAG components
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.llms.fake import FakeListLLM  # Import FakeListLLM for testing

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

# Sidebar
with st.sidebar:
    st.markdown('<div class="sub-header">CrediTrust Financial</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-text">Intelligent Complaint Analysis System</div>', unsafe_allow_html=True)
    st.divider()
    
    # Model settings
    st.markdown("### Model Settings")
    
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
    if st.button("Initialize RAG Pipeline"):
        with st.spinner("Initializing RAG pipeline..."):
            try:
                # Initialize the embedding model
                embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
                
                # Load the vector store
                vector_store_dir = "vector_store"
                vectorstore = FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)
                
                # Define the prompt template
                template = """
                You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints across different financial products.
                
                Use ONLY the following retrieved complaint excerpts to formulate your answer:
                
                {context}
                
                Question: {question}
                
                Instructions:
                1. Base your answer ONLY on the provided context.
                2. If the context doesn't contain enough information to answer the question, state that clearly.
                3. Be concise but comprehensive in your answer.
                4. Highlight patterns or trends if they are apparent in the context.
                5. Do not make up information that is not in the context.
                
                Answer: 
                """
                
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["context", "question"]
                )
                
                # Initialize a fake LLM for testing (instead of HuggingFaceHub)
                fake_responses = [
                    "Based on the retrieved complaints, customers are primarily concerned with unauthorized charges and billing disputes related to credit cards.",
                    "The complaints show that customers are unhappy with BNPL services due to hidden fees and unclear terms.",
                    "According to the complaints, money transfer issues include delayed processing times and unexpected fees.",
                    "The data indicates that savings account complaints focus on interest rate discrepancies and account maintenance fees.",
                    "Personal loan complaints frequently mention high interest rates and unexpected fees."
                ]
                
                llm = FakeListLLM(responses=fake_responses)
                
                # Create the retrieval QA chain
                st.session_state.rag_pipeline = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True
                )
                
                st.success("RAG pipeline initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing RAG pipeline: {str(e)}")
    
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
                result = st.session_state.rag_pipeline({"query": user_question})
                
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
                    "content": result["result"],
                    "sources": sources
                })
                
                # Rerun to update the display
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Sample questions
st.markdown("### Sample Questions")
st.markdown("""
<div class="highlight">
<strong>Try asking questions like:</strong>
<ul>
<li>What are the most common issues with credit cards?</li>
<li>Why are customers unhappy with Buy Now, Pay Later services?</li>
<li>What problems do customers face with money transfers?</li>
<li>Are there issues with savings account interest rates?</li>
<li>What complaints do customers have about personal loans?</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
    CrediTrust Financial - Intelligent Complaint Analysis System | Developed for 10 Academy Week 6 Project
</div>
""", unsafe_allow_html=True)