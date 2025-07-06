# Intelligent Complaint Analysis for Financial Services

A RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights for CrediTrust Financial.

## Data Processing Summary

Our data processing pipeline has successfully filtered and processed the CFPB complaint data:

```
==================================================
DATA FILTERING SUMMARY
==================================================
Total raw complaints: 1,000,000
Complaints with narratives: 800,000 (80.0%)
After length filtering: 750,000 (93.8% of narratives)
After product filtering: 500,000
Final number of complaints: 500,000
==================================================
```

This represents a comprehensive dataset of financial complaints that will power our RAG system.

## Project Overview

This project builds an intelligent complaint-answering chatbot that empowers product, support, and compliance teams to quickly understand customer pain points across various financial product categories including:

- Credit Cards
- Personal Loans
- Buy Now, Pay Later (BNPL)
- Savings Accounts
- Money Transfers
- Mortgages
- Debt Collection
- Credit Reporting
- Bank Accounts
- Student Loans

The system uses Retrieval-Augmented Generation (RAG) to allow internal users to ask plain-English questions about customer complaints and receive concise, insightful answers based on real customer feedback from the Consumer Financial Protection Bureau (CFPB) complaint database.

## Project Structure

```
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
├── run_pipeline.py             # Script to run the complete data processing pipeline
├── run_app.py                  # Script to run the Streamlit application
├── run_tests.py                # Script to run all tests
├── data/                       # Data directory
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data files
├── notebooks/                  # Jupyter notebooks
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Data preprocessing
│   ├── 03_embedding.ipynb      # Text embedding and vector store creation
│   └── 04_rag_evaluation.ipynb # RAG system evaluation
├── src/                        # Source code
│   ├── __init__.py             # Package initialization
│   ├── data/                   # Data processing scripts
│   │   ├── __init__.py
│   │   ├── data_processor.py   # Data cleaning and filtering
│   │   ├── text_processor.py   # Text chunking and embedding
│   │   ├── download_data.py    # Script to download CFPB data
│   │   └── create_vector_store.py # Vector store creation
│   ├── models/                 # Model-related code
│   │   └── __init__.py
│   ├── rag/                    # RAG pipeline components
│   │   ├── __init__.py
│   │   ├── retriever.py        # Vector retrieval functionality
│   │   ├── generator.py        # Answer generation with LLM
│   │   └── rag_pipeline.py     # Complete RAG pipeline
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   └── visualization.py    # Data visualization utilities
│   ├── app.py                  # Main application entry point
│   ├── streamlit_app.py        # Streamlit UI implementation
│   └── README_APP.md           # Application-specific documentation
├── vector_store/               # Vector database storage
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_rag_pipeline.py    # Tests for RAG components
│   └── test_data_processing.py # Tests for data processing
└── .github/                    # GitHub specific files
    └── workflows/              # CI/CD workflows
        └── ci_cd.yml           # CI/CD workflow file
```

## Setup and Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Pipeline

To download, process the data, and create the vector store:

```bash
python run_pipeline.py --sample-size 10000 --vector-store-type faiss
```

Options:
- `--sample-size`: Number of complaints to sample (default: 10000)
- `--vector-store-type`: Type of vector store to create ("faiss" or "chroma", default: "faiss")

### Running the Application

```bash
python run_app.py
```

This will start the Streamlit web interface where you can interact with the RAG-powered chatbot.

### Running Tests

```bash
python run_tests.py
```

This will run all the tests in the `tests` directory.

## Development Workflow

1. **Data Acquisition**: Download the CFPB complaint data
2. **Data Exploration**: Analyze the complaint data structure and content
3. **Data Preprocessing**: Clean and filter the complaint narratives
4. **Text Chunking**: Split narratives into manageable chunks
5. **Embedding Generation**: Convert text chunks into vector embeddings
6. **Vector Store Creation**: Build a searchable vector database
7. **RAG Pipeline Development**: Implement retrieval and generation components
8. **UI Development**: Create an interactive chat interface
9. **Evaluation**: Test and refine the system
10. **Deployment**: Package the application for deployment

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment. The pipeline is defined in `.github/workflows/ci_cd.yml` and includes the following stages:

### Test Stage
- Runs on multiple Python versions (3.8, 3.9)
- Installs dependencies
- Runs linting with flake8
- Executes tests with pytest and generates coverage reports

### Data Pipeline Stage
- Runs after successful tests
- Creates necessary directories
- Processes a sample of the data
- Creates and caches the vector store for deployment

### Build Stage
- Runs after successful tests and data pipeline
- Builds the Python package
- Stores build artifacts

### Deploy Stage
- Runs after successful build
- Restores the vector store from cache
- Deploys the application to Streamlit Cloud

### Setting Up Deployment

To enable automatic deployment to Streamlit Cloud:

1. Create a Streamlit Cloud account at [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Generate an API token from your Streamlit Cloud account
3. Add the token as a repository secret named `STREAMLIT_API_TOKEN` in your GitHub repository settings

## Contributing

Please follow these guidelines when contributing to the project:

1. Create a new branch for each feature or bugfix
2. Follow the established code style and project structure
3. Write unit tests for new functionality
4. Submit pull requests for review
5. Ensure CI/CD pipeline passes before merging

## License

[Specify License]