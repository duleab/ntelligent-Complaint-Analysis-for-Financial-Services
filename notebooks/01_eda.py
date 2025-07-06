"""
Exploratory Data Analysis (EDA) for CFPB Complaints Data

This script performs initial exploration and cleaning of the CFPB complaints data.
It focuses on the five specified financial products and prepares the data for the RAG pipeline.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
DATA_DIR = Path("../Data")
PROCESSED_DIR = DATA_DIR / "processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Constants
TARGET_PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Payday loan",  # Included as it's often related to BNPL
    "Savings account",
    "Money transfer"
]

def load_data():
    """Load the complaints data from the CSV file in chunks."""
    print("Loading data...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Data directory: {DATA_DIR.absolute()}")
    
    # Check if the data file exists in the nested directory structure
    data_file = DATA_DIR / "complaints.csv" / "complaints.csv"
    print(f"Looking for data file at: {data_file.absolute()}")
    
    if not data_file.exists():
        print(f"Error: Could not find data file at {data_file.absolute()}")
        print("Available files in Data directory:")
        for f in DATA_DIR.glob('*'):
            if f.is_file():
                print(f"  - {f.name} (size: {os.path.getsize(f) / (1024*1024):.2f} MB)")
            else:
                print(f"  - {f.name}/ (directory)")
                # List contents of subdirectories
                for sub_f in f.glob('*'):
                    if sub_f.is_file():
                        print(f"    - {sub_f.name} (size: {os.path.getsize(sub_f) / (1024*1024):.2f} MB)")
                    else:
                        print(f"    - {sub_f.name}/ (directory)")
        exit(1)
    
    # Read the CSV file in chunks
    try:
        file_size = os.path.getsize(data_file) / (1024*1024)  # in MB
        print(f"Reading CSV file (size: {file_size:.2f} MB) in chunks...")
        
        # First, just get the column names
        with open(data_file, 'r', encoding='utf-8', errors='replace') as f:
            columns = next(pd.read_csv(f, nrows=0)).columns.tolist()
        print(f"Found columns: {', '.join(columns)}")
        
        # Now process the file in chunks
        chunk_size = 10000  # Number of rows per chunk
        chunks = []
        total_rows = 0
        
        # Create a chunked reader
        chunk_reader = pd.read_csv(
            data_file,
            chunksize=chunk_size,
            dtype={'ZIP code': str},  # Preserve leading zeros in ZIP codes
            parse_dates=['Date received', 'Date sent to company'],
            low_memory=False,
            on_bad_lines='warn',
            encoding_errors='replace'
        )
        
        # Process chunks
        for i, chunk in enumerate(chunk_reader, 1):
            chunks.append(chunk)
            total_rows += len(chunk)
            print(f"Processed chunk {i}: {total_rows:,} rows so far...")
            
            # For testing, limit to first few chunks
            if i >= 3:  # Remove this in production
                print("Limiting to first 3 chunks for testing")
                break
        
        # Combine chunks into a single DataFrame
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            print(f"Successfully loaded {len(df):,} complaints")
            return df
        else:
            print("No data was loaded")
            exit(1)
            
    except Exception as e:
        import traceback
        print(f"Error loading data: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        exit(1)

def filter_data(df):
    """Filter data for target products and non-empty narratives.
    
    Args:
        df: Raw complaints DataFrame
        
    Returns:
        Filtered DataFrame with only target products and non-empty narratives
    """
    print("\nFiltering data...")
    
    # Filter for target products (case insensitive)
    mask = df['Product'].str.lower().str.contains('|'.join(p.lower() for p in TARGET_PRODUCTS), na=False)
    filtered = df[mask].copy()
    print(f"Found {len(filtered):,} complaints for target products")
    
    # Filter for non-empty narratives
    filtered = filtered[filtered['Consumer complaint narrative'].notna() & 
                       (filtered['Consumer complaint narrative'].str.strip() != '')].copy()
    print(f"After removing empty narratives: {len(filtered):,} complaints")
    
    return filtered

def analyze_data(df):
    """Perform basic analysis and visualization of the data."""
    print("\nAnalyzing data...")
    
    # 1. Distribution of complaints by product
    plt.figure(figsize=(12, 6))
    product_counts = df['Product'].value_counts()
    sns.barplot(x=product_counts.values, y=product_counts.index)
    plt.title('Number of Complaints by Product')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / 'complaints_by_product.png')
    
    # 2. Narrative length analysis
    df['narrative_length'] = df['Consumer complaint narrative'].str.split().str.len()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['narrative_length'], bins=50)
    plt.title('Distribution of Narrative Lengths (Words)')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / 'narrative_length_distribution.png')
    
    # 3. Basic statistics
    stats = {
        'total_complaints': len(df),
        'products': df['Product'].nunique(),
        'avg_narrative_length': df['narrative_length'].mean(),
        'min_narrative_length': df['narrative_length'].min(),
        'max_narrative_length': df['narrative_length'].max(),
        'complaints_by_product': df['Product'].value_counts().to_dict()
    }
    
    return stats

def clean_text(text):
    """Basic text cleaning function.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove common prefixes/suffixes
    prefixes = [
        "i am writing to file a complaint",
        "this is a complaint about",
        "i am filing a complaint regarding"
    ]
    
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    return text.strip()

def main():
    # Load and filter data
    df = load_data()
    filtered_df = filter_data(df)
    
    # Analyze data
    stats = analyze_data(filtered_df)
    
    # Clean the narrative text
    print("\nCleaning narrative text...")
    filtered_df['cleaned_narrative'] = filtered_df['Consumer complaint narrative'].apply(clean_text)
    
    # Save processed data
    output_path = PROCESSED_DIR / 'filtered_complaints.csv'
    filtered_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(filtered_df):,} processed complaints to {output_path}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total complaints: {stats['total_complaints']:,}")
    print(f"Average narrative length: {stats['avg_narrative_length']:.1f} words")
    print("\nComplaints by product:")
    for product, count in stats['complaints_by_product'].items():
        print(f"  - {product}: {count:,}")

if __name__ == "__main__":
    main()
