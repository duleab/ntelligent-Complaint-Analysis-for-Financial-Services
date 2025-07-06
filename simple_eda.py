"""
Simple EDA Script for CFPB Complaints Data

This script performs basic exploration of the complaints data.
It's designed to be lightweight and handle large files efficiently.
"""

import os
import pandas as pd
from pathlib import Path

def main():
    print("=== Simple EDA for CFPB Complaints Data ===\n")
    
    # Set up paths
    data_dir = Path("Data") 
    data_file = data_dir / "complaints.csv" / "complaints.csv"
    
    print(f"Data file: {data_file.absolute()}")
    print(f"File exists: {data_file.exists()}")
    
    if not data_file.exists():
        print("Error: Data file not found!")
        return
    
    # Read just the first 10,000 rows for initial analysis
    print("\nReading first 10,000 rows...")
    try:
        # Read a sample of the data
        print("Reading sample data...")
        df = pd.read_csv(
            data_file,
            nrows=10000,  # Just read first 10,000 rows
            dtype={'ZIP code': str},
            parse_dates=['Date received', 'Date sent to company'],
            low_memory=False,
            encoding_errors='replace'
        )
        
        print(f"\nFound {len(df.columns)} columns: {', '.join(df.columns.tolist())}")
        
        # Basic info
        print(f"\nLoaded {len(df)} complaints")
        print("\nFirst few rows:")
        print(df.head())
        
        # Basic statistics
        print("\n=== Basic Statistics ===")
        print(f"Date range: {df['Date received'].min()} to {df['Date received'].max()}")
        
        # Product distribution
        print("\nComplaints by product:")
        print(df['Product'].value_counts())
        
        # Narrative length
        df['narrative_length'] = df['Consumer complaint narrative'].str.split().str.len()
        print(f"\nAverage narrative length: {df['narrative_length'].mean():.1f} words")
        
        # Save sample for further analysis
        output_file = data_dir / "sample_complaints.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved sample data to: {output_file}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
