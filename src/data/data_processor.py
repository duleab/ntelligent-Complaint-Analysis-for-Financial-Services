import pandas as pd
import numpy as np
import re
import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple, Optional, Union

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class for processing complaint data for the RAG pipeline."""
    
    def __init__(self, raw_data_path: Union[str, Path], processed_data_path: Union[str, Path]):
        """
        Initialize the data processor.
        
        Args:
            raw_data_path (str or Path): Path to the raw data file
            processed_data_path (str or Path): Path to save the processed data
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.data = None
        
        # Ensure directories exist
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Load the raw complaint data.
        
        Returns:
            pd.DataFrame: The loaded data
        """
        try:
            logger.info(f"Loading data from {self.raw_data_path}")
            # First read just the column names
            columns = pd.read_csv(self.raw_data_path, nrows=0).columns.tolist()
            
            # Read the data in chunks to handle large files
            chunks = []
            chunk_size = 100000  # Process 100k rows at a time
            
            for chunk in pd.read_csv(self.raw_data_path, chunksize=chunk_size, low_memory=False, usecols=columns):
                chunks.append(chunk)
                logger.info(f"Loaded {sum(len(c) for c in chunks)} records...")
            
            self.data = pd.concat(chunks, ignore_index=True)
            logger.info(f"Successfully loaded {len(self.data)} total records")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def filter_data(self, products: List[str] = None, min_narrative_length: int = 50) -> pd.DataFrame:
        """Filter the complaint data.
        
        Args:
            products: List of product types to include. If None, all products are included.
            min_narrative_length: Minimum length of narrative text to include.
            
        Returns:
            Filtered DataFrame with statistics
        """
        if self.data is None:
            self.load_data()
            
        try:
            # Track statistics
            stats = {
                'total_raw_complaints': len(self.data),
                'with_narratives': 0,
                'after_length_filter': 0,
                'after_product_filter': 0,
                'final_count': 0
            }
            
            logger.info(f"Total raw complaints: {stats['total_raw_complaints']:,}")
            
            # Find narrative column (case insensitive)
            narrative_col = next((col for col in self.data.columns if 'narrative' in col.lower()), None)
            if narrative_col is None:
                logger.error("No narrative column found in the data")
                return pd.DataFrame()
                
            # First, check if we have any non-empty narratives
            has_narratives = self.data[narrative_col].notna() & (self.data[narrative_col].astype(str).str.strip() != '')
            filtered_data = self.data[has_narratives].copy()
            stats['with_narratives'] = len(filtered_data)
            
            logger.info(f"Complaints with narratives: {stats['with_narratives']:,} "
                       f"({stats['with_narratives']/stats['total_raw_complaints']*100:.1f}%)")
            
            # If no narratives found, try alternative column names
            if stats['with_narratives'] == 0:
                logger.warning("No narratives found in the 'Consumer complaint narrative' column. Trying alternative columns...")
                
                # Try common alternative column names
                alt_narrative_cols = [col for col in self.data.columns 
                                   if any(x in col.lower() for x in ['narrative', 'description', 'complaint'])]
                logger.info(f"Trying alternative columns: {alt_narrative_cols}")
                
                for col in alt_narrative_cols:
                    if col != narrative_col:  # Skip the column we already tried
                        has_content = self.data[col].notna() & (self.data[col].astype(str).str.strip() != '')
                        if has_content.any():
                            filtered_data = self.data[has_content].copy()
                            narrative_col = col
                            stats['with_narratives'] = len(filtered_data)
                            logger.info(f"Found {stats['with_narratives']} records with content in column: {col}")
                            break
            
            # If we still have data, process it
            if stats['with_narratives'] > 0:
                # Filter for minimum narrative length
                filtered_data['narrative_length'] = filtered_data[narrative_col].astype(str).str.len()
                before_length_filter = len(filtered_data)
                filtered_data = filtered_data[filtered_data['narrative_length'] >= min_narrative_length]
                stats['after_length_filter'] = len(filtered_data)
                
                logger.info(f"Complaints after length filtering (>{min_narrative_length} chars): "
                          f"{stats['after_length_filter']:,} "
                          f"({stats['after_length_filter']/stats['with_narratives']*100:.1f}% of narratives)")
                
                # If no data after length filtering, show some statistics
                if stats['after_length_filter'] == 0:
                    logger.warning("No narratives meet the minimum length requirement. Showing length statistics:")
                    length_stats = filtered_data['narrative_length'].describe()
                    logger.info(f"Narrative lengths: {length_stats}")
                    
                    # Try with a lower minimum length
                    min_narrative_length = 10  # Lower the threshold
                    filtered_data = filtered_data[filtered_data['narrative_length'] >= min_narrative_length]
                    stats['after_length_filter'] = len(filtered_data)
                    logger.info(f"Complaints with narratives > {min_narrative_length} chars: {stats['after_length_filter']:,}")
            
            # Filter for specific products if provided and we have data
            if products and stats['after_length_filter'] > 0:
                product_col = next((col for col in filtered_data.columns if 'product' in col.lower()), None)
                if product_col:
                    logger.info(f"Filtering for products: {products}")
                    before_count = len(filtered_data)
                    filtered_data = filtered_data[filtered_data[product_col].str.lower().isin([p.lower() for p in products])]
                    stats['after_product_filter'] = len(filtered_data)
                    logger.info(f"Complaints after product filtering: {stats['after_product_filter']:,} "
                              f"(removed {before_count - stats['after_product_filter']:,})")
            
            # Rename columns to standard names if needed
            rename_dict = {
                'Complaint ID': 'complaint_id',
                'Consumer complaint narrative': 'narrative',
                'Product': 'product',
                'Sub-product': 'sub_product',
                'Issue': 'issue',
                'Sub-issue': 'sub_issue',
                'Company': 'company',
                'State': 'state',
                'ZIP code': 'zip_code',
                'Date received': 'date_received',
                'Date sent to company': 'date_sent_to_company',
                'Company response to consumer': 'company_response',
                'Timely response?': 'timely_response',
                'Consumer disputed?': 'consumer_disputed',
                'Consumer consent provided?': 'consumer_consent_provided',
                'Submitted via': 'submitted_via',
                'Company public response': 'company_public_response',
                'Tags': 'tags'
            }
            
            # Only include columns that exist in the data
            rename_dict = {k: v for k, v in rename_dict.items() if k in filtered_data.columns}
            filtered_data = filtered_data.rename(columns=rename_dict)
            
            # Add complaint_id if not present
            if 'complaint_id' not in filtered_data.columns and 'Complaint ID' in filtered_data.columns:
                filtered_data = filtered_data.rename(columns={'Complaint ID': 'complaint_id'})
            
            # Ensure we have the required columns
            required_columns = ['complaint_id', 'narrative']
            for col in required_columns:
                if col not in filtered_data.columns:
                    logger.warning(f"Required column '{col}' not found in the data")
                    return pd.DataFrame()
            
            stats['final_count'] = len(filtered_data)
            
            # Log final statistics
            logger.info("\n" + "="*50)
            logger.info("DATA FILTERING SUMMARY")
            logger.info("="*50)
            logger.info(f"Total raw complaints: {stats['total_raw_complaints']:,}")
            logger.info(f"Complaints with narratives: {stats['with_narratives']:,} "
                      f"({stats['with_narratives']/stats['total_raw_complaints']*100:.1f}%)")
            logger.info(f"After length filtering: {stats['after_length_filter']:,} "
                      f"({stats['after_length_filter']/stats['with_narratives']*100:.1f}% of narratives)")
            if 'after_product_filter' in stats:
                logger.info(f"After product filtering: {stats['after_product_filter']:,}")
            logger.info(f"Final number of complaints: {stats['final_count']:,}")
            logger.info("="*50 + "\n")
            
            # Save the filtered data
            if self.processed_data_path:
                os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
                filtered_data.to_csv(self.processed_data_path, index=False)
                logger.info(f"Filtered data saved to {self.processed_data_path}")
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error filtering data: {str(e)}")
            # Log the available columns for debugging
            if 'filtered_data' in locals():
                logger.error(f"Available columns in filtered data: {filtered_data.columns.tolist()}")
            raise
    
    def clean_narratives(self) -> pd.DataFrame:
        """Clean the narrative text.
        
        Returns:
            pd.DataFrame: The data with cleaned narratives
        """
        if self.data is None:
            logger.warning("Data not loaded. Loading data first.")
            self.load_data()
        
        try:
            logger.info("Cleaning narratives")
            
            # Create a copy of the data
            cleaned_data = self.data.copy()
            
            # Define cleaning functions
            def remove_special_chars(text):
                # Keep alphanumeric, spaces, and basic punctuation
                return re.sub(r'[^\w\s.,!?;:()-]', ' ', text)
            
            def remove_extra_whitespace(text):
                # Replace multiple spaces with a single space
                return re.sub(r'\s+', ' ', text).strip()
            
            def remove_boilerplate(text):
                # Remove common boilerplate text
                boilerplate_patterns = [
                    r'XX+',  # Redacted information
                    r'XXXX+',
                    r'\{\d+\}',  # Reference numbers in curly braces
                    r'\[\d+\]',  # Reference numbers in square brackets
                ]
                
                for pattern in boilerplate_patterns:
                    text = re.sub(pattern, '', text)
                
                return text
            
            # Apply cleaning functions
            logger.info("Removing special characters")
            cleaned_data['cleaned_narrative'] = cleaned_data['narrative'].apply(remove_special_chars)
            
            logger.info("Removing boilerplate text")
            cleaned_data['cleaned_narrative'] = cleaned_data['cleaned_narrative'].apply(remove_boilerplate)
            
            logger.info("Removing extra whitespace")
            cleaned_data['cleaned_narrative'] = cleaned_data['cleaned_narrative'].apply(remove_extra_whitespace)
            
            # Update narrative length
            cleaned_data['cleaned_narrative_length'] = cleaned_data['cleaned_narrative'].str.len()
            
            self.data = cleaned_data
            return cleaned_data
        
        except Exception as e:
            logger.error(f"Error cleaning narratives: {str(e)}")
            raise
    
    def save_processed_data(self) -> None:
        """Save the processed data to a CSV file."""
        if self.data is None:
            logger.warning("No data to save")
            return
        
        try:
            logger.info(f"Saving processed data to {self.processed_data_path}")
            self.data.to_csv(self.processed_data_path, index=False)
            logger.info("Data saved successfully")
        
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def process_pipeline(self, products: List[str] = None, min_narrative_length: int = 50) -> pd.DataFrame:
        """Run the complete data processing pipeline.
        
        Args:
            products (list): List of products to include (if None, include all)
            min_narrative_length (int): Minimum narrative length to include
            
        Returns:
            pd.DataFrame: The processed data
        """
        try:
            logger.info("Running data processing pipeline")
            
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Filter data
            self.filter_data(products, min_narrative_length)
            
            # Step 3: Clean narratives
            self.clean_narratives()
            
            # Step 4: Save processed data
            self.save_processed_data()
            
            logger.info("Data processing pipeline completed successfully")
            return self.data
        
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    raw_data_path = Path("../../data/raw/complaints.csv")
    processed_data_path = Path("../../data/processed/processed_complaints.csv")
    
    # Define products of interest
    products_of_interest = [
        "Credit card",
        "Mortgage",
        "Debt collection",
        "Credit reporting, credit repair services, or other personal consumer reports",
        "Checking or savings account",
        "Money transfer, virtual currency, or money service",
        "Personal loan"
    ]
    
    # Create processor and run pipeline
    processor = DataProcessor(raw_data_path, processed_data_path)
    processed_data = processor.process_pipeline(products=products_of_interest, min_narrative_length=100)
    
    print(f"Processed {len(processed_data)} records")
    print(f"Sample of processed data:\n{processed_data[['product', 'issue', 'cleaned_narrative_length']].head()}")