import pandas as pd
import os

# Path to your data
DATA_DIR = os.path.join('..', '..', 'Data')
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, 'sample_complaints.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'processed_complaints.csv')

# Check if files exist
if not os.path.exists(SAMPLE_DATA_PATH) and not os.path.exists(PROCESSED_DATA_PATH):
    print(f"Error: No data files found. Please make sure you've run the download_data.py script first.")
    exit(1)

# Use processed data if available, otherwise use sample data
DATA_PATH = PROCESSED_DATA_PATH if os.path.exists(PROCESSED_DATA_PATH) else SAMPLE_DATA_PATH
print(f"Using data from: {DATA_PATH}")

# Function to check for BNPL mentions
def check_bnpl_mentions():
    # Read the data in chunks to handle large files
    chunk_size = 1000
    bnpl_mentions = []
    
    # BNPL related keywords to search for (case insensitive)
    bnpl_keywords = [
        'buy now pay later', 'bnpl', 'pay later', 'buy now', 
        'installment', 'split payment', 'deferred payment',
        'pay in 4', 'pay in four', 'klarna', 'affirm', 'afterpay',
        'sezzle', 'zip pay', 'quadpay', 'splitit', 'perpay',
        'paypal credit', 'paypal pay later', 'amazon pay later'
    ]
    
    # Try to read the file with different encodings if needed
    encodings = ['utf-8', 'latin-1', 'ISO-8859-1']
    
    for encoding in encodings:
        try:
            # Read the first chunk to check the file structure
            first_chunk = pd.read_csv(DATA_PATH, nrows=1, encoding=encoding, low_memory=False)
            logger.info(f"Successfully read file with {encoding} encoding")
            logger.info(f"Columns found: {first_chunk.columns.tolist()}")
            break
        except UnicodeDecodeError:
            logger.warning(f"Failed to read with {encoding} encoding, trying next...")
            continue
    else:
        logger.error("Failed to read file with any encoding")
        return
    
    # Now read the full file in chunks
    for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size, low_memory=False, encoding=encoding):
        # Convert all string columns to lowercase for case-insensitive search
        str_columns = chunk.select_dtypes(include=['object']).columns
        chunk_lower = chunk[str_columns].apply(lambda x: x.astype(str).str.lower())
        
        # Check for BNPL mentions in any column
        for col in chunk_lower.columns:
            for keyword in bnpl_keywords:
                matches = chunk[chunk_lower[col].str.contains(keyword, na=False)]
                if not matches.empty:
                    # Add the column name and keyword that matched
                    matches['matched_column'] = col
                    matches['matched_keyword'] = keyword
                    bnpl_mentions.append(matches)
        
        # If we found any matches, we can stop searching
        if bnpl_mentions:
            break
    
    # Combine all matches
    if bnpl_mentions:
        all_matches = pd.concat(bnpl_mentions)
        print("\nFound BNPL-related complaints in the following columns:")
        print(all_matches[['matched_column', 'matched_keyword']].value_counts())
        
        print("\nSample of BNPL-related complaints:")
        print(all_matches[['Complaint ID', 'Product', 'Issue', 'Consumer complaint narrative']].head())
        
        # Save results
        output_file = os.path.join(DATA_DIR, 'processed', 'bnpl_mentions.csv')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        all_matches.to_csv(output_file, index=False)
        print(f"\nSaved {len(all_matches)} BNPL-related complaints to {output_file}")
    else:
        print("No BNPL-related complaints found in the first 1000 records.")
        print("\nAvailable product categories:")
        # Print unique product categories from the first chunk
        if 'Product' in chunk.columns:
            print(chunk['Product'].value_counts().head(20))
        else:
            print("No 'Product' column found in the data.")
            print("\nAvailable columns:", chunk.columns.tolist())

if __name__ == "__main__":
    print("Checking for BNPL-related complaints in the dataset...")
    check_bnpl_mentions()
