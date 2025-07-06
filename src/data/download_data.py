import os
import pandas as pd
import requests
import zipfile
import io
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_cfpb_data(output_dir, sample_size=None):
    """Download the CFPB complaint database CSV file.
    
    Args:
        output_dir (str): Directory to save the downloaded data
        sample_size (int, optional): Number of rows to sample. If 0 or None, all data is kept.
    
    Returns:
        str: Path to the downloaded file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # URL for the CFPB complaint database
    url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
    output_file = output_path / "complaints.csv"
    
    # Check if file already exists
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024 * 1024)  # Size in MB
        logger.info(f"File already exists at {output_file} (Size: {file_size:.2f} MB)")
        if file_size < 100:  # If file is too small, it might be incomplete
            logger.warning("Existing file seems too small. Redownloading...")
            output_file.unlink()
        else:
            return str(output_file)
    
    logger.info(f"Downloading CFPB complaint data from {url}")
    
    try:
        # Download the zip file with a timeout and streaming
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Save the zip file temporarily
        temp_zip = output_path / "temp_complaints.zip"
        with open(temp_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Extract the CSV file from the zip archive
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            # Get the name of the CSV file in the archive
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV file found in the downloaded zip archive")
            
            csv_file = csv_files[0]
            logger.info(f"Extracting {csv_file} from zip archive")
            
            if sample_size is None or sample_size == 0:
                # Extract the entire file
                logger.info("Extracting full dataset (this may take a while...)")
                zip_ref.extract(csv_file, output_path)
                # Rename the extracted file if needed
                extracted_file = output_path / csv_file
                if str(extracted_file) != str(output_file):
                    os.rename(extracted_file, output_file)
                
                # Verify the extracted file size
                if output_file.exists():
                    file_size = output_file.stat().st_size / (1024 * 1024)  # Size in MB
                    logger.info(f"Extracted file size: {file_size:.2f} MB")
                    if file_size < 100:  # If file is too small, it might be incomplete
                        raise ValueError(f"Extracted file seems too small ({file_size:.2f} MB). The download might be incomplete.")
            else:
                # Read a sample of the data
                logger.info(f"Sampling {sample_size} rows from the dataset")
                with zip_ref.open(csv_file) as f:
                    # Read the full file first to get all columns
                    df = pd.read_csv(f, nrows=1)
                    columns = df.columns.tolist()
                    # Now read the actual sample with all columns
                    f.seek(0)
                    df = pd.read_csv(f, nrows=sample_size, usecols=columns, low_memory=False)
                    df.to_csv(output_file, index=False)
        
        # Clean up the temporary zip file
        if temp_zip.exists():
            temp_zip.unlink()
            
        logger.info(f"Successfully downloaded and extracted data to {output_file}")
        return str(output_file)
        
        logger.info(f"Data successfully downloaded and saved to {output_file}")
        return str(output_file)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading data: {e}")
        raise
    except zipfile.BadZipFile as e:
        logger.error(f"Error extracting zip file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def main():
    """Main function to download CFPB complaint data."""
    parser = argparse.ArgumentParser(description="Download CFPB complaint data")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="../../data/raw",
        help="Directory to save the downloaded data"
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=None,
        help="Number of rows to sample. If not provided, all data is downloaded."
    )
    args = parser.parse_args()
    
    try:
        file_path = download_cfpb_data(args.output_dir, args.sample_size)
        logger.info(f"Data downloaded to {file_path}")
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())