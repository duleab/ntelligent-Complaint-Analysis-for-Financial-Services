import os
import sys
import unittest
import pandas as pd
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_processor import DataProcessor
from src.data.text_processor import TextProcessor

class TestDataProcessing(unittest.TestCase):
    """Test cases for the data processing components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a small test dataframe
        self.test_data = pd.DataFrame({
            'product': ['Credit card', 'Mortgage', 'Credit card', 'Personal loan'],
            'issue': ['Fees', 'Interest rate', 'Billing dispute', 'Application'],
            'consumer_complaint_narrative': [
                'I was charged a late fee even though I paid on time.',
                'The interest rate on my mortgage is higher than what was initially disclosed.',
                'I disputed a charge on my credit card but the company did not resolve it.',
                'I applied for a personal loan and was approved, but the terms changed at closing.'
            ],
            'date_received': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01']
        })
        
        # Create a temporary directory for test outputs
        self.test_dir = Path('test_outputs')
        self.test_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test output files
        for file in self.test_dir.glob('*'):
            file.unlink()
        
        # Remove test directory
        self.test_dir.rmdir()
    
    def test_data_processor_initialization(self):
        """Test that the data processor can be initialized."""
        processor = DataProcessor(
            raw_data_path=self.test_dir / "test_raw.csv",
            processed_data_path=self.test_dir / "test_processed.csv"
        )
        self.assertIsNotNone(processor)
    
    def test_data_cleaning(self):
        """Test the data cleaning functionality."""
        processor = DataProcessor(
            raw_data_path=self.test_dir / "test_raw.csv",
            processed_data_path=self.test_dir / "test_processed.csv"
        )
        
        # Save test data to CSV
        self.test_data.to_csv(self.test_dir / "test_raw.csv", index=False)
        
        # Load and process the data
        processor.load_data()
        processor.clean_narratives()
        
        # Check that all narratives are cleaned
        for narrative in processor.data['cleaned_narrative']:
            self.assertNotIn('XXXX', narrative)
            self.assertNotIn('  ', narrative)
    
    def test_data_filtering(self):
        """Test the data filtering functionality."""
        processor = DataProcessor(
            raw_data_path=self.test_dir / "test_raw.csv",
            processed_data_path=self.test_dir / "test_processed.csv"
        )
        
        # Save test data to CSV
        self.test_data.to_csv(self.test_dir / "test_raw.csv", index=False)
        
        # Load the data
        processor.load_data()
        
        # Test filtering by product
        processor.filter_data(products=['Credit card'])
        
        # Check that only credit card complaints remain
        self.assertEqual(len(processor.data), 2)
        self.assertTrue(all(product == 'Credit card' for product in processor.data['product']))
        
        # Reset processor with fresh data
        processor = DataProcessor(
            raw_data_path=self.test_dir / "test_raw.csv",
            processed_data_path=self.test_dir / "test_processed.csv"
        )
        processor.load_data()
        
        # Test filtering by narrative length
        processor.filter_data(min_narrative_length=50)
        
        # Check that short narratives are removed
        self.assertLess(len(processor.data), len(self.test_data))
        for narrative in processor.data['narrative']:
            self.assertGreaterEqual(len(narrative), 50)
    
    def test_text_processor_initialization(self):
        """Test that the text processor can be initialized."""
        processor = TextProcessor()
        self.assertIsNotNone(processor)
    
    def test_text_chunking(self):
        """Test the text chunking functionality."""
        processor = TextProcessor()
        
        # Test chunking a single text
        text = "This is a test narrative. It has multiple sentences. We want to see how it gets chunked."
        chunks = processor.chunk_text(text)
        
        # Check that we got chunks
        self.assertIsNotNone(chunks)
        self.assertGreater(len(chunks), 0)
    
    def test_embedding_generation(self):
        """Test the embedding generation functionality."""
        processor = TextProcessor()
        
        # Test generating embeddings for a single text
        text = "This is a test narrative."
        embedding = processor.embed_text(text)
        
        # Check that we got an embedding
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)

if __name__ == "__main__":
    unittest.main()