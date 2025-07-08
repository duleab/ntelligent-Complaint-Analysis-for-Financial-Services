"""Test script for the text preprocessing functions.
Run this to verify that all cleaning functions work as expected.
"""
import re
import unittest
from cleaned_preprocessing import (
    remove_special_characters,
    remove_extra_whitespace,
    remove_boilerplate,
    clean_text
)

class TestTextCleaning(unittest.TestCase):    
    def test_remove_special_characters(self):
        self.assertEqual(remove_special_characters("Hello! @world #test"), "Hello  world  test")
        self.assertEqual(remove_special_characters("Testing 1, 2, 3..."), "Testing 1, 2, 3   ")
        self.assertEqual(remove_special_characters(123), '')
        self.assertEqual(remove_special_characters(None), '')
    
    def test_remove_extra_whitespace(self):
        self.assertEqual(remove_extra_whitespace("hello    world"), "hello world")
        self.assertEqual(remove_extra_whitespace("new\nline\ttab"), "new line tab")
        self.assertEqual(remove_extra_whitespace("  leading  "), "leading")
    
    def test_remove_boilerplate(self):
        test_text = "I am writing to file a complaint about XXXX"
        self.assertEqual(
            remove_boilerplate(test_text), 
            " about [REDACTED]"
        )
        self.assertEqual(
            remove_boilerplate("Date: XX/XX/XXXX"), 
            "Date: [DATE]"
        )
    
    def test_clean_text_integration(self):
        dirty_text = """
        I am writing to file a complaint about my credit card XXXX-XXXX-XXXX-XXXX.
        The issue occurred on XX/XX/2023. Please help!
        """
        cleaned = clean_text(dirty_text)
        self.assertNotIn("XXXX-XXXX-XXXX-XXXX", cleaned)
        self.assertNotIn("XX/XX/2023", cleaned)
        self.assertNotIn("\n", cleaned)
        self.assertTrue(cleaned.islower())

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run a sample test
    test_text = """
    I am writing to file a complaint about my credit card XXXX-XXXX-XXXX-XXXX.
    The issue occurred on XX/XX/2023. Please help!
    """
    print("\nSample test:")
    print("Original:", repr(test_text))
    print("Cleaned:", repr(clean_text(test_text)))
