"""
Cleaned version of the text preprocessing functions for the RAG pipeline.
Copy the contents of each function into the corresponding notebook cell.
"""
import re

def remove_special_characters(text):
    """
    Remove special characters and normalize text.
    Keeps alphanumeric characters, basic punctuation, and whitespace.
    """
    if not isinstance(text, str):
        return ''
    # Keep only alphanumeric, basic punctuation, and whitespace
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_extra_whitespace(text):
    """Remove extra whitespace, including newlines and tabs."""
    if not isinstance(text, str):
        return ''
    # Replace newlines and tabs with spaces
    text = re.sub(r'[\n\t\r]', ' ', text)
    # Replace multiple spaces with a single space
    return re.sub(r'\s+', ' ', text).strip()

def remove_boilerplate(text):
    """
    Remove common boilerplate text and redacted information.
    Handles both case-sensitive and case-insensitive patterns.
    """
    if not isinstance(text, str):
        return ''
        
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Common boilerplate phrases to remove (case-insensitive)
    boilerplate_phrases = [
        'this is a complaint about',
        'i am writing to file a complaint',
        'please be advised that',
        'to whom it may concern',
        'dear sir/madam',
        'sincerely',
        'best regards',
        'thank you for your attention',
        'i look forward to your response',
        'please let me know if you need any further information',
        'yours truly',
        'regards',
        'complaint id:',
        'case number:',
        'reference number:'
    ]
    
    # Handle redacted patterns first (e.g., XXXX, XX/XX/XXXX, [redacted])
    text = re.sub(r'X{2,}', '[REDACTED]', text)  # Any sequence of 2+ X's
    text = re.sub(r'\b\d{1,2}/X{1,2}/X{2,4}\b', '[DATE]', text)  # Dates with X's
    text = re.sub(r'\bX{1,2}/X{1,2}/\d{2,4}\b', '[DATE]', text)  # More date patterns
    text = re.sub(r'\[redacted\]', '[REDACTED]', text, flags=re.IGNORECASE)
    
    # Remove boilerplate phrases
    for phrase in boilerplate_phrases:
        text = re.sub(re.escape(phrase), '', text, flags=re.IGNORECASE)
    
    # Clean up any leftover special patterns
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'\s*\[\s*', ' [', text)  # Fix spacing around brackets
    text = re.sub(r'\s*\]\s*', '] ', text)
    
    return text.strip()

def clean_text(text):
    """
    Apply all cleaning functions in sequence.
    
    Args:
        text (str): Input text to be cleaned
        
    Returns:
        str: Cleaned text with standardized formatting
    """
    if not isinstance(text, str):
        return ''
    
    # Apply cleaning functions in optimal sequence
    text = text.strip()  # First remove any leading/trailing whitespace
    if not text:  # If empty after stripping, return empty string
        return ''
        
    # Apply cleaning functions in sequence
    text = remove_boilerplate(text)  # Handles its own lowercase conversion
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)
    
    return text
