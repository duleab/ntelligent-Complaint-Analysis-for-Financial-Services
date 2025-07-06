import os
import sys
from pathlib import Path

def main():
    print("=== Simple Test Script ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Test basic file operations
    test_file = Path("test_output.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("Test successful!")
        print(f"Successfully wrote to {test_file}")
        os.remove(test_file)
        print(f"Successfully deleted {test_file}")
    except Exception as e:
        print(f"Error in file operations: {e}")
    
    # Test accessing the data file
    data_file = Path("Data") / "complaints.csv" / "complaints.csv"
    print(f"\nTesting access to: {data_file}")
    print(f"File exists: {data_file.exists()}")
    print(f"File size: {data_file.stat().st_size / (1024*1024):.2f} MB")
    
    # Try to read first few lines
    try:
        print("\nFirst 3 lines of the file:")
        with open(data_file, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f, 1):
                print(f"{i}: {line.strip()}")
                if i >= 3:
                    break
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    main()
