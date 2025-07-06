import os
from pathlib import Path

print("=== Test Script ===")
print(f"Python executable: {os.sys.executable}")
print(f"Current working directory: {os.getcwd()}")

# Test file access
data_dir = Path("Data")
print(f"\nContents of Data directory:")
for item in data_dir.iterdir():
    print(f"- {item.name} (is_dir: {item.is_dir()})")
    if item.name == "complaints.csv" and item.is_dir():
        print("  Contents of complaints.csv directory:")
        try:
            for subitem in item.iterdir():
                print(f"  - {subitem.name} (size: {subitem.stat().st_size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"  Error listing contents: {e}")

# Try to read a small portion of the file
try:
    csv_file = data_dir / "complaints.csv" / "complaints.csv"
    print(f"\nAttempting to read first 100 bytes of {csv_file}...")
    with open(csv_file, 'rb') as f:
        print(f"First 100 bytes: {f.read(100)}")
except Exception as e:
    print(f"Error reading file: {e}")
    import traceback
    traceback.print_exc()
