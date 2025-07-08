import os
import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit application with proper environment setup."""
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Set the PYTHONPATH to include the project root
    os.environ["PYTHONPATH"] = str(project_root)
    
    # Path to the Streamlit app
    app_path = project_root / "src" / "streamlit_app.py"
    
    print(f"Starting Streamlit app from: {app_path}")
    print("You can access the app at http://localhost:8501 once it's running")
    print("Press Ctrl+C to stop the app")
    
    # Run the Streamlit app
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            check=True
        )
    except KeyboardInterrupt:
        print("\nStreamlit app stopped")
    except subprocess.CalledProcessError as e:
        print(f"\nError running Streamlit app: {e}")
        print("Make sure you have installed all required dependencies with 'pip install -r requirements.txt'")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main()