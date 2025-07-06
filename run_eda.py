"""
Run EDA and save output to a file
"""
import sys
from pathlib import Path

def main():
    # Redirect stdout to a file
    output_file = Path("eda_output.txt")
    print(f"Saving output to {output_file.absolute()}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Save original stdout
        original_stdout = sys.stdout
        sys.stdout = f
        
        try:
            # Import and run the EDA script
            print("=== Starting EDA Analysis ===\n")
            from simple_eda import main as run_eda
            run_eda()
        except Exception as e:
            import traceback
            print("\n=== Error occurred ===")
            traceback.print_exc()
        finally:
            # Restore original stdout
            sys.stdout = original_stdout
    
    print(f"\nAnalysis complete. Results saved to: {output_file}")

if __name__ == "__main__":
    main()
