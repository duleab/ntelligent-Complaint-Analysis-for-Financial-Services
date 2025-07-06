import unittest
import sys
import os
from pathlib import Path

def run_tests():
    """Run all tests in the tests directory."""
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Add the project root to the path
    sys.path.append(str(project_root))
    
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return 0 if all tests passed, 1 otherwise
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    exit(run_tests())