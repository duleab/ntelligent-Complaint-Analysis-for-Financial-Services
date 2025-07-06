"""Simple test script for text cleaning functions.
Run this in your notebook with: %run test_cleaning.py
"""

def test_clean_text(test_input, expected_output, test_num):
    """Test the clean_text function with given input and expected output."""
    from cleaned_preprocessing import clean_text
    
    result = clean_text(test_input)
    passed = (result == expected_output)
    
    print(f"Test {test_num}:")
    print(f"  Input:    {repr(test_input)}")
    print(f"  Expected: {repr(expected_output)}")
    print(f"  Got:      {repr(result)}")
    print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")
    print()
    
    return passed

def run_tests():
    """Run all test cases for the text cleaning functions."""
    print("Running cleaning function tests...\n")
    
    test_cases = [
        # Test 1: Redacted text and boilerplate
        (
            "I am writing to file a complaint about XXXX-XXXX-XXXX-XXXX",
            "about [REDACTED] - [REDACTED] - [REDACTED] - [REDACTED] - issue with my account",
            1
        ),
        # Test 2: Basic text
        (
            "This is a normal complaint about poor service.",
            "this is a normal complaint about poor service",
            2
        ),
        # Test 3: Extra whitespace
        (
            "   Too  much   whitespace    here   ",
            "too much whitespace here",
            3
        ),
        # Test 4: Special characters
        (
            "Special!@# characters$%^&*()",
            "special characters",
            4
        )
    ]
    
    # Run tests
    passed_count = 0
    for test_input, expected_output, test_num in test_cases:
        if test_clean_text(test_input, expected_output, test_num):
            passed_count += 1
    
    # Print summary
    total_tests = len(test_cases)
    print(f"\nTest Results: {passed_count}/{total_tests} tests passed")
    if passed_count == total_tests:
        print("✅ All tests passed successfully!")
    else:
        print(f"❌ {total_tests - passed_count} tests failed")

if __name__ == "__main__":
    run_tests()
if __name__ == '__main__':
    test_cleaning()
