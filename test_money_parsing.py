#!/usr/bin/env python3

import sys
sys.path.append('.')
from universal_layout_pipeline import UniversalLayoutPipeline

def test_money_parsing():
    pipeline = UniversalLayoutPipeline()
    
    test_cases = [
        # Problem cases from your invoice
        ("65,200.00", 65200.0),  # Should NOT be 65.2
        ("3,000.00", 3000.0),
        ("1,500.00", 1500.0),
        ("e5,200.00", 5200.0),   # OCR error case
        
        # Edge cases
        ("5200,00", 5200.0),     # EU decimal format
        ("3,000", 3000.0),       # Thousands separator only
        ("500.00", 500.0),       # Normal decimal
        ("1.500,00", 1500.0),    # EU format with thousands
    ]
    
    print("Testing money parsing fixes:")
    print("=" * 50)
    
    all_passed = True
    for input_text, expected in test_cases:
        result = pipeline._parse_money_safe(input_text)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"{status} '{input_text}' -> {result} (expected {expected})")
        if result != expected:
            all_passed = False
    
    print("=" * 50)
    print(f"Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    return all_passed

if __name__ == "__main__":
    test_money_parsing()