from universal_layout_pipeline import UniversalLayoutPipeline
import re

def debug_totals():
    pipeline = UniversalLayoutPipeline()
    
    # Get text boxes
    text_boxes = pipeline.extract_text_with_coordinates("Invoice_ocr/IMG_0187.jpg")
    
    # Reconstruct all text
    all_text = ' '.join([box.text for box in text_boxes])
    
    print("=== ALL TEXT ===")
    print(all_text)
    
    print("\n=== CURRENCY MATCHES ===")
    currency_matches = re.findall(r'\\$([0-9,]+\\.?\\d{0,2})', all_text)
    print("Currency matches:", currency_matches)
    
    print("\n=== PAYMENT AMOUNT DUE SEARCH ===")
    payment_due_match = re.search(r'(?i)payment\\s+amount\\s+due[:\\s]*\\$?([0-9,]+\\.?\\d{0,2})', all_text)
    if payment_due_match:
        print("Found Payment Amount Due:", payment_due_match.group(1))
    else:
        print("No Payment Amount Due found")
    
    # Check for the specific pattern we saw in debug
    print("\\n=== LOOKING FOR $910 ===")
    if '$910' in all_text:
        print("Found $910 in text")
        # Find context around $910
        idx = all_text.find('$910')
        context = all_text[max(0, idx-50):idx+50]
        print("Context:", context)

if __name__ == "__main__":
    debug_totals()