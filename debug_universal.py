from universal_layout_pipeline import UniversalLayoutPipeline
import json

def debug_extraction(image_path):
    pipeline = UniversalLayoutPipeline()
    
    # Step 1: Get text boxes
    text_boxes = pipeline.extract_text_with_coordinates(image_path)
    
    print(f"=== DEBUG: {image_path} ===")
    print(f"Found {len(text_boxes)} text boxes:")
    
    # Show all text boxes
    for i, box in enumerate(text_boxes):
        print(f"{i:2d}: '{box.text}' at {box.bbox} (conf: {box.confidence:.2f})")
    
    # Check for numeric patterns (potential table data)
    numeric_boxes = [box for box in text_boxes if any(c.isdigit() for c in box.text)]
    print(f"\nNumeric boxes ({len(numeric_boxes)}):")
    for box in numeric_boxes:
        print(f"  '{box.text}' at {box.bbox}")
    
    # Check for currency patterns
    currency_boxes = [box for box in text_boxes if '$' in box.text or any(c.isdigit() and '.' in box.text for c in box.text)]
    print(f"\nCurrency boxes ({len(currency_boxes)}):")
    for box in currency_boxes:
        print(f"  '{box.text}' at {box.bbox}")

if __name__ == "__main__":
    debug_extraction("Invoice_ocr/IMG_0187.jpg")