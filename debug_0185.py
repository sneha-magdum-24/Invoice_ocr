from universal_layout_pipeline import UniversalLayoutPipeline

def debug_img_0185():
    pipeline = UniversalLayoutPipeline()
    
    # Get OCR results
    text_boxes = pipeline.extract_text_with_coordinates("Invoice_ocr/IMG_0185.jpg")
    pipeline._current_text_boxes = text_boxes
    
    print("=== ALL OCR TEXT ===")
    all_text = ' '.join([box.text for box in text_boxes])
    print(all_text)
    
    print(f"\n=== {len(text_boxes)} TEXT BOXES ===")
    for i, box in enumerate(text_boxes):
        print(f"{i:2d}: '{box.text}' at {box.bbox} conf:{box.confidence:.2f}")
    
    # Check money patterns
    money_boxes = [box for box in text_boxes if re.search(r'\$?\d+(\.\d{2})?', box.text)]
    print(f"\n=== {len(money_boxes)} MONEY BOXES ===")
    for box in money_boxes:
        print(f"'{box.text}' at {box.bbox}")
    
    # Check vendor patterns
    print(f"\n=== VENDOR EXTRACTION ===")
    vendor = pipeline._extract_vendor_with_scoring(all_text)
    print(f"Vendor: {vendor}")
    
    # Check totals
    print(f"\n=== TOTALS EXTRACTION ===")
    totals = pipeline._extract_financial_totals(all_text)
    print(f"Totals: {totals}")

if __name__ == "__main__":
    import re
    debug_img_0185()