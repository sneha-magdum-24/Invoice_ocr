#!/usr/bin/env python3

import sys
sys.path.append('/Users/anantingale/Desktop/Invoice-to-OCR')

from universal_layout_pipeline import UniversalLayoutPipeline
import re

def debug_money_detection():
    pipeline = UniversalLayoutPipeline()
    
    # Extract text boxes
    text_boxes = pipeline.extract_text_with_coordinates('Invoice_ocr/IMG_0185.jpg')
    
    print("=== ALL MONEY TOKENS ===")
    money_boxes = []
    for i, box in enumerate(text_boxes):
        text = box.text.strip()
        if (re.search(r'^[\$€e]?\d+[,.]?\d*(\.\d{2})?$', text) or 
            re.search(r'^\d+[,.]?\d{3}[,.]?\d{2}$', text)):
            money_boxes.append(box)
            print(f"{i}: '{text}' at X={box.bbox[0]}, Y={box.bbox[1]}")
    
    print(f"\nFound {len(money_boxes)} money tokens")
    
    if len(money_boxes) >= 2:
        # Show X positions
        x_positions = [(box.bbox[0] + box.bbox[2]/2, box.text) for box in money_boxes]
        x_positions.sort(key=lambda x: x[0])
        
        print("\n=== MONEY TOKENS BY X POSITION ===")
        for x, text in x_positions:
            print(f"X={x:.1f}: '{text}'")
        
        # Show potential columns
        mid_point = len(x_positions) // 2
        print(f"\nUnit price column (left {mid_point}):")
        for i in range(mid_point):
            print(f"  '{x_positions[i][1]}'")
        
        print(f"\nLine total column (right {len(x_positions)-mid_point}):")
        for i in range(mid_point, len(x_positions)):
            print(f"  '{x_positions[i][1]}'")

if __name__ == "__main__":
    debug_money_detection()