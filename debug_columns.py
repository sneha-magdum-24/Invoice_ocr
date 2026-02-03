#!/usr/bin/env python3

import sys
sys.path.append('/Users/anantingale/Desktop/Invoice-to-OCR')

from universal_layout_pipeline import UniversalLayoutPipeline
import re

def debug_columns():
    pipeline = UniversalLayoutPipeline()
    
    # Extract text boxes
    text_boxes = pipeline.extract_text_with_coordinates('Invoice_ocr/IMG_0185.jpg')
    
    # Find money tokens
    money_boxes = []
    for box in text_boxes:
        text = box.text.strip()
        if (re.search(r'^[\$€e]?\d+[,.]?\d*(\.\d{2})?$', text) or 
            re.search(r'^\d+[,.]\d{3}[,.]\d{2}$', text)):
            money_boxes.append(box)
    
    print("=== ALL MONEY TOKENS BY POSITION ===")
    x_positions = [(box.bbox[0] + box.bbox[2]/2, box.text, box.bbox[1]) for box in money_boxes]
    x_positions.sort(key=lambda x: x[0])
    
    for x, text, y in x_positions:
        print(f"X={x:.1f}, Y={y:.1f}: '{text}'")
    
    # Current split
    mid_point = len(x_positions) // 2
    print(f"\n=== CURRENT SPLIT (mid_point={mid_point}) ===")
    print("Unit price column (left):")
    for i in range(mid_point):
        x, text, y = x_positions[i]
        print(f"  X={x:.1f}: '{text}'")
    
    print("Line total column (right):")
    for i in range(mid_point, len(x_positions)):
        x, text, y = x_positions[i]
        print(f"  X={x:.1f}: '{text}'")
    
    # Better split based on X gaps
    print(f"\n=== BETTER SPLIT (by X gaps) ===")
    x_coords = [x for x, _, _ in x_positions]
    
    # Find the biggest gap
    gaps = []
    for i in range(1, len(x_coords)):
        gap = x_coords[i] - x_coords[i-1]
        gaps.append((gap, i))
    
    if gaps:
        biggest_gap = max(gaps)
        split_point = biggest_gap[1]
        
        print(f"Biggest gap: {biggest_gap[0]:.1f}px at position {split_point}")
        print("Unit price column (left):")
        for i in range(split_point):
            x, text, y = x_positions[i]
            print(f"  X={x:.1f}: '{text}'")
        
        print("Line total column (right):")
        for i in range(split_point, len(x_positions)):
            x, text, y = x_positions[i]
            print(f"  X={x:.1f}: '{text}'")

if __name__ == "__main__":
    debug_columns()