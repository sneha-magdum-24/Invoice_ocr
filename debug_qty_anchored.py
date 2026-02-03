#!/usr/bin/env python3

import sys
sys.path.append('/Users/anantingale/Desktop/Invoice-to-OCR')

from universal_layout_pipeline import UniversalLayoutPipeline
import re

def debug_qty_anchored():
    pipeline = UniversalLayoutPipeline()
    
    # Extract text boxes
    text_boxes = pipeline.extract_text_with_coordinates('Invoice_ocr/IMG_0185.jpg')
    
    print("=== LOOKING FOR QUANTITY ANCHORS ===")
    qty_anchors = []
    for i, box in enumerate(text_boxes):
        qty_match = re.search(r'(\d+)\s*(hour|hrs|h)', box.text.lower())
        if qty_match:
            qty_anchors.append({
                'qty': int(qty_match.group(1)),
                'y_center': box.bbox[1] + box.bbox[3]/2,
                'box': box,
                'text': box.text
            })
            print(f"Found qty anchor: {box.text} -> qty={qty_match.group(1)} at Y={box.bbox[1] + box.bbox[3]/2}")
    
    print(f"\nTotal qty anchors found: {len(qty_anchors)}")
    
    print("\n=== LOOKING FOR MONEY TOKENS ===")
    money_boxes = []
    for i, box in enumerate(text_boxes):
        text = box.text.strip()
        if re.search(r'^[\$€e]?\d+[,.]?\d*(\.\d{2})?$', text):
            money_boxes.append(box)
            print(f"Money token: '{text}' at X={box.bbox[0] + box.bbox[2]/2:.1f}, Y={box.bbox[1] + box.bbox[3]/2:.1f}")
    
    print(f"\nTotal money tokens found: {len(money_boxes)}")
    
    if len(money_boxes) >= 2:
        # Show column separation
        x_positions = [(box.bbox[0] + box.bbox[2]/2, box) for box in money_boxes]
        x_positions.sort(key=lambda x: x[0])
        
        x_coords = [x for x, _ in x_positions]
        gaps = [(x_coords[i] - x_coords[i-1], i) for i in range(1, len(x_coords))]
        
        if gaps:
            biggest_gap = max(gaps)
            split_point = biggest_gap[1]
            print(f"\nBiggest gap: {biggest_gap[0]:.1f}px at position {split_point}")
            
            unit_price_boxes = [box for _, box in x_positions[:split_point]]
            total_boxes = [box for _, box in x_positions[split_point:]]
            
            print(f"Unit price column: {len(unit_price_boxes)} tokens")
            for box in unit_price_boxes:
                print(f"  '{box.text}' at Y={box.bbox[1] + box.bbox[3]/2:.1f}")
            
            print(f"Total column: {len(total_boxes)} tokens")  
            for box in total_boxes:
                print(f"  '{box.text}' at Y={box.bbox[1] + box.bbox[3]/2:.1f}")
    
    # Test row matching
    if qty_anchors and len(money_boxes) >= 2:
        print(f"\n=== TESTING ROW MATCHING ===")
        for anchor in qty_anchors:
            print(f"\nTesting qty {anchor['qty']} at Y={anchor['y_center']:.1f}")
            row_band = 40
            
            # Find candidates in row band
            up_candidates = []
            total_candidates = []
            
            for box in money_boxes:
                box_y = box.bbox[1] + box.bbox[3]/2
                if abs(box_y - anchor['y_center']) <= row_band:
                    print(f"  Money token '{box.text}' at Y={box_y:.1f} (diff: {abs(box_y - anchor['y_center']):.1f}px)")

if __name__ == "__main__":
    debug_qty_anchored()