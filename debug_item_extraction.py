#!/usr/bin/env python3

import sys
sys.path.append('/Users/anantingale/Desktop/Invoice-to-OCR')

from universal_layout_pipeline import UniversalLayoutPipeline
import re

def debug_item_extraction():
    pipeline = UniversalLayoutPipeline()
    
    # Extract text boxes
    text_boxes = pipeline.extract_text_with_coordinates('Invoice_ocr/IMG_0185.jpg')
    pipeline._current_text_boxes = text_boxes
    
    # Get page height
    page_height = max(box.bbox[1] + box.bbox[3] for box in text_boxes)
    
    # Find money tokens
    money_boxes = []
    for box in text_boxes:
        text = box.text.strip()
        if (re.search(r'^[\$€e]?\d+[,.]?\d*(\.\d{2})?$', text) or 
            re.search(r'^\d+[,.]\d{3}[,.]\d{2}$', text)):
            money_boxes.append(box)
    
    print(f"Found {len(money_boxes)} money tokens")
    
    # Cluster by X position
    x_positions = [(box.bbox[0] + box.bbox[2]/2, box) for box in money_boxes]
    x_positions.sort(key=lambda x: x[0])
    
    mid_point = len(x_positions) // 2
    line_total_boxes = [box for _, box in x_positions[mid_point:]]
    
    print(f"\nChecking {len(line_total_boxes)} line total boxes:")
    
    for i, total_box in enumerate(line_total_boxes):
        y_center = total_box.bbox[1] + total_box.bbox[3]/2
        
        # Find description boxes
        desc_boxes = []
        for box in text_boxes:
            box_y_center = box.bbox[1] + box.bbox[3]/2
            if (abs(box_y_center - y_center) < 50 and
                box.bbox[0] < total_box.bbox[0] and
                not re.search(r'^[\$€e]?\d+[,.]?\d*(\.\d{2})?$', box.text.strip())):
                desc_boxes.append(box)
        
        print(f"\nRow {i+1}: Amount '{total_box.text}' at Y={y_center}")
        
        if desc_boxes:
            desc_boxes.sort(key=lambda b: b.bbox[0])
            description = ' '.join(box.text for box in desc_boxes)
            print(f"  Description: '{description}'")
            
            # Check header filter
            is_header = pipeline._is_header_row(description, y_center, page_height)
            print(f"  Is header: {is_header}")
            
            if not is_header:
                # Check row score
                row_score = pipeline._score_item_row(description, total_box.text)
                print(f"  Row score: {row_score}")
                
                if row_score >= 1:
                    print(f"  ✅ WOULD EXTRACT")
                else:
                    print(f"  ❌ Score too low")
            else:
                print(f"  ❌ Filtered as header")
        else:
            print(f"  No description found")

if __name__ == "__main__":
    debug_item_extraction()