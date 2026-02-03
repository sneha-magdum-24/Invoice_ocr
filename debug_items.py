from universal_layout_pipeline import UniversalLayoutPipeline
import re

def debug_items():
    pipeline = UniversalLayoutPipeline()
    text_boxes = pipeline.extract_text_with_coordinates("Invoice_ocr/IMG_0185.jpg")
    pipeline._current_text_boxes = text_boxes
    
    print("=== DEBUGGING ITEM EXTRACTION ===")
    
    # Check money detection
    money_boxes = []
    for box in text_boxes:
        text = box.text.strip()
        if (re.search(r'^[\$€e]?\d+[,.]?\d*(\.\d{2})?$', text) or 
            re.search(r'^\d+[,.]\d{3}[,.]\d{2}$', text)):
            money_boxes.append(box)
    
    print(f"Found {len(money_boxes)} money boxes:")
    for box in money_boxes:
        print(f"  '{box.text}' at {box.bbox}")
    
    if len(money_boxes) >= 2:
        # Test clustering
        x_positions = [box.bbox[0] + box.bbox[2]/2 for box in money_boxes]
        x_positions.sort()
        amount_column_x = x_positions[-len(x_positions)//2:]
        avg_amount_x = sum(amount_column_x) / len(amount_column_x)
        
        print(f"\nAmount column X average: {avg_amount_x}")
        
        amount_boxes = [box for box in money_boxes 
                       if abs((box.bbox[0] + box.bbox[2]/2) - avg_amount_x) < 50]
        
        print(f"Amount boxes in column: {len(amount_boxes)}")
        for box in amount_boxes:
            print(f"  '{box.text}' at {box.bbox}")
            
            # Check for descriptions
            y_center = box.bbox[1] + box.bbox[3]/2
            desc_boxes = []
            for other_box in text_boxes:
                box_y_center = other_box.bbox[1] + other_box.bbox[3]/2
                if (abs(box_y_center - y_center) < 20 and
                    other_box.bbox[0] < box.bbox[0] and
                    not re.search(r'^[\$€e]?\d+[,.]?\d*(\.\d{2})?$', other_box.text.strip())):
                    desc_boxes.append(other_box)
            
            print(f"    Descriptions: {[b.text for b in desc_boxes]}")

if __name__ == "__main__":
    debug_items()