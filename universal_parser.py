import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import json
import re
import requests

@dataclass
class TextBox:
    text: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    page: int = 0

class UniversalInvoiceParser:
    """Universal invoice parser that can handle any invoice format worldwide"""
    
    def __init__(self):
        pass
    
    def extract_text_with_coordinates(self, image_path: str) -> List[TextBox]:
        """Extract text with coordinates using best available OCR"""
        try:
            return self._extract_with_paddleocr(image_path)
        except:
            try:
                return self._extract_with_easyocr(image_path)
            except:
                return self._extract_with_tesseract(image_path)
    
    def _extract_with_paddleocr(self, image_path: str) -> List[TextBox]:
        """Extract using PaddleOCR"""
        from paddleocr import PaddleOCR
        
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(image_path, cls=True)
        
        text_boxes = []
        for line in result[0]:
            bbox_points = line[0]
            text = line[1][0]
            confidence = line[1][1]
            
            x = min(point[0] for point in bbox_points)
            y = min(point[1] for point in bbox_points)
            w = max(point[0] for point in bbox_points) - x
            h = max(point[1] for point in bbox_points) - y
            
            text_boxes.append(TextBox(text, (int(x), int(y), int(w), int(h)), confidence))
        
        return text_boxes
    
    def _extract_with_easyocr(self, image_path: str) -> List[TextBox]:
        """Extract using EasyOCR"""
        import easyocr
        
        reader = easyocr.Reader(['en'])
        results = reader.readtext(image_path)
        
        text_boxes = []
        for bbox_points, text, confidence in results:
            x = min(point[0] for point in bbox_points)
            y = min(point[1] for point in bbox_points)
            w = max(point[0] for point in bbox_points) - x
            h = max(point[1] for point in bbox_points) - y
            
            text_boxes.append(TextBox(text, (int(x), int(y), int(w), int(h)), confidence))
        
        return text_boxes
    
    def _extract_with_tesseract(self, image_path: str) -> List[TextBox]:
        """Extract using Tesseract"""
        import pytesseract
        from PIL import Image
        
        img = Image.open(image_path)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        text_boxes = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:
                text = data['text'][i].strip()
                if text:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    confidence = int(data['conf'][i]) / 100.0
                    text_boxes.append(TextBox(text, (x, y, w, h), confidence))
        
        return text_boxes
    
    def _parse_money_safe(self, text: str) -> float:
        """Universal money parser"""
        clean = re.sub(r'[€$£¥₹₽¢₩₪₦₨₡₵₴₸₺₼₾₿\s]', '', text.strip())
        
        if not clean or not re.search(r'\d', clean):
            return 0
        
        # Handle different decimal separators
        if ',' in clean and '.' in clean:
            clean = clean.replace(',', '')
        elif ',' in clean:
            parts = clean.split(',')
            if len(parts) == 2 and len(parts[1]) == 2:
                clean = clean.replace(',', '.')
            else:
                clean = clean.replace(',', '')
        
        try:
            return float(clean)
        except:
            return 0
    
    def extract_items_universal(self, text_boxes: List[TextBox]) -> List[Dict[str, Any]]:
        """Universal item extraction - works with any invoice format"""
        
        # Step 1: Find ALL money values
        all_money = []
        for box in text_boxes:
            value = self._parse_money_safe(box.text)
            if value >= 0.01:  # Any reasonable amount
                all_money.append((value, box))
        
        if len(all_money) < 1:
            return []
        
        print(f"DEBUG: Found {len(all_money)} money values")
        
        # Step 2: Group money by rows
        money_rows = {}
        for value, box in all_money:
            y = box.bbox[1] + box.bbox[3]//2
            found_row = False
            for existing_y in money_rows:
                if abs(y - existing_y) <= 60:  # Same row
                    money_rows[existing_y].append((value, box))
                    found_row = True
                    break
            if not found_row:
                money_rows[y] = [(value, box)]
        
        # Step 3: Find quantity patterns
        qty_patterns = []
        for box in text_boxes:
            text = box.text.strip()
            y = box.bbox[1] + box.bbox[3]//2
            
            # Hours pattern
            qty_match = re.search(r'\b(\d{1,3})\s*(h|hr|hrs|hour|hours)\b', text.lower())
            if qty_match:
                qty_patterns.append((int(qty_match.group(1)), y, box))
                continue
            
            # Pure numbers (1-99)
            if re.match(r'^\d{1,2}$', text) and 1 <= int(text) <= 99:
                qty_patterns.append((int(text), y, box))
                continue
            
            # Qty: X pattern
            if re.search(r'(qty|quantity)[:\s]*(\d+)', text.lower()):
                match = re.search(r'(qty|quantity)[:\s]*(\d+)', text.lower())
                qty_patterns.append((int(match.group(2)), y, box))
        
        print(f"DEBUG: Found {len(qty_patterns)} quantity patterns")
        
        # Step 4: Extract items using multiple methods
        items = []
        
        # Method 1: Quantity-based matching
        for qty, qty_y, qty_box in qty_patterns:
            row_money = []
            for money_y, money_list in money_rows.items():
                if abs(money_y - qty_y) <= 100:
                    row_money.extend(money_list)
            
            if len(row_money) >= 2:
                # Find qty * unit_price = amount
                for i, (val1, box1) in enumerate(row_money):
                    for j, (val2, box2) in enumerate(row_money):
                        if i != j:
                            error1 = abs(qty * val1 - val2) / max(val2, 1)
                            error2 = abs(qty * val2 - val1) / max(val1, 1)
                            
                            if error1 < 0.15:
                                unit_price, amount = val1, val2
                            elif error2 < 0.15:
                                unit_price, amount = val2, val1
                            else:
                                continue
                            
                            desc = self._find_description(text_boxes, qty_y, qty_box.bbox[0])
                            
                            items.append({
                                'description': desc,
                                'quantity': qty,
                                'unit_price': f"{unit_price:.2f}",
                                'amount': f"{amount:.2f}",
                                'confidence': 0.9
                            })
                            print(f"DEBUG: Method 1 - {qty} x {unit_price} = {amount} ({desc})")
                            break
        
        # Method 2: Infer quantities from money relationships
        if len(items) < len(money_rows) // 2:  # Not enough items found
            print("DEBUG: Method 2 - Inferring quantities...")
            
            for row_y, money_list in money_rows.items():
                if len(money_list) >= 2:
                    values = sorted([v for v, _ in money_list])
                    
                    # Try quantities 1-10
                    for test_qty in range(1, 11):
                        for i, unit_price in enumerate(values[:-1]):
                            for amount in values[i+1:]:
                                if abs(test_qty * unit_price - amount) / max(amount, 1) < 0.1:
                                    desc = self._find_description(text_boxes, row_y, 0)
                                    
                                    # Avoid duplicates
                                    if not any(item['description'] == desc for item in items):
                                        items.append({
                                            'description': desc,
                                            'quantity': test_qty,
                                            'unit_price': f"{unit_price:.2f}",
                                            'amount': f"{amount:.2f}",
                                            'confidence': 0.7
                                        })
                                        print(f"DEBUG: Method 2 - {test_qty} x {unit_price} = {amount} ({desc})")
                                    break
        
        # Method 3: Simple description + amount pairs
        if not items:
            print("DEBUG: Method 3 - Description + amount pairs...")
            
            for row_y, money_list in money_rows.items():
                if money_list:
                    amount = max(v for v, _ in money_list)
                    desc = self._find_description(text_boxes, row_y, 0)
                    
                    if len(desc) > 3 and amount > 1:
                        items.append({
                            'description': desc,
                            'quantity': 1,
                            'unit_price': f"{amount:.2f}",
                            'amount': f"{amount:.2f}",
                            'confidence': 0.5
                        })
                        print(f"DEBUG: Method 3 - {desc} = {amount}")
        
        return items[:10]  # Max 10 items
    
    def _find_description(self, text_boxes: List[TextBox], target_y: int, exclude_x: int) -> str:
        """Find description text near target Y coordinate"""
        desc_boxes = []
        
        for box in text_boxes:
            box_y = box.bbox[1] + box.bbox[3]//2
            text = box.text.strip()
            
            if (abs(box_y - target_y) <= 100 and
                len(text) > 1 and
                not re.match(r'^[\d.,]+$', text) and
                text.lower() not in ['qty', 'quantity', 'unit', 'price', 'amount', 'total', 'description'] and
                abs(box.bbox[0] - exclude_x) > 20):
                desc_boxes.append(box)
        
        if desc_boxes:
            desc_boxes.sort(key=lambda b: b.bbox[0])
            description = ' '.join(box.text.strip() for box in desc_boxes)
            return re.sub(r'\s+', ' ', description).strip()[:100]  # Limit length
        
        return "Service Item"
    
    def extract_header_info(self, text_boxes: List[TextBox]) -> Dict[str, str]:
        """Extract vendor, date, invoice number"""
        all_text = ' '.join(box.text for box in text_boxes)
        
        # Extract date
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
        ]
        
        date = ""
        for pattern in date_patterns:
            match = re.search(pattern, all_text)
            if match:
                date = match.group(1)
                break
        
        # Extract invoice number
        invoice_patterns = [
            r'(?i)invoice\s*#?\s*:?\s*([A-Z0-9-]+)',
            r'(?i)no\.?\s*:?\s*([A-Z0-9-]+)',
            r'\b([A-Z]{2,5}-?\d{3,})\b'
        ]
        
        invoice_number = ""
        for pattern in invoice_patterns:
            match = re.search(pattern, all_text)
            if match and len(match.group(1)) >= 3:
                invoice_number = match.group(1)
                break
        
        # Extract vendor (first meaningful line)
        vendor = ""
        for box in sorted(text_boxes, key=lambda b: b.bbox[1])[:10]:
            text = box.text.strip()
            if (len(text) > 3 and 
                not re.search(r'\d{3}', text) and
                text.lower() not in ['invoice', 'bill', 'receipt']):
                vendor = text
                break
        
        return {
            'vendor_name': vendor,
            'date': date,
            'invoice_number': invoice_number
        }
    
    def calculate_total(self, items: List[Dict[str, Any]]) -> str:
        """Calculate total from items"""
        if items:
            total = sum(float(item['amount']) for item in items)
            return f"{total:.2f}"
        return "0.00"
    
    def parse_invoice(self, image_path: str) -> Dict[str, Any]:
        """Main parsing function - handles any invoice format"""
        print(f"Parsing invoice: {image_path}")
        
        # Extract text
        text_boxes = self.extract_text_with_coordinates(image_path)
        print(f"Extracted {len(text_boxes)} text boxes")
        
        # Extract items
        items = self.extract_items_universal(text_boxes)
        print(f"Found {len(items)} items")
        
        # Extract header info
        header_info = self.extract_header_info(text_boxes)
        
        # Calculate total
        total = self.calculate_total(items)
        
        return {
            **header_info,
            'items': items,
            'total': total
        }

def main():
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python universal_parser.py <image_path>")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
    
    parser = UniversalInvoiceParser()
    
    try:
        result = parser.parse_invoice(image_path)
        
        print("\n" + "="*50)
        print("UNIVERSAL PARSER RESULT")
        print("="*50)
        print(json.dumps(result, indent=2))
        
        # Save result
        output_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_universal.json"
        with open(output_name, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {output_name}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()