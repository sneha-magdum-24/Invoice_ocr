#!/usr/bin/env python3
"""
Simple invoice extractor using MinerU-style approach
Combines preprocessing + PaddleOCR + Llama 3.2
"""

import cv2
import numpy as np
import json
import requests
from paddleocr import PaddleOCR
import sys

class SimpleInvoiceExtractor:
    def __init__(self):
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    
    def preprocess(self, img_path):
        """Enhanced preprocessing"""
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return thresh
    
    def extract_text(self, img_path):
        """Extract text with PaddleOCR"""
        processed = self.preprocess(img_path)
        result = self.ocr.ocr(processed, cls=True)
        
        if not result or not result[0]:
            return ""
        
        lines = [line[1][0] for line in result[0] if line and line[1]]
        return "\n".join(lines)
    
    def extract_with_llm(self, text):
        """Extract structured data with Llama 3.2"""
        prompt = f"""Extract invoice data as valid JSON only. No markdown, no comments.

Schema:
{{
  "vendor_name": null,
  "date": null,
  "subtotal": null,
  "tax_amount": null,
  "discount_amount": null,
  "total": null,
  "items": [{{"item_name": null, "unit_price": null, "quantity": null, "amount": null}}]
}}

Rules:
1. Extract ALL service lines with prices
2. Clean item names (no dollar signs or prices in item_name)
3. Use exact subtotal/total from invoice
4. Discount is NOT an item

Invoice:
{text}

JSON:"""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:latest",
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.0, "num_predict": 2000}
                },
                timeout=90
            )
            response.raise_for_status()
            return json.loads(response.json()['response'])
        except Exception as e:
            print(f"LLM error: {e}")
            return None
    
    def process(self, img_path):
        """Main pipeline"""
        print(f"Processing: {img_path}")
        
        text = self.extract_text(img_path)
        if not text:
            print("No text extracted")
            return None
        
        print(f"Extracted {len(text)} chars")
        
        result = self.extract_with_llm(text)
        if result:
            print("✓ Success")
            return result
        
        print("✗ Failed")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python mineru_simple.py <image_path>")
        return
    
    extractor = SimpleInvoiceExtractor()
    result = extractor.process(sys.argv[1])
    
    if result:
        print("\n" + "="*60)
        print(json.dumps(result, indent=2))
        
        output = sys.argv[1].rsplit('.', 1)[0] + '_extracted.json'
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {output}")

if __name__ == "__main__":
    main()
