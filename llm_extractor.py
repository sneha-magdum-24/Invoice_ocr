#!/usr/bin/env python3

import json
import requests
from universal_layout_pipeline import UniversalLayoutPipeline

class LLMInvoiceExtractor:
    def __init__(self):
        self.pipeline = UniversalLayoutPipeline()
        self.ollama_url = "http://localhost:11434/api/generate"
    
    def extract_with_llm(self, image_path: str) -> dict:
        # Get OCR text + positions
        text_boxes = self.pipeline.extract_text_with_coordinates(image_path)
        
        # Format for LLM
        ocr_data = []
        for box in text_boxes:
            ocr_data.append({
                "text": box.text,
                "x": box.bbox[0],
                "y": box.bbox[1],
                "confidence": box.confidence
            })
        
        # LLM prompt
        prompt = f"""Extract invoice data from this OCR text with positions:

{json.dumps(ocr_data, indent=2)}

Return JSON with:
- vendor_name
- date  
- invoice_number
- items: [{{description, quantity, unit_price, amount}}]
- total

Focus on:
1. Parse amounts correctly (65,200.00 = 65200, not 65.2)
2. Find line items with qty + unit price + total
3. Use spatial positions to group related data

JSON:"""

        # Call Mistral
        response = requests.post(self.ollama_url, json={
            "model": "mistral:7b",
            "prompt": prompt,
            "stream": False
        })
        
        result = response.json()["response"]
        
        # Parse JSON from response
        try:
            start = result.find('{')
            end = result.rfind('}') + 1
            return json.loads(result[start:end])
        except:
            return {"error": "Failed to parse LLM response", "raw": result}

def main():
    extractor = LLMInvoiceExtractor()
    result = extractor.extract_with_llm("Invoice_ocr/IMG_0185.jpg")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()