#!/usr/bin/env python3
"""
Universal Layout Pipeline Example

This demonstrates how the universal approach works on different document types
without needing format-specific rules.
"""

from universal_layout_pipeline import UniversalLayoutPipeline
import json
import os

def demo_universal_pipeline():
    """Demonstrate the universal pipeline on different document types"""
    
    pipeline = UniversalLayoutPipeline()
    
    # Example schemas for different document types
    schemas = {
        'invoice': {
            'vendor_name': 'string',
            'date': 'string', 
            'items': [{'description': 'string', 'amount': 'number'}],
            'total': 'number'
        },
        'receipt': {
            'store_name': 'string',
            'date': 'string',
            'items': [{'item': 'string', 'price': 'number'}],
            'total': 'number'
        },
        'schedule': {
            'title': 'string',
            'date': 'string',
            'events': [{'time': 'string', 'event': 'string'}]
        }
    }
    
    print("Universal Layout Pipeline Demo")
    print("=" * 50)
    print("\nThis pipeline works on ANY document format by:")
    print("1. Extracting text + coordinates (OCR)")
    print("2. Detecting layout blocks (tables, columns, headers)")
    print("3. Processing each block type appropriately")
    print("4. Converting to your business JSON schema")
    print("\nSupported schemas:", list(schemas.keys()))
    
    # Check for sample images in current directory
    sample_files = [f for f in os.listdir('.') 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf'))]
    
    if not sample_files:
        print("\nNo sample images found in current directory.")
        print("Add some invoice/receipt images to test the pipeline.")
        return
    
    print(f"\nFound {len(sample_files)} sample files:")
    for i, f in enumerate(sample_files):
        print(f"{i+1}. {f}")
    
    # Process first file as example
    if sample_files:
        sample_file = sample_files[0]
        print(f"\nProcessing sample file: {sample_file}")
        
        try:
            # Process as invoice
            result = pipeline.process_document(sample_file, 'invoice')
            
            print("\nUniversal Pipeline Result:")
            print("-" * 30)
            print(json.dumps(result, indent=2))
            
            # Show how it adapts to different schemas
            print(f"\nThe same pipeline can convert to different schemas:")
            print("- Invoice schema: vendor_name, items[], total")
            print("- Receipt schema: store_name, items[], total") 
            print("- Schedule schema: title, events[], date")
            print("\nJust change the schema_type parameter!")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure you have the required OCR libraries installed:")
            print("pip install paddlepaddle paddleocr")
            print("pip install easyocr")
            print("pip install pytesseract")

def create_test_schemas():
    """Create example business schemas"""
    
    schemas = {
        'invoice_schema.json': {
            "type": "object",
            "properties": {
                "vendor_name": {"type": "string"},
                "invoice_number": {"type": "string"},
                "date": {"type": "string", "format": "date"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit_price": {"type": "number"},
                            "amount": {"type": "number"}
                        }
                    }
                },
                "subtotal": {"type": "number"},
                "tax": {"type": "number"},
                "total": {"type": "number"}
            }
        },
        
        'receipt_schema.json': {
            "type": "object", 
            "properties": {
                "store_name": {"type": "string"},
                "location": {"type": "string"},
                "date": {"type": "string"},
                "time": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item": {"type": "string"},
                            "price": {"type": "number"}
                        }
                    }
                },
                "subtotal": {"type": "number"},
                "tax": {"type": "number"},
                "total": {"type": "number"}
            }
        },
        
        'schedule_schema.json': {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "location": {"type": "string"},
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object", 
                        "properties": {
                            "time": {"type": "string"},
                            "event": {"type": "string"},
                            "speaker": {"type": "string"},
                            "room": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
    
    print("Creating example business schemas...")
    for filename, schema in schemas.items():
        with open(filename, 'w') as f:
            json.dump(schema, f, indent=2)
        print(f"Created: {filename}")

def show_pipeline_architecture():
    """Show the universal pipeline architecture"""
    
    print("\nUniversal Layout Pipeline Architecture")
    print("=" * 50)
    
    architecture = """
    INPUT: Any document image/PDF
           ↓
    ┌─────────────────────────────────────┐
    │ STEP 1: OCR + Coordinates           │
    │ • PaddleOCR / EasyOCR / Tesseract   │
    │ • Returns: text + bounding boxes    │
    └─────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────┐
    │ STEP 2: Layout Detection            │
    │ • Group text into blocks            │
    │ • Detect: tables, columns, headers  │
    │ • Universal - no format rules       │
    └─────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────┐
    │ STEP 3: Block Processing            │
    │ • Table → extract rows/columns      │
    │ • Column → read in order            │
    │ • Header/Footer → extract metadata  │
    └─────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────┐
    │ STEP 4: Schema Conversion           │
    │ • Map blocks to business JSON       │
    │ • Invoice / Receipt / Schedule      │
    │ • Extensible to any schema          │
    └─────────────────────────────────────┘
           ↓
    OUTPUT: Structured business JSON
    """
    
    print(architecture)
    
    print("\nKey Benefits:")
    print("• Works on ANY document layout")
    print("• No format-specific rules needed")
    print("• Handles tables, columns, multi-page")
    print("• Extensible to new document types")
    print("• Combines geometric + content analysis")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--schemas':
        create_test_schemas()
    elif len(sys.argv) > 1 and sys.argv[1] == '--architecture':
        show_pipeline_architecture()
    else:
        demo_universal_pipeline()