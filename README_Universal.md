# Universal OCR Layout Pipeline

A universal approach to document OCR that works on any layout by understanding document structure rather than relying on format-specific rules.

## The Universal Approach

Instead of writing rules for every document format, this pipeline:

1. **Extracts text + coordinates** using OCR
2. **Detects layout blocks** (tables, columns, headers, footers)
3. **Processes each block type** appropriately
4. **Converts to business JSON** using your schema

This works on invoices, receipts, schedules, forms, or any document type.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process any document
python universal_layout_pipeline.py invoice.jpg invoice

# See demo
python demo_universal.py

# View architecture
python demo_universal.py --architecture
```

## How It Works

### Step 1: OCR + Coordinates
```python
# Get text with bounding boxes
text_boxes = pipeline.extract_text_with_coordinates("document.jpg")
# Returns: [TextBox(text="Invoice", bbox=(10, 20, 100, 30), confidence=0.95)]
```

### Step 2: Layout Detection
```python
# Detect layout blocks automatically
blocks = pipeline.detect_layout_blocks(text_boxes, image_shape)
# Returns: [LayoutBlock(type="table", bbox=..., text_boxes=[...])]
```

### Step 3: Block Processing
```python
# Handle each block type differently
if block.type == "table":
    extract_rows_and_columns(block)
elif block.type == "column":
    read_in_correct_order(block)
```

### Step 4: Schema Conversion
```python
# Convert to your business JSON
result = pipeline.convert_to_business_json(blocks, schema_type="invoice")
```

## Supported Document Types

- **Invoices**: Any format, any layout
- **Receipts**: Store receipts, restaurant bills
- **Schedules**: Conference programs, meeting agendas
- **Forms**: Applications, surveys
- **Tables**: Financial reports, data sheets

## Layout Block Types

The pipeline automatically detects:

- **Tables**: Structured data in rows/columns
- **Multi-column text**: Newspapers, brochures
- **Headers/Footers**: Company info, page numbers
- **Paragraphs**: Regular text blocks
- **Figures**: Images, logos (detected but not processed)

## Business Schemas

Define your output format:

```python
invoice_schema = {
    "vendor_name": "string",
    "date": "string",
    "items": [{"description": "string", "amount": "number"}],
    "total": "number"
}

receipt_schema = {
    "store_name": "string", 
    "items": [{"item": "string", "price": "number"}],
    "total": "number"
}
```

## Example Usage

```python
from universal_layout_pipeline import UniversalLayoutPipeline

pipeline = UniversalLayoutPipeline()

# Process any document type
result = pipeline.process_document("document.jpg", schema_type="invoice")

print(json.dumps(result, indent=2))
```

## OCR Engines

The pipeline tries multiple OCR engines in order:

1. **PaddleOCR** (best for structured documents)
2. **EasyOCR** (good general purpose)
3. **Tesseract** (fallback)

## Advanced Features

### Custom Schemas
```python
# Add your own document type
def convert_to_custom_schema(self, blocks):
    return {
        "custom_field": extract_from_blocks(blocks),
        "items": process_tables(blocks["tables"])
    }
```

### Layout Validation
```python
# Validate extracted data
def validate_result(result):
    if not result.get("total"):
        flag_for_human_review(result)
```

## Benefits

- **Universal**: Works on any document layout
- **No Rules**: No format-specific programming needed
- **Extensible**: Easy to add new document types
- **Robust**: Handles tables, columns, multi-page documents
- **Accurate**: Combines geometric and content analysis

## Comparison

| Approach | Format-Specific Rules | Universal Pipeline |
|----------|----------------------|-------------------|
| New Format | Write new rules | Works automatically |
| Maintenance | High (many rules) | Low (one pipeline) |
| Accuracy | Good for known formats | Good for all formats |
| Development | Weeks per format | Days for any format |

## Files

- `universal_layout_pipeline.py` - Main pipeline implementation
- `demo_universal.py` - Example usage and demos
- `requirements.txt` - Dependencies
- `*_schema.json` - Example business schemas

## Dependencies

- OpenCV (image processing)
- PaddleOCR (best OCR for layouts)
- EasyOCR (backup OCR)
- Tesseract (fallback OCR)
- NumPy (array processing)

## Future Enhancements

- Integration with LayoutParser for advanced layout detection
- Table Transformer models for complex table structures
- Multi-page document support
- Confidence scoring and validation
- Interactive correction interface