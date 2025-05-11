"""Generates a random sample for manual labeling of cosmetic products and attributes in a sample of transcriptions."""

import pandas as pd
import os
import sys

# Add path to utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils

# Paths and parameters
SAMPLE_SIZE = 250
input_path = utils.SENTENCES_TRANSCRIPTIONS_FILE
output_path = os.path.join(utils.BERT_DATA_FOLDER, f"sample_for_labeling_{utils.get_timestamp()}.xlsx")
guide_path = os.path.join(utils.BERT_DATA_FOLDER, "labeling_guide.txt")


def generate_random_sample():
    """Generate a random sample for manual labeling."""
    print(f"Loading data from: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        df = pd.read_excel(input_path)
        total_rows = len(df)

        print(f"Data loaded: {total_rows} rows")
        print(f"Available columns: {', '.join(df.columns)}")

        # Sample selection with fallback for small datasets
        if SAMPLE_SIZE >= total_rows:
            print(f"Warning: Sample size ({SAMPLE_SIZE}) >= total rows ({total_rows})")
            sample = df.copy()
        else:
            print(f"Selecting random sample of {SAMPLE_SIZE} rows...")
            sample = df.sample(n=SAMPLE_SIZE, random_state=42)

        # Add annotation columns
        sample['products_detected'] = ""
        sample['attributes_detected'] = ""

        # Save sample
        sample.to_excel(output_path, index=False)
        print(f"Sample saved to: {output_path}")
        print(f"Sample size: {len(sample)} rows")

        # Preview
        print("\nSample preview:")
        print(sample.head(5)[['transcription', 'products_detected', 'attributes_detected']])

        generate_labeling_guide()
        return sample

    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def generate_labeling_guide():
    """Create labeling instructions for annotators."""
    guide_content = """
COSMETICS LABELING GUIDELINES

For each transcription we identify:

1. COSMETIC PRODUCTS:
- Format: "product_name (product_type)" e.g., "Genifit (skincare)"
- Separate multiple products with commas
- Common types: concealer, foundation, mascara, lipstick, serum, etc.
- Include brand names when mentioned

2. PRODUCT ATTRIBUTES:
- Format: comma-separated attributes e.g., "glowing, long-lasting"
- Common attributes: matte, dewy, pigmented, hydrating, etc.
- Include descriptive adjectives and benefits

EXAMPLES:

| Transcription                      | Products Detected                | Attributes Detected     |
|-----------------------------------|----------------------------------|--------------------------|
| "I use the Genifit from Lancôme"  | "Genifit (skincare), Lancôme"    | ""                       |
| "This concealer is perfect"       | "concealer (makeup)"             | "perfect"                |
| "A foundation that doesn't crease"| "foundation (makeup)"            | "no creasing"            |

NOTES:
- Leave cells empty if no products/attributes are mentioned
- Consider context for implicit product references
- Focus on cosmetic-related content only
"""
    try:
        with open(guide_path, "w", encoding="utf-8") as f:
            f.write(guide_content)
        print(f"\nLabeling guide saved to: {guide_path}")
    except IOError as e:
        print(f"Error saving guide: {e}")

if __name__ == "__main__":
    print("Generating random sample for manual labeling...")
    generate_random_sample()
    print("\nProcess complete. You can now open the generated Excel file for manual annotation.")
