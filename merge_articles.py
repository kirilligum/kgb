import os
import json
from pathlib import Path

def merge_cleaned_articles(input_dir, output_file):
    """Merge all cleaned article JSON files into a single JSON file"""
    merged_data = {}
    input_path = Path(input_dir)

    # Iterate through all JSON files in the input directory
    for json_file in input_path.glob("*.json"):
        print(f"Processing {json_file.name}")
        
        # Read the content of each JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                json_data = json.load(f)
                # Use the filename (without extension) as the key
                merged_data[json_file.stem] = json_data
            except json.JSONDecodeError as e:
                print(f"Error reading {json_file.name}: {str(e)}")
                continue

    # Write the merged JSON data to the output file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
    
    print(f"\nMerged {len(merged_data)} articles into {output_file}")

if __name__ == "__main__":
    input_directory = "data/cleaned_articles"
    output_file = "data/merged_articles.json"
    
    merge_cleaned_articles(input_directory, output_file)
