import os
import json
from pathlib import Path


def merge_cleaned_articles(input_dir, output_file):
    """
    Merge all cleaned JSON article files into a single JSON file.
    Articles will be stored in an array.
    """
    merged_data = {}
    input_path = Path(input_dir)

    # Iterate through all JSON files in the input directory
    for json_file in input_path.glob("*_cleaned.json"):
        try:
            # Read the content of each JSON file
            with open(json_file, "r", encoding="utf-8") as f:
                article_data = json.load(f)

                # Iterate over each article in the article_data
                for key, data in article_data.items():
                    # Only add articles where article_found is true
                    if data.get("article_found", False):
                        merged_data[key] = data

        except json.JSONDecodeError as e:
            print(f"Error reading {json_file.name}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error processing {json_file.name}: {str(e)}")

    # Write the merged JSON data to the output file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully merged {len(merged_data)} articles into {output_file}")
    except Exception as e:
        print(f"Error writing merged file: {str(e)}")


if __name__ == "__main__":
    input_directory = "projects/prls/cleaned_articles"
    output_file = "projects/prls/merged_articles.json"

    merge_cleaned_articles(input_directory, output_file)
