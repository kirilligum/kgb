import os
import json
from openai import OpenAI
from pathlib import Path

# Initialize OpenAI client
client = OpenAI()

def clean_article(html_content):
    """Clean article using OpenAI API"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Note: Replace with actual gpt4o-mini model name when available
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts clean article text from HTML."
                },
                {
                    "role": "user",
                    "content": f"These are HTMLs of news articles, extract text related to the news article only. HTML content:\n\n{html_content}"
                }
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing article: {str(e)}")
        return None

def process_directory(input_dir, output_dir):
    """Process all HTML files in directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Process only first 2 articles using slice
    for html_file in list(input_path.glob('*.html'))[:2]:
            
        print(f"Processing {html_file.name}")
        
        # Read HTML content
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Clean article
        cleaned_text = clean_article(html_content)
        
        if cleaned_text:
            # Save cleaned text
            output_file = output_path / f"{html_file.stem}_cleaned.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            print(f"Saved cleaned text to {output_file}")
        else:
            print(f"Failed to clean {html_file.name}")

if __name__ == "__main__":
    input_directory = "data/elon_suing_openai"
    output_directory = "data/cleaned_articles"
    
    process_directory(input_directory, output_directory)
