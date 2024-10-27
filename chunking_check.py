import json
from openai import OpenAI
from pathlib import Path
import logging

# Initialize OpenAI client
client = OpenAI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_chunk(chunk_text):
    """Check if the chunk is a complete, grammatically correct sentence using GPT-4o-mini."""
    prompt = f"""
    You are a linguistics expert. For the provided text, determine if it is a complete, grammatically correct sentence. Respond with "Yes" or "No" and provide a brief explanation if the answer is "No".

    Text: "{chunk_text}"
    """
    try:
        completion = client.beta.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        if completion and completion.choices:
            response = completion.choices[0].message.content.strip()
            return response
        else:
            return "No response from GPT-4o-mini."
    except Exception as e:
        logging.error(f"Error checking chunk: {str(e)}")
        return f"Error: {str(e)}"

def process_articles():
    """Process articles and check each chunk."""
    input_merged = "data/merged_articles.json"
    input_chunks = "data/chunked_articles_spacy.json"
    output_file = "data/chunked_articles_spacy_checked.json"

    # Check if input files exist
    if not Path(input_merged).is_file():
        logging.error(f"Input file {input_merged} does not exist.")
        return
    if not Path(input_chunks).is_file():
        logging.error(f"Input file {input_chunks} does not exist.")
        return

    # Load the merged articles
    try:
        with open(input_merged, "r", encoding="utf-8") as f:
            merged_articles = json.load(f)
    except Exception as e:
        logging.error(f"Error loading {input_merged}: {str(e)}")
        return

    # Load the chunked articles
    try:
        with open(input_chunks, "r", encoding="utf-8") as f:
            chunked_articles = json.load(f)
    except Exception as e:
        logging.error(f"Error loading {input_chunks}: {str(e)}")
        return

    # Dictionary to store the results
    checked_chunks = {}

    # Iterate over each article
    for file_name, chunks in chunked_articles.items():
        logging.info(f"Checking chunks for article: {file_name}")
        article_results = []

        # Iterate over each chunk
        for chunk in chunks:
            result = check_chunk(chunk)
            article_results.append({
                "chunk": chunk,
                "check_result": result
            })

        # Store the results for the article
        checked_chunks[file_name] = article_results

    # Save the checked chunks to a new JSON file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(checked_chunks, f, indent=4, ensure_ascii=False)
        logging.info(f"Successfully checked chunks and saved results to {output_file}")
    except Exception as e:
        logging.error(f"Error writing to {output_file}: {str(e)}")

if __name__ == "__main__":
    process_articles()
