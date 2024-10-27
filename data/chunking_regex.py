import json
import re
from pathlib import Path

def chunk_sentences(article_text):
    """Chunk article text into sentences using regex."""
    # Regex pattern to split text into sentences
    # Improved regex pattern to handle abbreviations, ellipses, and quotes
    sentence_pattern = r'(?<!\b(?:Mr|Mrs|Ms|Dr|Inc|Ltd|Jr|Sr|St|Co|U\.S|U\.K|e\.g|i\.e|etc)\.)(?<!\.\.\.)(?<=\.|\?|!|;|:|”|’|\))\s+'
    sentences = re.split(sentence_pattern, article_text)
    return sentences

def process_articles(input_file, output_file):
    """Process articles and chunk them into sentences."""
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    chunked_articles = {}

    for file_name, article in articles.items():
        print(f"Chunking article: {file_name}")
        body_text = article.get("body_text", "")

        if body_text:
            sentences = chunk_sentences(body_text)
            chunked_articles[file_name] = sentences

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunked_articles, f, indent=4, ensure_ascii=False)
    print(f"Successfully chunked articles into {output_file}")

if __name__ == "__main__":
    input_file = "data/merged_articles.json"
    output_file = "data/chunked_articles_regex.json"

    process_articles(input_file, output_file)
