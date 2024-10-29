import json
import logging
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter


def split_text_into_sentences(input_file, output_file):
    """Process articles and split them into sentences using Flair"""
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Initialize sentence splitter
    splitter = SegtokSentenceSplitter()

    # Use splitter to split text into list of sentences
    sentences = [sentence for sentence in splitter.split(text) if sentence.to_plain_string().strip()]

    logging.info(f"Split text into {len(sentences)} sentences.")
    chunked_articles = {
        input_file: [sentence.to_plain_string() for sentence in sentences]
    }
    logging.info(f"Chunked {len(chunked_articles[input_file])} sentences from input file: {input_file}")

    # Write the chunked sentences to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunked_articles, f, indent=2, ensure_ascii=False)
    print(f"Successfully chunked articles into {output_file}")


if __name__ == "__main__":
    input_file = "projects/prls/inputs/inputted_facts_2024-10-29T02:39:37-07:00.txt"
    output_file = "projects/prls/chunked_articles_flair.json"
    split_text_into_sentences(input_file, output_file)
