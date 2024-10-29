import json
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter

def process_articles(input_file, output_file):
    """Process articles and split them into sentences using Flair"""
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Initialize sentence splitter
    splitter = SegtokSentenceSplitter()

    # Use splitter to split text into list of sentences
    sentences = splitter.split(text)

    # Predict tags for sentences
    tagger = Classifier.load('ner')
    tagger.predict(sentences)

    # Store sentences in a dictionary
    chunked_articles = {input_file: [sentence.to_plain_string() for sentence in sentences]}

    # Write the chunked sentences to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunked_articles, f, indent=4, ensure_ascii=False)
    print(f"Successfully chunked articles into {output_file}")

if __name__ == "__main__":
    input_file = "projects/prls/inputs/inputted_facts_2024-10-29T02:39:37-07:00.txt"
    output_file = "projects/prls/chunked_articles_flair.json"
    process_articles(input_file, output_file)
