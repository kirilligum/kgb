import json
from flair.data import Sentence
from flair.models import SequenceTagger

def chunk_sentences_with_flair(article_text):
    """Chunk article text into sentences using Flair."""
    tagger = SequenceTagger.load("flair/chunk-english")
    sentence = Sentence(article_text)
    tagger.predict(sentence)
    return [span.text for span in sentence.get_spans("chunk")]

def process_articles(input_file, output_file):
    """Process articles and chunk them into sentences using Flair."""
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    chunked_articles = {}

    for file_name, article in articles.items():
        print(f"Chunking article with Flair: {file_name}")
        body_text = article.get("body_text", "")

        if body_text:
            sentences = chunk_sentences_with_flair(body_text)
            chunked_articles[file_name] = sentences

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunked_articles, f, indent=4, ensure_ascii=False)
    print(f"Successfully chunked articles into {output_file}")

if __name__ == "__main__":
    input_file = "data/merged_articles.json"
    output_file = "data/chunked_articles_flair.json"

    process_articles(input_file, output_file)
