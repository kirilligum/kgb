import json
import spacy

def chunk_sentences_with_spacy(article_text):
    """Chunk article text into sentences using spaCy."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(article_text)
    return [sent.text for sent in doc.sents]

def process_articles(input_file, output_file):
    """Process articles and chunk them into sentences using spaCy."""
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    chunked_articles = {}

    for file_name, article in list(articles.items())[:]:
        article["body_text"] = article.get("body_text", "").split('\n')[:]
        print(f"Chunking article with spaCy: {file_name}")
        body_text = article.get("body_text", "")

        if body_text:
            sentences = chunk_sentences_with_spacy(body_text)
            chunked_articles[file_name] = sentences

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunked_articles, f, indent=4, ensure_ascii=False)
    print(f"Successfully chunked articles into {output_file}")

if __name__ == "__main__":
    input_file = "projects/prls/merged_articles.json"
    output_file = "projects/prls/chunked_articles_spacy.json"

    process_articles(input_file, output_file)
