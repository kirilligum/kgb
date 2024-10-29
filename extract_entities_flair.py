import json
import logging
from flair.data import Sentence
from flair.nn import Classifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load the NER tagger
tagger = Classifier.load("ner-ontonotes-large")


def extract_entities_with_flair(article_text):
    """Extract named entities using Flair NER"""
    sentence = Sentence(article_text)
    tagger.predict(sentence)
    entities = [
        {"entity": entity.text, "type": entity.get_label("ner").value}
        for entity in sentence.get_spans("ner")
    ]
    return entities


def process_articles(input_file, output_file):
    """Process articles and extract named entities using Flair"""
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    extracted_entities = {}

    total_sentences = sum(len(sentences) for sentences in articles.values())
    processed_sentences = 0

    for file_name, sentences_list in articles.items():
        processed_sentences += len(sentences_list)
        logging.info(f"Extracting entities from article: {file_name}")
        sentence_entities = []

        for i, sentence in enumerate(sentences_list):
            logging.info(
                f"Extracting entities from sentence {i + 1}/{len(sentences_list)} in article: {file_name}"
            )
            entities = extract_entities_with_flair(sentence)
            sentence_entities.append(entities)

        extracted_entities[file_name] = sentence_entities
        logging.info(f"Processed {processed_sentences}/{total_sentences} sentences.")
        logging.info(f"Extracted entities for {len(sentence_entities)} sentences in article: {file_name}")
        logging.info(f"Extracted entities for {len(sentence_entities)} sentences in article: {file_name}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_entities, f, indent=4, ensure_ascii=False)
    logging.info(f"Successfully extracted entities into {output_file}")


if __name__ == "__main__":
    input_file = "projects/prls/decontextualized_articles.json"
    output_file = "projects/prls/extracted_entities.json"

    process_articles(input_file, output_file)
