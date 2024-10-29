import json
import logging
from flair.data import Sentence
from flair.nn import Classifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load the entity linker
linker = Classifier.load("linker")


def extract_linked_entities_with_flair(article_text):
    """Extract linked entities using Flair entity linker"""
    sentence = Sentence(article_text)
    linker.predict(sentence)
    linked_entities = [
        {"entity": entity.text, "link": entity.get_label("link").value}
        for entity in sentence.get_spans("link")
    ]
    return linked_entities


def process_articles(input_file, output_file):
    """Process articles and extract linked entities using Flair"""
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    extracted_linked_entities = {}

    total_sentences = sum(len(sentences) for sentences in articles.values())
    processed_sentences = 0

    for file_name, sentences_list in articles.items():
        processed_sentences += len(sentences_list)
        logging.info(f"Extracting linked entities from article: {file_name}")
        sentence_linked_entities = []

        for i, sentence in enumerate(sentences_list):
            logging.info(
                f"Extracting linked entities from sentence {i + 1}/{len(sentences_list)} in article: {file_name}"
            )
            linked_entities = extract_linked_entities_with_flair(sentence)
            sentence_linked_entities.append(linked_entities)

        extracted_linked_entities[file_name] = sentence_linked_entities
        logging.info(f"Processed {processed_sentences}/{total_sentences} sentences.")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_linked_entities, f, indent=4, ensure_ascii=False)
    logging.info(f"Successfully extracted linked entities into {output_file}")


if __name__ == "__main__":
    input_file = "projects/prls/decontextualized_articles.json"
    output_file = "projects/prls/extracted_linked_entities.json"

    process_articles(input_file, output_file)
