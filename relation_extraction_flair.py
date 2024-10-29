import json
import logging
from flair.data import Sentence
from flair.nn import Classifier
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize OpenAI client
client = OpenAI()

# Load the Flair relation extractor
relation_extractor = Classifier.load('relations')


def extract_and_validate_relationships(
    original_text, paraphrased_text, entities, sentence_index, total_sentences
):
    logging.info("Starting relationship extraction and validation")
    relationships = []

    # Use Flair to extract relationships
    sentence = Sentence(original_text)
    relation_extractor.predict(sentence)
    flair_relations = sentence.get_labels('relation')

    for relation in flair_relations:
        entity1, candidate_relation, entity2 = relation.value.split()
        logging.info(
            f"Extracted relationship using Flair: {entity1} {candidate_relation} {entity2}"
        )

        # Validate the relationship in the paraphrased text using OpenAI
        validate_prompt = (
            f"In the paraphrased text: \"{paraphrased_text}\", "
            f"is the relationship \"{entity1} {candidate_relation} {entity2}\" correct? (true/false)"
        )
        logging.info(
            f"Validating relationship: {entity1} {candidate_relation} {entity2}"
        )
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts and validates relationships between entities.",
                },
                {
                    "role": "user",
                    "content": validate_prompt,
                },
            ],
            response_format=RelationshipValidation,
        )
        validation = None
        if response and response.choices:
            parsed_response = response.choices[0].message.parsed
            if parsed_response:
                validation = parsed_response.is_valid

        # If validated, add to the final relationships
        if validation:
            relationships.append((
                entity1,
                candidate_relation,
                entity2,
            ))

    logging.info("Completed relationship extraction and validation")
    return relationships


def process_articles():
    # Load data from JSON files
    logging.info(
        "Loading decontextualized articles from projects/prls/decontextualized_articles.json"
    )
    with open("projects/prls/decontextualized_articles.json", "r", encoding="utf-8") as f:
        decontextualized_articles = json.load(f)

    logging.info("Loading paraphrased articles from projects/prls/paraphrased_articles.json")
    with open("projects/prls/paraphrased_articles.json", "r", encoding="utf-8") as f:
        paraphrased_articles = json.load(f)

    logging.info("Loading extracted entities from projects/prls/extracted_entities.json")
    with open("projects/prls/extracted_entities.json", "r", encoding="utf-8") as f:
        extracted_entities = json.load(f)

    all_relationships = {}

    for file_index, (file_name, sentences_list) in enumerate(
        decontextualized_articles.items(), start=1
    ):
        logging.info(
            f"Processing article {file_index}/{len(decontextualized_articles)}: {file_name}"
        )
        paraphrased_sentences = paraphrased_articles.get(file_name, {}).get(
            "paraphrased_sentences", []
        )
        entities_list = extracted_entities.get(file_name, [])

        if len(sentences_list) != len(paraphrased_sentences) or len(
            sentences_list
        ) != len(entities_list):
            logging.warning(
                f"Mismatch in number of sentences and entities for article: {file_name}"
            )
            continue

        article_relationships = []

        for sentence_index, original_sentence in enumerate(sentences_list[:2], start=1):
            entities = entities_list[sentence_index - 1]
            logging.info(
                f"Processing sentence {sentence_index}/{len(sentences_list)} in article: {file_name} with entities: {[entity['entity'] for entity in entities]}"
            )
            paraphrased_sentence = paraphrased_sentences[sentence_index - 1]
            entities = entities_list[sentence_index - 1]
            relationships = extract_and_validate_relationships(
                original_sentence,
                paraphrased_sentence,
                entities,
                sentence_index,
                len(sentences_list),
            )
            article_relationships.append(relationships)

        all_relationships[file_name] = article_relationships

    # Save the relationships to a JSON file
    logging.info("Saving extracted relationships to projects/prls/extracted_relationships.json")
    with open("projects/prls/extracted_relationships.json", "w", encoding="utf-8") as f:
        json.dump(all_relationships, f, indent=2, ensure_ascii=False)
    print(
        "Successfully extracted and validated relationships into data/extracted_relationships.json"
    )


if __name__ == "__main__":
    process_articles()
