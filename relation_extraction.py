import json
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from pydantic import BaseModel

# Initialize OpenAI client
client = OpenAI()


class RelationshipValidation(BaseModel):
    is_valid: bool


class Relationship(BaseModel):
    entity1: str
    relation: str
    entity2: str


def extract_and_validate_relationships(original_text, paraphrased_text, entities, sentence_index, total_sentences):
    logging.info("Starting relationship extraction and validation")
    relationships = []

    # Loop over each entity pair (entity1, entity2)
    for entity1 in entities:
        for entity2 in entities:
            if entity1 != entity2:
                # Step 3A: Extract candidate relationship from original text
                relationship_format = Relationship.schema_json(indent=2)
                extract_prompt = (
                    f"You are an expert in Natural Language Processing techniques. You are doing relation extraction. you are give a text (below) and two entities:\"{entity1['entity']}\" and \"{entity2['entity']}\". As a first step, write an explaination of the relationship between the entities according to the text."
                    f"Next, identify the relationship between \"{entity1['entity']}\" and \"{entity2['entity']}\"."
                    f"## the text:\n \"{original_text}\". "
                    f"## Output format:\n json {relationship_format} ."
                )
                logging.info(f"Extracting relationship between {entity1['entity']} and {entity2['entity']} in sentence {sentence_index}/{total_sentences}")
                response = client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Your job is to do an NLP task 'relation extraction' between given entities according to a text.",
                        },
                        {
                            "role": "user",
                            "content": extract_prompt,
                        },
                    ],
                    response_format=Relationship,
                )
                logging.info("Received response for relationship extraction")
                candidate_relation_response = response.choices[0].message.parsed if response and response.choices else None
                if candidate_relation_response:
                    candidate_relation = candidate_relation_response.relation
                else:
                    candidate_relation = ""

                # Proceed only if a relationship was identified
                if candidate_relation:
                    # Step 3B: Validate the relationship in the paraphrased text
                    validate_prompt = (
                        f"In the paraphrased text: \"{paraphrased_text}\", "
                        f"is the relationship \"{entity1['entity']} {candidate_relation} {entity2['entity']}\" correct? (true/false)"
                    )
                    logging.info(f"Validating relationship: {entity1['entity']} {candidate_relation} {entity2['entity']}")
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
                    logging.info("Received response for relationship validation")
                    validation = None
                    if response and response.choices:
                        parsed_response = response.choices[0].message.parsed
                        if parsed_response:
                            validation = parsed_response.is_valid

                    # If validated, add to the final relationships
                    if validation:
                        relationships.append((
                            entity1["entity"],
                            candidate_relation,
                            entity2["entity"],
                        ))

    logging.info("Completed relationship extraction and validation")
    return relationships


def process_articles():
    # Load data from JSON files
    logging.info("Loading decontextualized articles from data/decontextualized_articles.json")
    with open("data/decontextualized_articles.json", "r", encoding="utf-8") as f:
        decontextualized_articles = json.load(f)

    logging.info("Loading paraphrased articles from data/paraphrased_articles.json")
    with open("data/paraphrased_articles.json", "r", encoding="utf-8") as f:
        paraphrased_articles = json.load(f)

    logging.info("Loading extracted entities from data/extracted_entities.json")
    with open("data/extracted_entities.json", "r", encoding="utf-8") as f:
        extracted_entities = json.load(f)

    all_relationships = {}

    for file_index, (file_name, sentences_list) in enumerate(decontextualized_articles.items(), start=1):
        logging.info(f"Processing article {file_index}/{len(decontextualized_articles)}: {file_name}")
        paraphrased_sentences = paraphrased_articles.get(file_name, {}).get(
            "paraphrased_sentences", []
        )
        entities_list = extracted_entities.get(file_name, [])

        if len(sentences_list) != len(paraphrased_sentences) or len(sentences_list) != len(entities_list):
            logging.warning(f"Mismatch in number of sentences and entities for article: {file_name}")
            continue

        article_relationships = []

        for sentence_index, original_sentence in enumerate(sentences_list, start=1):
            logging.info(f"Processing sentence {sentence_index}/{len(sentences_list)} in article: {file_name}")
            paraphrased_sentence = paraphrased_sentences[sentence_index - 1]
            entities = entities_list[sentence_index - 1]
            relationships = extract_and_validate_relationships(
                original_sentence, paraphrased_sentence, entities, sentence_index, len(sentences_list)
            )
            article_relationships.extend(relationships)

        all_relationships[file_name] = article_relationships

    # Save the relationships to a JSON file
    logging.info("Saving extracted relationships to data/extracted_relationships.json")
    with open("data/extracted_relationships.json", "w", encoding="utf-8") as f:
        json.dump(all_relationships, f, indent=4, ensure_ascii=False)
    print(
        "Successfully extracted and validated relationships into data/extracted_relationships.json"
    )


if __name__ == "__main__":
    process_articles()
