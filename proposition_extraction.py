import json
import logging
from openai import OpenAI
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
client = OpenAI()


class Proposition(BaseModel):
    proposition: str


def extract_proposition(entity1, relation, entity2, sentence):
    """Extract a proposition using OpenAI API"""
    prompt = (
        f"Given the following information:\n\n"
        f"1. Entity 1: {entity1}\n"
        f"2. Relation: {relation}\n"
        f"3. Entity 2: {entity2}\n"
        f"4. Sentence: \"{sentence}\"\n\n"
        f"Write a single, clear and self-contained sentence (a proposition) that combines this information, "
        f"clearly describing the relationship between {entity1} and {entity2}.\n\n"
        f"Proposition:"
    )

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts propositions from given information.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format=Proposition,
        )

        if completion and completion.choices:
            proposition = completion.choices[0].message.parsed
            return proposition

        logging.warning("No valid response received from OpenAI API")
        return None

    except Exception as e:
        logging.error(f"Error extracting proposition: {str(e)}")
        return None


def process_articles():
    """Process articles and extract propositions"""
    logging.info("Loading decontextualized articles from data/decontextualized_articles.json")
    with open("data/decontextualized_articles.json", "r", encoding="utf-8") as f:
        decontextualized_articles = json.load(f)

    logging.info("Loading extracted relationships from data/extracted_relationships.json")
    with open("data/extracted_relationships.json", "r", encoding="utf-8") as f:
        extracted_relationships = json.load(f)

    all_propositions = {}

    for file_name, sentences_list in decontextualized_articles.items():
        logging.info(f"Processing article: {file_name}")
        relationships_list = extracted_relationships.get(file_name, [])

        article_propositions = []

        for sentence_index, sentence in enumerate(sentences_list):
            relationships = relationships_list[sentence_index] if sentence_index < len(relationships_list) else []
            for entity1, relation, entity2 in relationships:
                logging.info(f"Extracting proposition for relationship: {entity1} {relation} {entity2}")
                proposition = extract_proposition(entity1, relation, entity2, sentence)
                if proposition:
                    article_propositions.append(proposition.proposition)

        all_propositions[file_name] = article_propositions

    with open("data/extracted_propositions.json", "w", encoding="utf-8") as f:
        json.dump(all_propositions, f, indent=4, ensure_ascii=False)
    logging.info("Successfully extracted propositions into data/extracted_propositions.json")


if __name__ == "__main__":
    process_articles()
