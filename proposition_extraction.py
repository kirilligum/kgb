import json
import logging
from openai import OpenAI
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
client = OpenAI()


class Proposition(BaseModel):
    proposition: str


def extract_proposition(entity1, relation, entity2, sentence):
    """Extract a proposition using OpenAI API"""
    # proposition_definition = """
    # A proposition is a natural language sentence or statement that expresses a single, clear, and self-contained idea, fact, or claim about entities and their relationship. It conveys semantic meaning about a specific situation, event, or attribute of entities, enabling it to function independently as a fact in a knowledge graph or as a response to a query.
    # """
    prompt = (
        f"Create a clear and standalone proposition using the following information:\n\n"
        f"1. Entity 1: {entity1}\n"
        f"2. Relation: {relation}\n"
        f"3. Entity 2: {entity2}\n"
        f'4. Sentence: "{sentence}"\n\n'
        f"The proposition should:\n"
        f"- Focus on describing the relationship between {entity1} and {entity2}.\n"
        f"- Use any relevant details from the sentence that add clarity, specificity, or context to the relationship.\n"
        f"- Be a concise and self-contained statement that conveys all essential information for understanding this relationship.\n\n"
        # f"## Definition of proposition: \n{proposition_definition}"
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
    logging.info(
        "Loading decontextualized articles from data/decontextualized_articles.json"
    )
    with open("data/decontextualized_articles.json", "r", encoding="utf-8") as f:
        decontextualized_articles = json.load(f)

    logging.info(
        "Loading extracted relationships from data/extracted_relationships.json"
    )
    with open("data/extracted_relationships.json", "r", encoding="utf-8") as f:
        extracted_relationships = json.load(f)

    all_propositions = {}

    total_sentences = sum(len(sentences) for sentences in decontextualized_articles.values())
    total_propositions = 0
    processed_sentences = 0

    logging.info(f"Processed {processed_sentences}/{total_sentences} sentences.")
    logging.info(f"Extracted {total_propositions} propositions.")

    for file_name, sentences_list in decontextualized_articles.items():
        processed_sentences += len(sentences_list)
        logging.info(f"Processing article: {file_name}")
        relationships_list = extracted_relationships.get(file_name, [])

        article_propositions = []

        for sentence_index, sentence in enumerate(sentences_list):
            relationships = (
                relationships_list[sentence_index]
                if sentence_index < len(relationships_list)
                else []
            )
            for entity1, relation, entity2 in relationships:
                logging.info(
                    f"Extracting proposition for relationship: {entity1} {relation} {entity2}"
                )
                proposition = extract_proposition(entity1, relation, entity2, sentence)
                if proposition:
                    article_propositions.append(proposition.proposition)
                    total_propositions += 1

        all_propositions[file_name] = article_propositions

    with open("data/extracted_propositions.json", "w", encoding="utf-8") as f:
        json.dump(all_propositions, f, indent=4, ensure_ascii=False)
    logging.info(
        "Successfully extracted propositions into data/extracted_propositions.json"
    )


if __name__ == "__main__":
    process_articles()
