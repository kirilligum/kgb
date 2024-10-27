import json
from openai import OpenAI
from pydantic import BaseModel
from pathlib import Path

# Initialize OpenAI client
client = OpenAI()


class NamedEntity(BaseModel):
    entity: str
    type: str


class NamedEntities(BaseModel):
    entities: list[NamedEntity]


def extract_entities_from_article(article_text):
    """Extract named entities using OpenAI API"""
    common_ner_types = """
    <common_ner>
    Commonly Used Entity Types:
        Person (PER): Individuals' names.
        Organization (ORG): Companies, institutions, government agencies.
        Location (LOC): Geographical entities, cities, countries, addresses.   
        Date (DATE): Temporal expressions, including specific dates, days of the week, months.
        Time (TIME): Time expressions, like specific times, durations.   
        Money (MONEY): Monetary values, currencies.   
        Percentage (PERCENT): Numerical values representing percentages.   
        GPE (Geopolitical Entity): Countries, cities, states. This sometimes overlaps with LOC.   
        FAC (Facility): Buildings, airports, bridges, highways.
        Product (PROD): Names of products.
        Event (EVT): Named events like wars, conferences, festivals.
        Work of Art (WOA): Titles of books, songs, movies.
        Law (LAW): Names of legal documents and acts.
        Language (LANGUAGE): Names of languages.
    </common_ner>
    """
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts named entities from text.",
                },
                {
                    "role": "user",
                    "content": f"Read the article text carfully. Extract named entities from the following article text:\n\n<article_text>{article_text}</article_text>\n\n use the short form of common NER types but also add new ones when you need to. {common_ner_types}",
                },
            ],
            response_format=NamedEntities,
        )

        if completion and completion.choices:
            entities = completion.choices[0].message.parsed
            return entities

        print("No valid response received from OpenAI API")
        return None

    except Exception as e:
        print(f"Error extracting entities: {str(e)}")
        return None


def process_articles(input_file, output_file):
    """Process articles and extract named entities"""
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    extracted_entities = {}

    for file_name, sentences_list in list(articles.items())[:2]:
        print(f"Extracting entities from article: {file_name}")
        sentence_entities = []

        for sentences in sentences_list:
            for sentence in sentences:
                entities = extract_entities_from_article(sentence)
                if entities:
                    sentence_entities.append([
                        {"entity": entity.entity, "type": entity.type} for entity in entities.entities
                    ])

        extracted_entities[file_name] = sentence_entities

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_entities, f, indent=4, ensure_ascii=False)
    print(f"Successfully extracted entities into {output_file}")


if __name__ == "__main__":
    input_file = "data/decontextualized_articles.json"
    output_file = "data/extracted_entities.json"

    process_articles(input_file, output_file)
