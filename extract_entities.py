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
                    "content": f"Extract named entities from the following text:\n\n{article_text}",
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

    for article in articles[:2]:
        file_name = article.get("file_name")
        print(f"Extracting entities from article: {file_name}")
        # body_text = article.get("body_text", "")
        body_text = json.dumps(article, indent=4)

        if body_text:
            entities = extract_entities_from_article(body_text)
            if entities:
                extracted_entities[file_name] = entities

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_entities, f, indent=4, ensure_ascii=False)
    print(f"Successfully extracted entities into {output_file}")


if __name__ == "__main__":
    input_file = "data/merged_articles.json"
    output_file = "data/extracted_entities.json"

    process_articles(input_file, output_file)
