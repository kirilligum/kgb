import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

def extract_and_validate_relationships(original_text, paraphrased_text, entities):
    relationships = []

    # Loop over each entity pair (entity1, entity2)
    for entity1 in entities:
        for entity2 in entities:
            if entity1 != entity2:
                # Step 3A: Extract candidate relationship from original text
                extract_prompt = (f"Identify the relationship between \"{entity1['entity']}\" and \"{entity2['entity']}\" "
                                  f"in the text: \"{original_text}\". Possible relationships include "
                                  f"Lives In, Works For, and Located In.\nRelationship:")
                response = client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that extracts and validates relationships between entities.",
                        },
                        {
                            "role": "user",
                            "content": extract_prompt,
                        },
                    ],
                    response_format=None,
                )
                candidate_relation = response.choices[0].text.strip()

                # Proceed only if a relationship was identified
                if candidate_relation:
                    # Step 3B: Validate the relationship in the paraphrased text
                    validate_prompt = (f"In the paraphrased text: \"{paraphrased_text}\", "
                                       f"is the relationship \"{entity1['entity']} {candidate_relation} {entity2['entity']}\" correct? (Yes/No)")
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
                        response_format=None,
                    )
                    validation = response.choices[0].text.strip()

                    # If validated, add to the final relationships
                    if validation.lower() == 'yes':
                        relationships.append((entity1['entity'], candidate_relation, entity2['entity']))

    return relationships

def process_articles():
    # Load data from JSON files
    with open("data/decontextualized_articles.json", "r", encoding="utf-8") as f:
        decontextualized_articles = json.load(f)

    with open("data/paraphrased_articles.json", "r", encoding="utf-8") as f:
        paraphrased_articles = json.load(f)

    with open("data/extracted_entities.json", "r", encoding="utf-8") as f:
        extracted_entities = json.load(f)

    all_relationships = {}

    for file_name, sentences_list in decontextualized_articles.items():
        paraphrased_sentences = paraphrased_articles.get(file_name, {}).get("paraphrased_sentences", [])
        entities_list = extracted_entities.get(file_name, [])

        if len(sentences_list) != len(paraphrased_sentences) or len(sentences_list) != len(entities_list):
            print(f"Warning: Mismatch in number of sentences and entities for article: {file_name}")
            continue

        article_relationships = []

        for i, original_sentence in enumerate(sentences_list):
            paraphrased_sentence = paraphrased_sentences[i]
            entities = entities_list[i]
            relationships = extract_and_validate_relationships(original_sentence, paraphrased_sentence, entities)
            article_relationships.extend(relationships)

        all_relationships[file_name] = article_relationships

    # Save the relationships to a JSON file
    with open("data/extracted_relationships.json", "w", encoding="utf-8") as f:
        json.dump(all_relationships, f, indent=4, ensure_ascii=False)
    print("Successfully extracted and validated relationships into data/extracted_relationships.json")

if __name__ == "__main__":
    process_articles()
