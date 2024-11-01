import json
import logging
from openai import OpenAI
from pydantic import BaseModel
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
client = OpenAI()


class ParaphrasedArticle(BaseModel):
    file_name: str
    paraphrased_text: str


def paraphrase_article(article_text, entities):
    """Paraphrase article using OpenAI API"""
    few_shot_prompt = """
    Paraphrase the given text using the given entities.

    Given the text: {text}
    Use the entities {entities} to paraphrase the text. 
    Positive Examples:
    1.
    Original Text: "Apple Inc. unveiled its latest iPhone at a keynote event held in Cupertino, California." Entities: ["Apple Inc.:Org", "iPhone:Product", "Cupertino:Loc", "California:Loc"] Paraphrased Text: "At an event in Cupertino, California, Apple Inc. introduced the new iPhone." Explanation: The paraphrased text is more concise and rearranges the clauses to improve readability.
    2.
    Original Text: "The World Health Organization (WHO) issued a warning about the spread of the Zika virus in South America." Entities: ["World Health Organization:Org", "Zika virus:Disease", "South America:Loc"] Paraphrased Text: "The WHO cautioned about Zika virus transmission in South America." Explanation: The paraphrased text replaces the lengthy organization name with its acronym and simplifies the language.
    3.
    Original Text: "Elon Musk, the CEO of Tesla, announced plans to build a Gigafactory in Nevada." Entities: ["Elon Musk:Per", "Tesla:Org", "Gigafactory:Product", "Nevada:Loc"] Paraphrased Text: "Tesla's CEO, Elon Musk, revealed intentions to establish a Gigafactory in Nevada." Explanation: The paraphrased text rearranges the sentence to place emphasis on the entity "Tesla" and uses more formal language.
    4.
    Original Text: "The Mona Lisa, a portrait by Leonardo da Vinci, is on display at the Louvre Museum in Paris." Entities: ["Mona Lisa:Artwork", "Leonardo da Vinci:Per", "Louvre Museum:Org", "Paris:Loc"] Paraphrased Text: "Painted by Leonardo da Vinci, the Mona Lisa can be viewed at the Louvre Museum in Paris." Explanation: The paraphrased text clarifies the relationship between the artwork and the artist and uses passive voice to emphasize the artwork's location.
    5.
    Original Text: "The United Nations Security Council condemned the recent missile test by North Korea." Entities: ["United Nations Security Council:Org", "missile test:Event", "North Korea:Loc"] Paraphrased Text: "North Korea's latest missile test drew criticism from the UN Security Council." Explanation: The paraphrased text uses a more active voice to emphasize the actions of North Korea and shortens the organization name for brevity.
    Negative Examples:
    1.
    Original Text: "The cat sat on the mat." Entities: ["cat:Animal", "mat:Object"] Paraphrased Text: "The animal was positioned upon the object." Explanation: While grammatically correct, the paraphrased text loses the simplicity and clarity of the original.
    2.
    Original Text: "She went to the store to buy some milk." Entities: ["store:Loc", "milk:Product"] Paraphrased Text: "She visited the location for the purpose of acquiring the product." Explanation: The paraphrased text is overly verbose and uses unnecessarily formal language.
    3.
    Original Text: "The movie was so funny that I laughed out loud." Entities: ["movie:Entertainment"] Paraphrased Text: "The entertainment elicited such amusement that audible laughter ensued." Explanation: The paraphrased text is unnecessarily complex and uses overly sophisticated vocabulary.
    4.
    Original Text: "He told me a secret, but I promised not to tell anyone." Entities: ["secret:Information"] Paraphrased Text: "He confided sensitive information, but I pledged to maintain its confidentiality." Explanation: The paraphrased text, while maintaining the meaning, loses the natural and conversational tone of the original.
    5.
    Original Text: "The sun was shining brightly in the sky." Entities: ["sun:Celestial Body", "sky:Loc"] Paraphrased Text: "The celestial body radiated luminosity within the expanse of the location." Explanation: The paraphrased text is excessively formal and uses unnecessarily scientific terminology.
    """

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that paraphrases text using given entities.",
                },
                {
                    "role": "user",
                    "content": f"{few_shot_prompt}\n\nOriginal Text: {article_text}\nEntities: {entities}",
                },
            ],
            response_format=ParaphrasedArticle,
        )

        if completion and completion.choices:
            paraphrased = completion.choices[0].message.parsed
            return paraphrased

        logging.warning("No valid response received from OpenAI API")
        return None

    except Exception as e:
        logging.error(f"Error paraphrasing article: {str(e)}")
        return None


def process_articles(output_file):
    """Process articles and paraphrase them"""
    logging.info("Loading decontextualized articles from projects/prls/decontextualized_articles.json")
    with open("projects/prls/decontextualized_articles.json", "r", encoding="utf-8") as f:
        decontextualized_articles = json.load(f)

    logging.info("Loading extracted entities from projects/prls/extracted_entities.json")
    with open("projects/prls/extracted_entities.json", "r", encoding="utf-8") as f:
        extracted_entities = json.load(f)

    paraphrased_articles = {}

    total_sentences = sum(len(sentences) for sentences in decontextualized_articles.values())
    processed_sentences = 0

    for file_name, sentences_list in list(decontextualized_articles.items())[:]:
        processed_sentences += len(sentences_list)
        sentences_list = sentences_list[:]
        logging.info(f"Paraphrasing article: {file_name}")
        entities_list = extracted_entities.get(file_name, [])
        if len(sentences_list) != len(entities_list):
            logging.warning(f"Mismatch in number of sentences and entities for article: {file_name}")

        paraphrased_sentences = []
        entities_list = extracted_entities.get(file_name, [])

        for i, sentence in enumerate(sentences_list):
            logging.info(f"Paraphrasing sentence {i+1}/{len(sentences_list)} in article: {file_name}")
            entities = entities_list[i] if i < len(entities_list) else []
            paraphrased = paraphrase_article(sentence, entities)
            if paraphrased:
                paraphrased_sentences.append(paraphrased.paraphrased_text)

        logging.info(f"Paraphrased {len(paraphrased_sentences)} sentences in article: {file_name}")
        logging.info(f"Processed {processed_sentences}/{total_sentences} sentences.")
        paraphrased_articles[file_name] = {
            "paraphrased_sentences": paraphrased_sentences
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(paraphrased_articles, f, indent=4, ensure_ascii=False)
    logging.info(f"Processed {processed_sentences}/{total_sentences} sentences.")
    logging.info(f"Successfully paraphrased articles into {output_file}")


if __name__ == "__main__":
    output_file = "projects/prls/paraphrased_articles.json"

    process_articles(output_file)
