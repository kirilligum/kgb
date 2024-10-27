import json
from openai import OpenAI
from pydantic import BaseModel
from pathlib import Path
from extract_entities import extract_entities_from_article

# Initialize OpenAI client
client = OpenAI()

class ParaphrasedArticle(BaseModel):
    file_name: str
    original_text: str
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

        print("No valid response received from OpenAI API")
        return None

    except Exception as e:
        print(f"Error paraphrasing article: {str(e)}")
        return None


def process_articles(input_file, output_file):
    """Process articles and paraphrase them"""
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    paraphrased_articles = {}

    for file_name, article in list(articles.items()):
        print(f"Paraphrasing article: {file_name}")
        body_text = article.get("body_text", "")
        entities = extract_entities_from_article(body_text)

        if body_text and entities:
            paraphrased = paraphrase_article(body_text, entities)
            if paraphrased:
                paraphrased_articles[file_name] = paraphrased.dict()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(paraphrased_articles, f, indent=4, ensure_ascii=False)
    print(f"Successfully paraphrased articles into {output_file}")


if __name__ == "__main__":
    input_file = "data/merged_articles.json"
    output_file = "data/paraphrased_articles.json"

    process_articles(input_file, output_file)
