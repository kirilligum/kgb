import json
import logging
from openai import OpenAI
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
client = OpenAI()


class AtomicFact(BaseModel):
    facts: list[str]


def extract_atomic_facts(sentence, propositions):
    """Extract atomic facts using OpenAI API"""
    prompt = (
        f"Instructions:\n\n"
        f"1. Carefully analyze the sentence and the provided propositions.\n"
        f"2. Extract atomic facts that convey new, distinct pieces of information NOT already present in the propositions.\n"
        f"3. Express each atomic fact as a concise and grammatically correct sentence.\n"
        f"4. Ensure each atomic fact is complete and doesn't depend on information outside the given sentence.\n"
        f"5. Avoid generating redundant or overlapping facts, both amongst themselves and with the propositions.\n"
        f'6. If the sentence contains no additional factual information beyond what\'s in the propositions, state: "No new atomic facts found."\n'
        f"7. Limit the number of atomic facts to a maximum of 5.\n\n"
        f'Sentence: "{sentence}"\n'
        f"Propositions: {propositions}\n"
    )

    try:
        completion = client.beta.chat.completions.parse(
            # model="gpt-4o",
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts atomic facts from given sentences.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format=AtomicFact,
        )

        if completion and completion.choices:
            atomic_facts = completion.choices[0].message.parsed
            return atomic_facts

        logging.warning("No valid response received from OpenAI API")
        return None

    except Exception as e:
        logging.error(f"Error extracting atomic facts: {str(e)}")
        return None


def process_articles():
    """Process articles and extract atomic facts"""
    logging.info(
        "Loading decontextualized articles from projects/prls/decontextualized_articles.json"
    )
    with open(
        "projects/prls/decontextualized_articles.json", "r", encoding="utf-8"
    ) as f:
        decontextualized_articles = json.load(f)

    total_sentences = sum(len(sentences) for sentences in decontextualized_articles.values())
    processed_sentences = 0
    total_atomic_facts = 0
    logging.info(
        "Loading decontextualized articles from projects/prls/decontextualized_articles.json"
    )
    with open(
        "projects/prls/decontextualized_articles.json", "r", encoding="utf-8"
    ) as f:
        decontextualized_articles = json.load(f)

    logging.info(
        "Loading extracted propositions from projects/prls/extracted_propositions.json"
    )
    with open("projects/prls/extracted_propositions.json", "r", encoding="utf-8") as f:
        extracted_propositions = json.load(f)

    all_atomic_facts = {}

    total_sentences = sum(len(sentences) for sentences in decontextualized_articles.values())
    processed_sentences = 0
    total_atomic_facts = 0

    for file_name, sentences_list in decontextualized_articles.items():
        processed_sentences += len(sentences_list)
        logging.info(f"Processing article: {file_name}")
        propositions_list = extracted_propositions.get(file_name, [])

        article_atomic_facts = [[] for _ in sentences_list]

        for sentence_index, sentence in enumerate(sentences_list):
            propositions = (
                propositions_list[sentence_index]
                if sentence_index < len(propositions_list)
                else []
            )
            logging.info(
                f"Extracting atomic facts for sentence {sentence_index + 1}/{len(sentences_list)} in article: {file_name}"
            )
            atomic_facts = extract_atomic_facts(sentence, propositions)
            if atomic_facts:
                facts_count = len(atomic_facts.facts)
                article_atomic_facts[sentence_index].extend(atomic_facts.facts)
                total_atomic_facts += facts_count
                logging.info(f"Extracted {facts_count} atomic facts for sentence {sentence_index + 1}.")

        all_atomic_facts[file_name] = article_atomic_facts
        logging.info(f"Extracted atomic facts for {len(article_atomic_facts)} sentences in article: {file_name}")
        logging.info(f"Processed {processed_sentences}/{total_sentences} sentences.")

    with open("projects/prls/extracted_atomic_facts.json", "w", encoding="utf-8") as f:
        json.dump(all_atomic_facts, f, indent=4, ensure_ascii=False)
    logging.info(
        "Successfully extracted atomic facts into projects/prls/extracted_atomic_facts.json"
    )


if __name__ == "__main__":
    process_articles()
