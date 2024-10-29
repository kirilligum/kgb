import json
import logging
from openai import OpenAI
from pydantic import BaseModel

# Initialize OpenAI client
client = OpenAI()


class DecontextualizedSentence(BaseModel):
    sentence: str


def decontextualize_sentences(
    previous_sentences, current_sentence, num_previous_sentences=1000
):
    """Decontextualize a sentence using OpenAI API"""
    prompt = """
    Instruction:
    Rewrite the below sentence by resolving all entity coreferences with the preceding sentences from the document.
    - Resolve all inter-sentence pronoun references.
    - Make sure that all pronouns in a sentence refer to some named entity within the same sentence.
    - Explicitly mention entity names wherever necessary to remove ambiguity from a sentence. Remember to make each sentence clear and unambiguous.
    - For each entity, use only the one most informative name.
    - Do not generate anything except the rewritten sentence.
    - Avoid pronouns or ambiguous references in facts and triplets. Instead, directly include all relevant named entities in facts.

    ## Example 1:

    ### Previous sentences from Document:
    Elon Musk, the CEO of Tesla, highlighted the company's commitment to innovation and sustainability.
    He announced that the new Tesla Model Z would set new standards in electric vehicles.

    ### Current sentence:
    He said it will be available for purchase early next year.

    ### Output:
    Elon Musk said the Tesla Model Z will be available for purchase early next year.

    ---

    ## Example 2:

    ### Previous sentences from Document:
    António Guterres, the UN Secretary-General, urged immediate action to combat global warming.
    He emphasized the importance of global cooperation.

    ### Current sentence:
    He warned that delay could lead to irreversible damage.

    ### Output:
    António Guterres warned that delay in addressing climate change could lead to irreversible damage.

    ---

    """

    examples = """
    ## Example 1:

    ### Previous sentences from Document:
    Elon Musk, the CEO of Tesla, highlighted the company's commitment to innovation and sustainability.
    He announced that the new Tesla Model Z would set new standards in electric vehicles.

    ### Current sentence:
    He said it will be available for purchase early next year.

    ### Output:
    Elon Musk said the Tesla Model Z will be available for purchase early next year.

    ---

    ## Example 2:

    ### Previous sentences from Document:
    António Guterres, the UN Secretary-General, urged immediate action to combat global warming.
    He emphasized the importance of global cooperation.

    ### Current sentence:
    He warned that delay could lead to irreversible damage.

    ### Output:
    António Guterres warned that delay in addressing climate change could lead to irreversible damage.

    ---

    ## Example 3:

    ### Previous sentences from Document:
    Critics have praised Haruki Murakami's unique storytelling and depth of characters.
    His latest novel explores themes of isolation and connection.

    ### Current sentence:
    They find its themes particularly relevant today.

    ### Output:
    Critics find the themes of Haruki Murakami's latest novel particularly relevant today.

    ---

    ## Negative Example 1:

    ### Previous sentences from Document:
    Marie Curie was a pioneering physicist and chemist who conducted groundbreaking research on radioactivity.
    She was the first woman to win a Nobel Prize.

    ### Current sentence:
    She remains an inspiration to scientists worldwide.

    ### Incorrect Output:
    She remains an inspiration to scientists worldwide.

    ### Explanation:
    - **Pronoun Not Resolved**: "She" is not resolved to "Marie Curie."
    - **Lack of Clarity**: Does not explicitly mention "Marie Curie," leaving the sentence ambiguous.

    ### Correction:
    Marie Curie remains an inspiration to scientists worldwide.

    ---

    ## Negative Example 2:

    ### Previous sentences from Document:
    Alexander Fleming discovered penicillin in 1928.
    His discovery led to the development of antibiotics.

    ### Current sentence:
    It has saved countless lives.

    ### Incorrect Output:
    It has saved countless lives.

    ### Explanation:
    - **Pronoun Not Resolved**: "It" is not resolved to "penicillin."
    - **Ambiguous Reference**: Does not specify what "it" refers to.

    ### Correction:
    Penicillin has saved countless lives.

    ---

    ## Negative Example 3:

    ### Previous sentences from Document:
    Dr. Jane Goodall delivered a lecture on wildlife conservation and the importance of preserving natural habitats.
    She emphasized the role of young people in making a difference.

    ### Current sentence:
    She believes they can change the world.

    ### Incorrect Output:
    She believes they can change the world.

    ### Explanation:
    - **Pronouns Not Resolved**: "She" and "they" are not resolved to "Dr. Jane Goodall" and "young people."
    - **Ambiguous Reference**: Does not specify who "they" refers to.

    ### Correction:
    Dr. Jane Goodall believes young people can change the world.

    ---

    ## Negative Example 4:

    ### Previous sentences from Document:
    The committee discussed the budget cuts affecting the healthcare sector.
    Several members voiced concerns over the impact on patient care.

    ### Current sentence:
    They decided to propose an alternative plan.

    ### Incorrect Output:
    They decided to propose an alternative plan.

    ### Explanation:
    - **Pronoun Not Resolved**: "They" is not specified; it could refer to "the committee" or "members."
    - **Ambiguity**: Does not clarify who is proposing the alternative plan.

    ### Correction:
    The committee decided to propose an alternative plan.
    """

    user_input = (
        prompt
        + examples
        + "\n## Current sentence\n"
        + current_sentence
        + "\n\n\n\n## Previous sentences\n"
        + "\n".join(previous_sentences)
    )

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that follows instructions carefully. You will be given instructions, positive examples, negative examples with corrections, previous sentences, and current centence. Your jobe is to decontextualization the current sentence based on the previous sentences.",
                },
                {"role": "user", "content": user_input},
            ],
            response_format=DecontextualizedSentence,
        )

        if completion and completion.choices:
            decontextualized = completion.choices[0].message.parsed
            return decontextualized

        print("No valid response received from OpenAI API")
        return None

    except Exception as e:
        print(f"Error decontextualizing sentence: {str(e)}")
        return None


def process_articles(input_file, output_file):
    """Process articles and decontextualize sentences"""
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    decontextualized_articles = {}

    num_previous_sentences = (
        3  # Default value for the number of previous sentences to consider
    )

    total_sentences = sum(len(sentences) for sentences in articles.values())
    processed_sentences = 0

    for file_name, sentences in articles.items():
        processed_sentences += len(sentences)
        print(f"Decontextualizing article: {file_name}")
        decontextualized_sentences = []

        for i, current_sentence in enumerate(sentences[:]):
            previous_sentences = sentences[max(0, i - num_previous_sentences) : i]
            decontextualized = decontextualize_sentences(
                previous_sentences, current_sentence, num_previous_sentences
            )
            if decontextualized:
                decontextualized_sentences.append(decontextualized.sentence)

        decontextualized_articles[file_name] = decontextualized_sentences

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(decontextualized_articles, f, indent=4, ensure_ascii=False)
    logging.info(f"Processed {processed_sentences}/{total_sentences} sentences.")
    print(f"Successfully decontextualized articles into {output_file}")


if __name__ == "__main__":
    input_file = "projects/prls/chunked_articles_flair.json"
    output_file = "projects/prls/decontextualized_articles.json"

    process_articles(input_file, output_file)
