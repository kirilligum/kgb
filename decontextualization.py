import json
from openai import OpenAI
from pydantic import BaseModel

# Initialize OpenAI client
client = OpenAI()

class DecontextualizedSentence(BaseModel):
    sentence: str

def decontextualize_sentences(previous_sentences, current_sentence):
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

    Now, please provide the rewritten sentence based on the following:

    ### Previous sentences from Document:
    {previous_sentences}

    ### Current sentence:
    {current_sentence}

    ### Output:
    """

    user_input = prompt.format(
        previous_sentences="\n".join(previous_sentences),
        current_sentence=current_sentence
    )

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that follows instructions carefully."},
                {"role": "user", "content": user_input}
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

    for file_name, sentences in articles.items():
        print(f"Decontextualizing article: {file_name}")
        decontextualized_sentences = []

        for i, current_sentence in enumerate(sentences):
            previous_sentences = sentences[:i]
            decontextualized = decontextualize_sentences(previous_sentences, current_sentence)
            if decontextualized:
                decontextualized_sentences.append(decontextualized.sentence)

        decontextualized_articles[file_name] = decontextualized_sentences

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(decontextualized_articles, f, indent=4, ensure_ascii=False)
    print(f"Successfully decontextualized articles into {output_file}")

if __name__ == "__main__":
    input_file = "data/chunked_articles_spacy.json"
    output_file = "data/decontextualized_articles.json"

    process_articles(input_file, output_file)
