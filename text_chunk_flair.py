import os
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter

def split_text_into_sentences(input_file):
    """Split text into sentences using Flair's SegtokSentenceSplitter"""
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Initialize sentence splitter
    splitter = SegtokSentenceSplitter()

    # Use splitter to split text into list of sentences
    sentences = splitter.split(text)

    # Predict tags for sentences
    tagger = Classifier.load('ner')
    tagger.predict(sentences)

    # Iterate through sentences and print predicted labels
    for sentence in sentences:
        print(sentence)

if __name__ == "__main__":
    input_file = "projects/prls/inputs/inputted_facts_2024-10-29T02:39:37-07:00.txt"
    split_text_into_sentences(input_file)
