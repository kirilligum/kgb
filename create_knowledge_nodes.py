import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_knowledge_nodes():
    """Create knowledge nodes from extracted information."""
    logging.info("Loading data files...")
    decontextualized_articles = load_json("projects/prls/decontextualized_articles.json")
    extracted_entities = load_json("projects/prls/extracted_entities.json")
    extracted_relationships = load_json("projects/prls/extracted_relationships.json")
    extracted_propositions = load_json("projects/prls/extracted_propositions.json")
    extracted_atomic_facts = load_json("projects/prls/extracted_atomic_facts.json")

    knowledge_graph = {}

    total_sentences = sum(len(sentences) for sentences in decontextualized_articles.values())
    processed_sentences = 0

    for file_name, sentences in decontextualized_articles.items():
        processed_sentences += len(sentences)
        logging.info(f"Creating knowledge nodes for article: {file_name}")
        article_nodes = []

        for i, sentence in enumerate(sentences):
            entities = extracted_entities.get(file_name, [])[i]
            relationships = extracted_relationships.get(file_name, [])[i]
            propositions = extracted_propositions.get(file_name, [])[i]
            atomic_facts = extracted_atomic_facts.get(file_name, [])[i]

            node = {
                "sentence": sentence,
                "entities": entities,
                "relationships": relationships,
                "propositions": propositions,
                "atomic_facts": atomic_facts,
            }
            article_nodes.append(node)

        knowledge_graph[file_name] = article_nodes
        logging.info(f"Created {len(article_nodes)} knowledge nodes for article: {file_name}")
        logging.info(f"Processed {processed_sentences}/{total_sentences} sentences.")

    output_file = "projects/prls/knowledge_graph.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(knowledge_graph, f, indent=4, ensure_ascii=False)
    logging.info(f"Processed {processed_sentences}/{total_sentences} sentences.")
    logging.info(f"Successfully created knowledge nodes in {output_file}")

if __name__ == "__main__":
    create_knowledge_nodes()
