## Knowledge Graph Node Creation Steps

1. Chunking: Divide the input document into smaller chunks of text. This makes it easier for the language model to process the information. 


2. Decontextualization: Replace pronouns and ambiguous references with their full forms within each chunk. This ensures that entities are represented consistently throughout the knowledge graph.  For example, if a text chunk refers to "he" after mentioning "Albert Einstein" in a previous chunk, "he" should be replaced with "Albert Einstein." 


3. Entity Extraction: Identify and extract all relevant entities from each decontextualized chunk of text. Entities could be people, locations, organizations, dates, or events. 


4. Paraphrasing: Paraphrase the decontextualized text using identified entities to improve clarity and readability. This involves rephrasing sentences while maintaining the original meaning, often using a more concise or formal language structure.


5. Relation Extraction: Identify and extract the specific relationships between entities based on the text.  For example, a relationship between "Albert Einstein" and "Institute for Advanced Study" might be "worked at." 


6. Event Extraction (Optional): Extract events from the text chunk, combining extracted entities and relations.  For example, an event could be "Einstein's arrival in the United States," which would involve the entities "Albert Einstein," "United States," and "1933" and the relation "arrived in." 


7. Proposition Extraction: Extract meaningful statements from each chunk that describe relationships between entities, relations, and events.  For example, a proposition could be "Albert Einstein joined the Institute for Advanced Study in 1933."  This step comes after event extraction so that the propositions can be informed by the events identified in the previous step. 


8. Atomic Fact Generation: Generate concise statements that represent the key information extracted from the chunk, based on the extracted entities, relations, events, and propositions.  Each fact should include the source text that supports it.  For example, an atomic fact could be "Albert Einstein worked at the Institute for Advanced Study. (Source: 'Einstein joined the Institute for Advanced Study in 1933.')" 


9. Node and Edge Creation: Create nodes in the knowledge graph for each distinct entity and event.  Use extracted relationships as edges connecting the nodes.  Store source texts and propositions as node attributes.  For example, a node for "Albert Einstein" might include attributes such as "Source" and "Propositions." 

