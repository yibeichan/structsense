import logging
import json
from .utils import hybrid_search, get_weaviate_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OntologyKnowledgeTool")


def OntologyKnowledgeTool(aligned_ner_terms):
    """Extracts entities and ontology metadata, performs hybrid search,
    and returns structured ontology knowledge grouped by entity.

    Args:
        aligned_ner_terms (Dict): Named Entity Recognition (NER) terms aligned with ontologies.

    Returns:
        Dict: A structured knowledge dictionary with ontology search results.
    """
    knowledge_dict = {}

    if not aligned_ner_terms:
        logger.error("No aligned NER terms provided.")
        return {"error": "Missing required input: aligned_ner_terms."}

    # Extract the correct key dynamically
    if isinstance(aligned_ner_terms, dict):
        # Find the key that contains the NER terms (e.g., extracted_terms, aligned_ner_term)
        key_name = next((key for key in aligned_ner_terms.keys() if isinstance(aligned_ner_terms[key], dict)), None)
        if key_name:
            terms_data = aligned_ner_terms[key_name]
        else:
            logger.error("No valid key found in aligned_ner_terms.")
            return {"error": "Invalid input structure. No recognized key found."}
    else:
        logger.error("Unexpected format for aligned_ner_terms.")
        return {"error": "Invalid input structure."}

    # Iterate through entity groups
    for term_group in terms_data.values():
        if not isinstance(term_group, list):
            logger.warning(f"Unexpected format in term group: {term_group}")
            continue

        for term in term_group:
            if not isinstance(term, dict):
                logger.warning(f"Skipping non-dictionary term: {term}")
                continue

            entity = term.get("entity")
            label = term.get("label")
            ontology_id = term.get("ontology_id", "unknown")
            ontology_label = term.get("ontology_label", "unknown")
            sentence = term.get("sentence", "No sentence provided.")
            doi = term.get("doi", "unknown")
            paper_title = term.get("paper_title", "unknown")

            if not entity:
                logger.warning(f"Skipping term due to missing entity: {term}")
                continue

            try:
                # Perform hybrid search for the entity
                client = get_weaviate_client()
                search_results = hybrid_search(client, entity)

                # Merge results if entity already exists
                if entity in knowledge_dict:
                    knowledge_dict[entity]["search_results"].extend(search_results)
                else:
                    knowledge_dict[entity] = {
                        "label": label,
                        "ontology_id": ontology_id,
                        "ontology_label": ontology_label,
                        "sentence": sentence,
                        "doi": doi,
                        "paper_title": paper_title,
                        "search_results": search_results
                    }
            except Exception as e:
                logger.error(f"Hybrid search failed for entity '{entity}': {e}")
                knowledge_dict[entity] = {
                    "label": label,
                    "ontology_id": ontology_id,
                    "ontology_label": ontology_label,
                    "sentence": sentence,
                    "doi": doi,
                    "paper_title": paper_title,
                    "search_results": None,
                    "error": str(e)
                }

    print("#" * 100)
    print(knowledge_dict)
    print("#" * 100)
    logger.info(f"Ontology knowledge retrieved: {knowledge_dict}")
    return json.dumps(knowledge_dict, indent=4)
