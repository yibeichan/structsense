import logging
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
        str: A plain, long string containing ontology knowledge.
    """
    response_text = ""

    if not aligned_ner_terms:
        logger.error("No aligned NER terms provided.")
        return "Error: Missing required input: aligned_ner_terms."

    # Extract the correct key dynamically
    if isinstance(aligned_ner_terms, dict):
        key_name = next((key for key in aligned_ner_terms.keys() if isinstance(aligned_ner_terms[key], dict)), None)
        if key_name:
            terms_data = aligned_ner_terms[key_name]
        else:
            logger.error("No valid key found in aligned_ner_terms.")
            return "Error: Invalid input structure. No recognized key found."
    else:
        logger.error("Unexpected format for aligned_ner_terms.")
        return "Error: Invalid input structure."

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

            if not entity:
                logger.warning(f"Skipping term due to missing entity: {term}")
                continue

            response_text += f"Entity: {entity}, Label: {label}. "

            try:
                # Perform hybrid search for the entity
                client = get_weaviate_client()
                search_results = hybrid_search(client, entity)
                client.close()

                # Append search results to response text
                if search_results:
                    response_text += "Search Results: "
                    for res in search_results:
                        response_text += (
                            f"[Label: {res.get('label', 'N/A')}, "
                            f"Definition: {res.get('definition', 'No definition available')}, "
                            f"Ontology: {res.get('ontology', 'N/A')}, "
                            f"Class URI: {res.get('class_uri', 'N/A')}]. "
                        )

            except Exception as e:
                logger.error(f"Hybrid search failed for entity '{entity}': {e}")
                response_text += "Search Results: Failed to retrieve search results. "

    return response_text.strip() if response_text else "No ontology knowledge found."
