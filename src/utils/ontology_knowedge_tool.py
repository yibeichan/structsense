# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DISCLAIMER: This software is provided "as is" without any warranty,
# express or implied, including but not limited to the warranties of
# merchantability, fitness for a particular purpose, and non-infringement.
#
# In no event shall the authors or copyright holders be liable for any
# claim, damages, or other liability, whether in an action of contract,
# tort, or otherwise, arising from, out of, or in connection with the
# software or the use or other dealings in the software.
# -----------------------------------------------------------------------------

# @Author  : Tek Raj Chhetri
# @Email   : tekraj@mit.edu
# @Web     : https://tekrajchhetri.com/
# @File    : ontology_knowedge_tool.py
# @Software: PyCharm

import logging
from crewai.tools import BaseTool
from pydantic import BaseModel, HttpUrl
from typing import Optional, Type, List, Dict
from .utils import hybrid_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OntologyKnowledgeTool")

class MatchingConcepts(BaseModel):
    """Represents an ontology entity with relevant metadata."""
    ontology: str
    class_id: str
    class_uri: HttpUrl
    definition: str
    label: str
    all_synonyms_combined: Optional[str] = None


class OntologyKnowledgeTool(BaseTool):
    """CrewAI Tool for retrieving ontology knowledge using hybrid search."""

    name: str = "Ontology Knowledge Retrieval"
    description: str = "Fetches relevant ontology concepts for entities to assist in knowledge alignment."
    args_schema: Type[BaseModel] = MatchingConcepts

    def _run(self, entities: List[str], labels: List[str]) -> Dict:
        """
        Iterates through entities and labels, performs a hybrid search, and returns structured ontology knowledge.

        Args:
            entities (List[str]): List of entity names.
            labels (List[str]): Corresponding labels for entities.

        Returns:
            Dict: A structured knowledge dictionary.
        """
        knowledge_dict = {}

        for entity, label in zip(entities, labels):
            try:
                # Perform hybrid search for the entity
                search_results = hybrid_search(f"{entity}")

                # Store results in dictionary
                knowledge_dict[entity] = {
                    "label": label,
                    "search_results": search_results
                }
            except Exception as e:
                logging.error(f"Hybrid search failed for entity '{entity}': {e}")
                knowledge_dict[entity] = {
                    "label": label,
                    "search_results": None,
                    "error": str(e)
                }
        logging.debug(f"Knowledge: {knowledge_dict}")
        return knowledge_dict