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
# @File    : crew.py
# @Software: PyCharm


from crewai import LLM, Agent


class InformationExtractorAgent:
    """Information Extractor Agent Crew.

    This crew is responsible for extracting the structured information based on the passed configuration.

    Example:
        Input:  From the given Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function. extract named entities from neuroscience statements.  A named entity is anything that can be referred to with a proper name.  Some common named entities in neuroscience articles are animal species (e.g., mouse, drosophila, zebrafish), anatomical regions (e.g., neocortex, mushroom body, cerebellum), experimental conditions (e.g., control, tetrodotoxin treatment, Scn1a knockout), and cell types (e.g., pyramidal neuron, direction-sensitive mechanoreceptor, oligodendrocyte)
        
        Output: 
        {
    "extracted_ner_terms": {
    "1": [
      {
        "entity": "APOE",
        "label": "GENE",
        "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
        "start": 29,
        "end": 33,
        "paper_location": "unknown",
        "paper_title": "unknown",
        "doi": "unknown"
      }
    ],
    "2": [
      {
        "entity": "astrocytes",
        "label": "CELL_TYPE",
        "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
        "start": 91,
        "end": 101,
        "paper_location": "unknown",
        "paper_title": "unknown",
        "doi": "unknown"
      }
    ],
    "3": [
      {
        "entity": "microglia",
        "label": "CELL_TYPE",
        "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
        "start": 106,
        "end": 115,
        "paper_location": "unknown",
        "paper_title": "unknown",
        "doi": "unknown"
      }
    ]
    }
    }

    """

    def __init__(self, agents_config):
        """Initializes the Information Extractor Crew.

        Args:
            agents_config (dict): Dictionary containing configuration for agents.
        """
        self.agents_config = agents_config

    def extractor_agent(self) -> Agent:
        """Creates and returns an extractor agent based on the configuration.

        The extractor agent is defined with a role, goal, backstory, and an LLM model.

        Returns:
            Agent: A configured CrewAI agent.
        """
        extractor_config = self.agents_config
        role_config = extractor_config.get("role", "Default Role")
        goal_config = extractor_config.get("goal", "Default Goal")
        backstory_config = extractor_config.get("backstory", "No backstory provided.")
        llm_config = extractor_config.get("llm", {})

        return Agent(
            role=role_config,
            goal=goal_config,
            backstory=backstory_config,
            allow_delegation=False,
            verbose=True,
            llm=LLM(**llm_config),
        )
