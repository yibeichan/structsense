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
# @File    : alignment_agent.py
# @Software: PyCharm


from crewai import LLM, Agent

class ConceptAlignmentAgent:
    """Concept Alignemnt Agent Crew.

    This crew is responsible for performing the concept alignment.
    
    Input:
        {
    "extracted_terms": {
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

    Output:
    {
    "aligned_terms": {
    "1": [
      {
        "entity": "APOE",
        "label": "GENE",
        "ontology_id": "HGNC:613",
        "ontology_label": "apolipoprotein E",
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
        "ontology_id": "CL:0000127",
        "ontology_label": "astrocyte",
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
        "ontology_id": "CL:0000129",
        "ontology_label": "microglial cell",
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

    def __init__(self, agents_config, embedderconfig, tools):
        """Initializes the Information Extractor Crew.

        Args:
            agents_config (dict): Dictionary containing configuration for agents.
        """
        self.agents_config = agents_config
        self.embedderconfig = embedderconfig
        self.tools = tools

    def alignment_agent(self) -> Agent:
        """Creates and returns an alignment agent based on the configuration.

        The alignment agent is defined with a role, goal, backstory, and an LLM model.

        Returns:
            Agent: A configured CrewAI agent.
        """
        extractor_config = self.agents_config
        role_config = extractor_config.get("role")
        goal_config = extractor_config.get("goal")
        backstory_config = extractor_config.get("backstory")
        llm_config = extractor_config.get("llm")
        embedder_config = self.embedderconfig.get("embedder_config")

        if len(self.tools) > 1:

            return Agent(
                role=role_config,
                goal=goal_config,
                backstory=backstory_config,
                embedder=embedder_config,
                tools=self.tools,
                # meaning the agent has to complete the task,
                # it would not be able delegate task to other agent to complete the task
                # we want each agent to complete the assigned task
                allow_delegation=False,
                verbose=True,
                llm=LLM(**llm_config),
            )
        else:
            return Agent(
                role=role_config,
                goal=goal_config,
                backstory=backstory_config,
                embedder=embedder_config,
                # meaning the agent has to complete the task,
                # it would not be able delegate task to other agent to complete the task
                # we want each agent to complete the assigned task
                allow_delegation=False,
                verbose=True,
                llm=LLM(**llm_config),
            )