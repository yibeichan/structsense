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
# @File    : ner_judge_task.py
# @Software: PyCharm


from crewai import Agent, Task


class JudgeTask:
    """Judge Crew Task.

    This crew is responsible for judging the output.

    Input (from ner_alignment_agent):
    {
     "aligned_ner_terms": {
    "1": [
      {
        "entity": "APOE",
        "label": "GENE",
        "ontology_id": "HGNC:613", 
        "ontology_label": "APOE gene",
        "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
        "start": 26,
        "end": 30,
        "paper_location": "introduction",
        "paper_title": "The role of APOE mutations in neurodegenerative disorders",
        "doi": "10.1101/2023.10.01.123456"
      }
    ],
    "2": [
      {
        "entity": "astrocytes",
        "label": "CELL_TYPE",
        "ontology_id": "CL:0000127", 
        "ontology_label": "Astrocyte",
        "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
        "start": 90,
        "end": 100,
        "paper_location": "introduction",
        "paper_title": "The role of APOE mutations in neurodegenerative disorders",
        "doi": "10.1101/2023.10.01.123456"
      }
    ],
    "3": [
      {
        "entity": "microglia",
        "label": "CELL_TYPE",
        "ontology_id": "CL:0000129", 
        "ontology_label": "Microglial cell",
        "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
        "start": 105,
        "end": 114,
        "paper_location": "introduction",
        "paper_title": "The role of APOE mutations in neurodegenerative disorders",
        "doi": "10.1101/2023.10.01.123456"
      }
    ]
    }
    }


    Output:
      "judge_ner_terms": {
    "1": [
      {
        "entity": "APOE",
        "label": "GENE",
        "ontology_id": "HGNC:613",
        "ontology_label": "APOE gene",
        "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
        "start": 26,
        "end": 30,
        "judge_score": 1.0,
        "paper_location": "introduction",
        "paper_title": "The role of APOE mutations in neurodegenerative disorders",
        "doi": "10.1101/2023.10.01.123456"
      }
    ],
    "2": [
      {
        "entity": "astrocytes",
        "label": "CELL_TYPE",
        "ontology_id": "CL:0000127",
        "ontology_label": "Astrocyte",
        "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
        "start": 90,
        "end": 100,
        "judge_score": 1.0,
        "paper_location": "introduction",
        "paper_title": "The role of APOE mutations in neurodegenerative disorders",
        "doi": "10.1101/2023.10.01.123456"
      }
    ],
    "3": [
      {
        "entity": "microglia",
        "label": "CELL_TYPE",
        "ontology_id": "CL:0000129",
        "ontology_label": "Microglial cell",
        "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
        "start": 105,
        "end": 114,
        "judge_score": 1.0,
        "paper_location": "introduction",
        "paper_title": "The role of APOE mutations in neurodegenerative disorders",
        "doi": "10.1101/2023.10.01.123456"
      }
    ]
    }
    }

    """

    def __init__(self, tasks_config):
        """Initializes the Alignment Crew.

        Args:
            tasks_config (dict): Dictionary containing configuration for tasks.
        """
        self.tasks_config = tasks_config

    def judge_task(self, agent: Agent) -> Task:
        """Creates and returns an Alignment task assigned to the agent.

        Args:
            agent (Agent): The agent that will execute the task.

        Returns:
            Task: A configured CrewAI task.
        """
        return Task(config=self.tasks_config,
                    agent=agent)
