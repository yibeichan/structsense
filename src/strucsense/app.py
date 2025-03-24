"""
This script initializes and runs the StrucSense CrewAI pipeline.
"""

from utils.utils import load_config
from crewai import Crew
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

from crew.extractor_agent import InformationExtractorAgent
from crew.extractor_task import InformationExtractorTask
from crew.alignment_agent import ConceptAlignmentAgent
from crew.alignment_task import ConceptAlignmentTask
from typing import Dict


def initialize_extractor(agent_config: Dict, task_config: Dict):
    """Initialize the information extractor agent and task."""
    extractor_agent_init = InformationExtractorAgent(agent_config["extractor_agent"])
    extractor_task_init = InformationExtractorTask(task_config["extractor_agent_task"])

    extractor_agent = extractor_agent_init.extractor_agent()
    extractor_task = extractor_task_init.extractor_task(extractor_agent)

    return extractor_agent, extractor_task


def initialize_alignment(agent_config: Dict, task_config: Dict):
    """Initialize the concept alignment agent and task."""
    alignment_agent_init = ConceptAlignmentAgent(agent_config["alignment_agent"])
    alignment_task_init = ConceptAlignmentTask(task_config["alignment_agent_task"])

    alignment_agent = alignment_agent_init.alignment_agent()
    alignment_task = alignment_task_init.alignment_task(alignment_agent)

    return alignment_agent, alignment_task


def StrucSense(agent_config: str, task_config: str, embedder_config: str, source: str = "asdf"):
    """Runs the CrewAI pipeline for structured knowledge extraction and alignment.

    Args:
        agent_config (str): Path to the agent configuration file.
        task_config (str): Path to the task configuration file.
        embedder_config (str): Path to the embedder configuration file.
        source (str, optional): The source of literature. Defaults to "asdf".
    """
    try:
        agentconfig = load_config(agent_config, "agent")
        taskconfig = load_config(task_config, "task")
        embedderconfig = load_config(embedder_config, "embedder")

        # Initialize agents and tasks
        extractor_agent, extractor_task = initialize_extractor(agentconfig, taskconfig)
        alignment_agent, alignment_task = initialize_alignment(agentconfig, taskconfig)

        # Set up memory components
        long_term_memory = LongTermMemory(
            storage=LTMSQLiteStorage(db_path="crew_memory/long_term_memory_storage.db")
        )
        short_term_memory = ShortTermMemory(
            storage=RAGStorage(
                embedder_config=embedderconfig.get("embedder_config"),
                type="short_term",
                path="crew_memory/"
            )
        )
        entity_memory = EntityMemory(
            storage=RAGStorage(
                embedder_config=embedderconfig.get("embedder_config"),
                type="short_term",
                path="crew_memory/"
            )
        )

        # Initialize CrewAI pipeline
        extractor_crew = Crew(
            agents=[extractor_agent, alignment_agent],
            tasks=[extractor_task, alignment_task],
            memory=True,
            long_term_memory_config=long_term_memory,
            short_term_memory=short_term_memory,
            entity_memory=entity_memory,
            verbose=True,
        )

        # Execute Crew pipeline
        inputs = {"literature": source}
        extractor_crew_result = extractor_crew.kickoff(inputs=inputs)

        # Print results
        print("*" * 100)
        print(extractor_crew_result)
        print("*" * 100)

    except Exception as e:
        print("[ERROR] An error occurred in StrucSense:", str(e))
