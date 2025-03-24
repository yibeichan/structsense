"""
It then calls this function if the script is run as the main program.
"""

from utils.utils import load_config
from crew.extractor_agent import InformationExtractorAgent
from crew.extractor_task import InformationExtractorTask
from crewai import Crew
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage


def StrucSense(agent_config, task_config, embedder_config, source="asdf"):
    agentconfig = load_config(agent_config, "agent")
    taskconfig = load_config(task_config, "task")
    embedderconfig = load_config(embedder_config, "embedder")

    inputs = {"literature": source}
    extractor_agent_init = InformationExtractorAgent(agentconfig["extractor_agent"])
    extractor_task_init = InformationExtractorTask(taskconfig["extractor_agent_task"])
    extractor_agent = extractor_agent_init.extractor_agent()
    extractor_task = extractor_task_init.extractor_task(extractor_agent)

    extractor_crew = Crew(
        agents=[
            extractor_agent,
        ],
        tasks=[extractor_task],
        memory=True,

        # Long-term memory for persistent storage across sessions
        long_term_memory_config=LongTermMemory(
            storage=LTMSQLiteStorage(
                db_path="crew_memory/long_term_memory_storage.db"
            )
        ),
        # Short-term memory for current context using RAG
        short_term_memory=ShortTermMemory(
            storage=RAGStorage(
                embedder_config=embedderconfig.get("embedder_config"),
                type="short_term",
                path="crew_memory/"
            )
        ),

        # Entity memory for tracking key information about entities
        entity_memory=EntityMemory(
            storage=RAGStorage(
                embedder_config=embedderconfig.get("embedder_config"),
                type="short_term",
                path="crew_memory/"
            )
        ),
        verbose=True,
    )
    extractor_crew_result = extractor_crew.kickoff(inputs=inputs)


    print("*" * 100)
    print(extractor_crew_result)
    print("*" * 100)