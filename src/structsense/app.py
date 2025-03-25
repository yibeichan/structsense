import logging
from typing import Dict

from utils.utils import load_config
from crewai import Crew
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

from crew.extractor_agent import InformationExtractorAgent
from crew.extractor_task import InformationExtractorTask
from crew.alignment_agent import ConceptAlignmentAgent
from crew.alignment_task import ConceptAlignmentTask
from utils.types import ExtractedStructuredTerms, AlignedStructuredTerms
from crewai.flow.flow import Flow, listen, start

from utils.ontology_knowedge_tool import OntologyKnowledgeTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("StructSenseFlow")



class StructSenseFlow(Flow):
    def __init__(self, agent_config: str, task_config: str, embedder_config: str, source_text: str):
        super().__init__()

        logger.info(f"Initializing StructSenseFlow with source: {source_text}")
        self.source_text = source_text

        # Load configurations with error handling
        try:
            self.agentconfig = load_config(agent_config, "agent")
            self.taskconfig = load_config(task_config, "task")
            self.embedderconfig = load_config(embedder_config, "embedder")
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise

        # Initialize memory components
        try:
            self.long_term_memory = LongTermMemory(
                storage=LTMSQLiteStorage(db_path="crew_memory/long_term_memory_storage.db")
            )
            self.short_term_memory = ShortTermMemory(
                storage=RAGStorage(
                    embedder_config=self.embedderconfig.get("embedder_config"),
                    type="short_term",
                    path="crew_memory/",
                )
            )
            self.entity_memory = EntityMemory(
                storage=RAGStorage(
                    embedder_config=self.embedderconfig.get("embedder_config"),
                    type="short_term",
                    path="crew_memory/",
                )
            )
        except Exception as e:
            logger.error(f"Memory initialization failed: {e}")
            raise

    def initialize_extractor(self, agent_config: Dict, task_config: Dict):
        """Initialize the information extractor agent and task."""
        try:
            extractor_agent_init = InformationExtractorAgent(agent_config["extractor_agent"])
            extractor_task_init = InformationExtractorTask(task_config["extractor_agent_task"])

            extractor_agent = extractor_agent_init.extractor_agent()
            extractor_task = extractor_task_init.extractor_task(extractor_agent)

            if not extractor_task:
                logger.error("Extractor task initialization failed.")
                return None, None

            return extractor_agent, extractor_task
        except Exception as e:
            logger.error(f"Extractor initialization failed: {e}")
            return None, None

    def initialize_alignment(self, agent_config: Dict, task_config: Dict):
        """Initialize the concept alignment agent and task."""
        try:
            alignment_agent_init = ConceptAlignmentAgent(agent_config["alignment_agent"])
            alignment_task_init = ConceptAlignmentTask(task_config["alignment_agent_task"])

            alignment_agent = alignment_agent_init.alignment_agent()
            alignment_task = alignment_task_init.alignment_task(alignment_agent)

            if not alignment_task:
                logger.error("Alignment task initialization failed.")
                return None, None

            return alignment_agent, alignment_task
        except Exception as e:
            logger.error(f"Alignment initialization failed: {e}")
            return None, None

    @start()
    def process_inputs(self):
        logger.debug("Starting processing ")

    @listen(process_inputs)
    async def extracted_structured_information(self) -> ExtractedStructuredTerms:
        logger.debug("Processing extracted_structured_information")
        logger.info("Starting structured information extraction.")
        logger.info(f"Source Text: {self.source_text}")

        extractor_agent, extractor_task = self.initialize_extractor(self.agentconfig, self.taskconfig)
        if not extractor_agent or not extractor_task:
            logger.error("Extractor agent or task not initialized.")
            return None

        inputs = {"literature": self.source_text}
        extractor_crew = Crew(
            agents=[extractor_agent],
            tasks=[extractor_task],
            memory=True,
            long_term_memory_config=self.long_term_memory,
            short_term_memory=self.short_term_memory,
            entity_memory=self.entity_memory,
            verbose=True,
        )

        extractor_crew_result = extractor_crew.kickoff(inputs=inputs)
        logger.debug(f"Extractor Crew Result: {extractor_crew_result}, {type(extractor_crew_result)}")

        if not extractor_crew_result:
            logger.warning("Extractor crew returned no results.")
            return None

        self.state["current_step"] = "extracted_structured_information"
        return extractor_crew_result.to_dict()

    @listen(extracted_structured_information)
    async def align_structured_information(self, data_from_previous_step_extracted_info) -> AlignedStructuredTerms:
        if not data_from_previous_step_extracted_info:
            logger.warning("No structured information extracted. Skipping alignment.")
            return None

        logger.info("Starting structured information alignment.")
        alignment_agent, alignment_task = self.initialize_alignment(self.agentconfig, self.taskconfig)
        if not alignment_agent or not alignment_task:
            logger.error("Alignment agent or task not initialized.")
            return None

        ontology_tool = OntologyKnowledgeTool()

        alignment_crew = Crew(
            agents=[alignment_agent],
            tasks=[alignment_task],
            memory=True,
            long_term_memory_config=self.long_term_memory,
            short_term_memory=self.short_term_memory,
            entity_memory=self.entity_memory,
            tools=[ontology_tool],
            verbose=True,
        )

        alignment_crew_result = alignment_crew.kickoff(inputs={"extracted_info": data_from_previous_step_extracted_info})
        logger.info(f"Alignment Crew Result: {alignment_crew_result}")

        if not alignment_crew_result:
            logger.warning("Alignment crew returned no results.")
            return None

        self.state["aligned_information"] = alignment_crew_result
        self.state["current_step"] = "aligned_information"
        return alignment_crew_result



def kickoff(agentconfig: str, taskconfig: str, embedderconfig: str, source_text: str):
    """Asynchronously kickoff the StructSense flow."""
    agentflow = StructSenseFlow(agentconfig, taskconfig, embedderconfig, source_text)
    result = agentflow.kickoff()
    print(result)
