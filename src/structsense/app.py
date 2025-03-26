import warnings
warnings.filterwarnings("ignore")

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
from crew.judge_agent import JudgeAgent
from crew.judge_task import JudgeTask
from utils.types import ExtractedStructuredTerms, AlignedStructuredTerms, JudgeStructuredTerms
from crewai.flow.flow import Flow, listen, start

from utils.ontology_knowedge_tool import OntologyKnowledgeTool

from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

import weave
import mlflow
import os
from dotenv import load_dotenv
import tracemalloc
tracemalloc.start()


load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("StructSenseFlow")
logger.info("Logging to Weights & Biases")
weave.init(project_name="StructSense")

logger.info("Logging to mlflow")
mlflow.crewai.autolog()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URL", "http://localhost:5000"))
mlflow.set_experiment("StructSense")

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

    def initialize_extractor(self, agent_config: Dict, task_config: Dict, tools=[]):
        """Initialize the information extractor agent and task."""
        try:
            extractor_agent_init = InformationExtractorAgent(agent_config["extractor_agent"],  self.embedderconfig, tools)
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

    def initialize_alignment(self, agent_config: Dict, task_config: Dict, tools=[]):
        """Initialize the concept alignment agent and task."""
        try:
            alignment_agent_init = ConceptAlignmentAgent(agent_config["alignment_agent"], self.embedderconfig, tools)
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

    def initialize_judge(self, agent_config: Dict, task_config: Dict, tools=[]):
        """Initialize the judge alignment agent and task."""
        try:
            judge_agent_init = JudgeAgent(agent_config["judge_agent"],  self.embedderconfig, tools)
            judge_task_init = JudgeTask(task_config["judge_agent_task"])

            judge_agent = judge_agent_init.judge_agent()
            judge_task = judge_task_init.judge_task(judge_agent)

            if not judge_task:
                logger.error("Judge task initialization failed.")
                return None, None

            return judge_agent, judge_task
        except Exception as e:
            logger.error(f"Judge initialization failed: {e}")
            return None, None

    @start("start_process")
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
        self.state["extractor_crew_result"] = extractor_crew_result.to_dict()
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
        # Tool is not used because of https://github.com/crewAIInc/crewAI/issues/949
        custom_source = OntologyKnowledgeTool(data_from_previous_step_extracted_info)
        ksrc = StringKnowledgeSource(
            content=custom_source
        )

        alignment_crew = Crew(
            agents=[alignment_agent],
            tasks=[alignment_task],
            memory=True,
            knowledge_sources=[ksrc],
            long_term_memory_config=self.long_term_memory,
            short_term_memory=self.short_term_memory,
            entity_memory=self.entity_memory,
            verbose=True,
        )

        alignment_crew_result = alignment_crew.kickoff(inputs={
            "extracted_info": data_from_previous_step_extracted_info,
        })
        logger.info(f"Alignment Crew Result: {alignment_crew_result}")

        if not alignment_crew_result:
            logger.warning("Alignment crew returned no results.")
            return None

        self.state["current_step"] = "aligned_information"
        return alignment_crew_result.to_dict()

    @listen(align_structured_information)
    async def judge_alignment(self, data_from_previous_step_align_structured_info):
        logger.debug("-" * 100)
        logger.debug(f"Previous data: {data_from_previous_step_align_structured_info}")
        logger.debug("-" * 100)
        if not data_from_previous_step_align_structured_info:
            logger.warning("No aligned structured information extracted. Skipping judgement.")
            return None

        logger.info("Starting judgement aligned info.")
        judge_agent, judge_task = self.initialize_judge(self.agentconfig, self.taskconfig)
        if not judge_agent or not judge_task:
            logger.error("Judge agent or task not initialized.")
            return None

        prev_data = data_from_previous_step_align_structured_info

        custom_source = OntologyKnowledgeTool(
            prev_data,
        )
        ksrc = StringKnowledgeSource(
            content=custom_source
        )

        logger.debug("#"*100)
        logger.debug(prev_data)
        logger.debug("-"*100)
        logger.debug(custom_source)
        logger.debug("#" * 100)


        judge_crew = Crew(
            agents=[judge_agent],
            tasks=[judge_task],
            memory=True,
            knowledge_sources=[ksrc],
            long_term_memory_config=self.long_term_memory,
            short_term_memory=self.short_term_memory,
            entity_memory=self.entity_memory,
            verbose=True,
        )

        judge_crew_result = judge_crew.kickoff(inputs={
            "aligned_structured_terms": prev_data,
        })
        logger.info(f"Alignment Crew Result: {judge_crew_result}")

        if not judge_crew_result:
            logger.warning("Alignment crew returned no results.")
            return None

        self.state["judge_information"] = judge_crew_result
        self.state["current_step"] = "judge"
        return judge_crew_result.to_dict()



def kickoff(agentconfig: str, taskconfig: str, embedderconfig: str, source_text: str):
    """ kickoff the StructSense flow."""
    agentflow = StructSenseFlow(agentconfig, taskconfig, embedderconfig, source_text)
    result = agentflow.kickoff()
    print(result)
