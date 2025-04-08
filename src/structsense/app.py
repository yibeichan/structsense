import warnings
warnings.filterwarnings("ignore")


import logging
import os
import sys
import tracemalloc
from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
from crewai import Crew
from crewai.flow.flow import Flow, listen, start
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.memory import EntityMemory, LongTermMemory, ShortTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from dotenv import load_dotenv
from utils.types import ExtractedTermsDynamic, AlignedTermsDynamic, JudgedTermsDynamic
from crew.dynamic_agent import DynamicAgent
from crew.dynamic_agent_task import DynamicAgentTask
from utils.ontology_knowedge_tool import OntologyKnowledgeTool
from utils.utils import extract_json_from_text, load_config, process_input_data
from pathlib import Path
tracemalloc.start()


load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize monitoring tools if enabled
if os.getenv("ENABLE_WEIGHTSANDBIAS", "false").lower() == "true":
    import weave

    weave.init(project_name="StructSense")

if os.getenv("ENABLE_MLFLOW", "false").lower() == "true":
    import mlflow

    mlflow.crewai.autolog()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URL", "http://localhost:5000"))
    mlflow.set_experiment("StructSense")

class StructSenseFlow(Flow):
    def __init__(self, agent_config: str, task_config: str, embedder_config: str, knowledge_config: Optional[str],
                 source_text: str):
        super().__init__()

        logger.info(f"Initializing StructSenseFlow with source: {source_text}")
        self.source_text = source_text

        # Load configurations with error handling
        try:
            self.agentconfig = load_config(agent_config, "agent")
            self.taskconfig = load_config(task_config, "task")
            self.embedderconfig = load_config(embedder_config, "embedder")
            if knowledge_config is None:
                os.environ["ENABLE_KG_SOURCE"] = "false"
                self.knowledgeconfig = {"search_key": {}}
            else:
                self.knowledgeconfig = load_config(knowledge_config, "knowledge")
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise

        # Initialize memory components
        self._initialize_memory()

    def _initialize_memory(self) -> None:
        """Initialize memory storage systems"""
        try:
            memory_path = Path("crew_memory")
            memory_path.mkdir(exist_ok=True)

            self.long_term_memory = LongTermMemory(
                storage=LTMSQLiteStorage(db_path=str(memory_path / "long_term_memory_storage.db"))
            )
            self.short_term_memory = ShortTermMemory(
                storage=RAGStorage(
                    embedder_config=self.embedderconfig.get("embedder_config"),
                    type="short_term",
                    path=str(memory_path),
                )
            )
            self.entity_memory = EntityMemory(
                storage=RAGStorage(
                    embedder_config=self.embedderconfig.get("embedder_config"),
                    type="short_term",
                    path=str(memory_path),
                )
            )
        except Exception as e:
            raise ConfigError(f"Failed to initialize memory systems: {str(e)}")

    def initialize_extractor(self, agent_config: Dict, task_config: Dict):
        """Initialize the information extractor agent and task."""
        try:
            extractor_agent_init = DynamicAgent(agents_config=agent_config["extractor_agent"],
                                                embedder_config=self.embedderconfig)
            extractor_task_init = DynamicAgentTask(tasks_config=task_config["extraction_task"])

            extractor_agent = extractor_agent_init.build_agent()
            extractor_task = extractor_task_init.build_task(pydantic_output=ExtractedTermsDynamic, agent=extractor_agent)

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
            alignment_agent_init = DynamicAgent(agents_config=agent_config["alignment_agent"],
                                                embedder_config=self.embedderconfig)
            alignment_task_init = DynamicAgentTask(tasks_config=task_config["alignment_task"])


            alignment_agent = alignment_agent_init.build_agent()
            alignment_task = alignment_task_init.build_task(pydantic_output=AlignedTermsDynamic, agent=alignment_agent)

            if not alignment_task:
                logger.error("Alignment task initialization failed.")
                return None, None

            return alignment_agent, alignment_task
        except Exception as e:
            logger.error(f"Alignment initialization failed: {e}")
            return None, None

    def initialize_judge(self, agent_config: Dict, task_config: Dict):
        """Initialize the concept alignment agent and task."""
        try:
            judge_agent_init = DynamicAgent(agents_config=agent_config["judge_agent"],  embedder_config=self.embedderconfig)
            judge_task_init = DynamicAgentTask(tasks_config=task_config["judge_task"])

            judge_agent = judge_agent_init.build_agent()
            judge_task = judge_task_init.build_task(pydantic_output=JudgedTermsDynamic, agent=judge_agent)

            if not judge_task:
                logger.error("Judge task initialization failed.")
                return None, None

            return judge_agent, judge_task
        except Exception as e:
            logger.error(f"Judge initialization failed: {e}")
            return None, None

    @start()
    def process_inputs(self):
        logger.debug("Starting processing ")

    @listen(process_inputs)
    async def extracted_structured_information(self):
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
        print("*"*100)
        print(extractor_crew_result.to_dict())
        print("*"*100)
        return extractor_crew_result.to_dict()

    @listen(extracted_structured_information)
    async def align_structured_information(self, data_from_previous_step_extracted_info):
        if not data_from_previous_step_extracted_info:
            logger.warning("No structured information extracted. Skipping alignment.")
            return None

        logger.info("Starting structured information alignment.")
        alignment_agent, alignment_task = self.initialize_alignment(self.agentconfig, self.taskconfig)
        if not alignment_agent or not alignment_task:
            logger.error("Alignment agent or task not initialized.")
            return None
        # Tool is not used because of https://github.com/crewAIInc/crewAI/issues/949
        if self._should_enable_knowledge_source():

            custom_source = OntologyKnowledgeTool(data_from_previous_step_extracted_info,
                                                  self.knowledgeconfig["search_key"])

            logger.debug("#" * 100)
            logger.debug("Knowledge source result")
            logger.debug("-" * 100)
            logger.debug(custom_source)
            logger.debug("#" * 100)

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
        else:

            alignment_crew = Crew(
                agents=[alignment_agent],
                tasks=[alignment_task],
                memory=True,
                long_term_memory_config=self.long_term_memory,
                short_term_memory=self.short_term_memory,
                entity_memory=self.entity_memory,
                verbose=True,
            )

        alignment_crew_result = alignment_crew.kickoff(inputs={
            "extracted_structured_information": data_from_previous_step_extracted_info,
        })
        logger.info(f"Alignment Crew Result: {alignment_crew_result}")

        if not alignment_crew_result:
            logger.warning("Alignment crew returned no results.")
            return None

        self.state["aligned_structured_information"] = alignment_crew_result
        self.state["current_step"] = "aligned_structured_information"
        return alignment_crew_result.to_dict()
    def _should_enable_knowledge_source(self) -> bool:
        """Check if knowledge source should be enabled"""

        return (
                os.getenv("ENABLE_KG_SOURCE", "false").lower() == "true"
        )
    @listen(align_structured_information)
    async def judge_alignment(self, data_from_previous_step_align_structured_info):
        if not data_from_previous_step_align_structured_info:
            logger.warning("No aligned structured information extracted. Skipping judgement.")
            return None

        logger.info("Starting judgement aligned info.")
        judge_agent, judge_task = self.initialize_judge(self.agentconfig, self.taskconfig)
        if not judge_agent or not judge_task:
            logger.error("Judge agent or task not initialized.")
            return None

        prev_data = data_from_previous_step_align_structured_info

        if self._should_enable_knowledge_source():
            custom_source = OntologyKnowledgeTool(
            prev_data,
            self.knowledgeconfig["search_key"]
            )
            logger.debug("#" * 100)
            logger.debug("Knowledge source result")
            logger.debug("-" * 100)
            logger.debug(custom_source)
            logger.debug("#" * 100)
            ksrc = StringKnowledgeSource(
                content=custom_source
            )

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
        else:
            judge_crew = Crew(
                agents=[judge_agent],
                tasks=[judge_task],
                memory=True,
                long_term_memory_config=self.long_term_memory,
                short_term_memory=self.short_term_memory,
                entity_memory=self.entity_memory,
                verbose=True,
            )

        judge_crew_result = judge_crew.kickoff(inputs={
            "aligned_structured_information": prev_data,
        })
        logger.info(f"Alignment Crew Result: {judge_crew_result}")

        if not judge_crew_result:
            logger.warning("Alignment crew returned no results.")
            return None

        self.state["judge_information"] = judge_crew_result
        self.state["current_step"] = "judge"
        return judge_crew_result.to_dict()



def kickoff(
        agentconfig: Union[str, dict],
        taskconfig: Union[str, dict],
        embedderconfig: Union[str, dict],
        flowconfig: Union[str, dict],
        input_source: Union[str, dict],
        knowledgeconfig: Optional[Union[str, dict]] = None
) -> Dict[str, Any]:
    """Run the StructSense flow with the given configurations"""
    try:
        processed_string = process_input_data(input_source)

        flow = StructSenseFlow(
            agent_config=agentconfig,
            task_config=taskconfig,
            embedder_config=embedderconfig,
            knowledge_config=knowledgeconfig,
            source_text=processed_string,
        )

        final_response =  flow.kickoff()
        logger.info(f"Returning {final_response}")
        return final_response
    except Exception as e:
        logger.error(f"Flow execution failed: {str(e)}")
        raise
