
import logging
import os
import sys
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any, Callable, List

# Filter warnings at the beginning
import warnings

warnings.filterwarnings("ignore")
import json
from crewai import Crew
from crewai.flow.flow import Flow, listen, start
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.memory import EntityMemory, LongTermMemory, ShortTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from dotenv import load_dotenv
from utils.human_in_loop_handler import HumanInTheLoop, ProgrammaticFeedbackHandler, HumanInterventionRequired
from utils.types import ExtractedTermsDynamic, AlignedTermsDynamic, JudgedTermsDynamic
from crew.dynamic_agent import DynamicAgent
from crew.dynamic_agent_task import DynamicAgentTask
from utils.ontology_knowedge_tool import OntologyKnowledgeTool
from utils.utils import load_config, process_input_data

# Start memory tracking
tracemalloc.start()
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


class StructSenseFlow(Flow):
    """
    A workflow for structured information extraction, alignment, and judgment using CrewAI.

    This flow processes text data to extract structured information, aligns it with
    external knowledge sources (if enabled), and provides judgment about the alignment.
    Includes human-in-the-loop capabilities for critical review points.
    """

    def __init__(
            self,
            agent_config: str,
            task_config: str,
            embedder_config: str,
            source_text: str,
            knowledge_config: Optional[str] = None,
            enable_human_feedback: bool = False,
            agent_feedback_config: Dict[str, bool] = None
    ):
        """
        Initialize the StructSenseFlow.

        Args:
            agent_config: Path to agent configuration or configuration dictionary
            task_config: Path to task configuration or configuration dictionary
            embedder_config: Path to embedder configuration or configuration dictionary
            source_text: The source text to process
            knowledge_config: Optional path to knowledge configuration or configuration dictionary
            enable_human_feedback: Whether to enable human-in-the-loop functionality globally
            agent_feedback_config: Optional dictionary mapping agent names to feedback enabled status
        """
        super().__init__()

        logger.info(f"Initializing StructSenseFlow")
        self.source_text = source_text

        # Initialize human-in-the-loop component
        self.human = HumanInTheLoop(
            enable_human_feedback=enable_human_feedback,
            agent_feedback_config=agent_feedback_config
        )

        # Load configurations
        try:
            self.agent_config = load_config(agent_config, "agent")
            self.task_config = load_config(task_config, "task")
            self.embedder_config = load_config(embedder_config, "embedder")

            if knowledge_config is None:
                os.environ["ENABLE_KG_SOURCE"] = "false"
                self.knowledge_config = {"search_key": {}}
            else:
                self.knowledge_config = load_config(knowledge_config, "knowledge")

        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise ConfigError(f"Failed to load configurations: {str(e)}")

        # Initialize monitoring tools if enabled
        self._setup_monitoring()

        # Initialize memory components
        self._initialize_memory()

    def _setup_monitoring(self) -> None:
        """Set up monitoring tools if enabled."""
        if os.getenv("ENABLE_WEIGHTSANDBIAS", "false").lower() == "true":
            try:
                import weave
                weave.init(project_name="StructSense")
                logger.info("Weights & Biases monitoring enabled")
            except ImportError:
                logger.warning("Weights & Biases package not found, monitoring disabled")

        if os.getenv("ENABLE_MLFLOW", "false").lower() == "true":
            try:
                import mlflow
                mlflow.crewai.autolog()
                mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URL", "http://localhost:5000"))
                mlflow.set_experiment("StructSense")
                logger.info("MLflow monitoring enabled")
            except ImportError:
                logger.warning("MLflow package not found, monitoring disabled")

    def _initialize_memory(self) -> None:
        """Initialize memory storage systems for the flow."""
        try:
            memory_path = Path("crew_memory")
            memory_path.mkdir(exist_ok=True)

            # Configure RAG storage options
            rag_storage_config = {
                "embedder_config": self.embedder_config.get("embedder_config"),
                "type": "short_term",
                "path": str(memory_path),
            }

            # Initialize memory components
            self.long_term_memory = LongTermMemory(
                storage=LTMSQLiteStorage(db_path=str(memory_path / "long_term_memory_storage.db"))
            )
            self.short_term_memory = ShortTermMemory(
                storage=RAGStorage(**rag_storage_config)
            )
            self.entity_memory = EntityMemory(
                storage=RAGStorage(**rag_storage_config)
            )

            logger.info("Memory systems initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize memory systems: {str(e)}"
            logger.error(error_msg)
            raise ConfigError(error_msg)

    def _initialize_agent_and_task(
            self,
            agent_key: str,
            task_key: str,
            pydantic_output_class
    ) -> Tuple[Optional[object], Optional[object]]:
        """
        Initialize an agent and its associated task.

        Args:
            agent_key: Key for the agent in agent configuration
            task_key: Key for the task in task configuration
            pydantic_output_class: Pydantic class for structured output

        Returns:
            Tuple containing the initialized agent and task
        """
        try:
            agent_init = DynamicAgent(
                agents_config=self.agent_config[agent_key],
                embedder_config=self.embedder_config
            )
            task_init = DynamicAgentTask(
                tasks_config=self.task_config[task_key]
            )

            agent = agent_init.build_agent()
            task = task_init.build_task(pydantic_output=pydantic_output_class, agent=agent)

            if not task:
                logger.error(f"{task_key} initialization failed")
                return None, None

            logger.info(f"Successfully initialized {agent_key} and {task_key}")
            return agent, task

        except Exception as e:
            logger.error(f"{agent_key}/{task_key} initialization failed: {e}")
            return None, None

    def _should_enable_knowledge_source(self) -> bool:
        """Check if knowledge source should be enabled."""
        return os.getenv("ENABLE_KG_SOURCE", "false").lower() == "true"

    def _create_crew_with_knowledge(
            self,
            agent,
            task,
            data_for_knowledge_tool: Optional[Dict] = None
    ) -> Crew:
        """
        Create a Crew instance with or without knowledge sources based on configuration.

        Args:
            agent: The agent to use in the crew
            task: The task to execute
            data_for_knowledge_tool: Data to provide to the knowledge tool if enabled

        Returns:
            Configured Crew instance
        """
        crew_config = {
            "agents": [agent],
            "tasks": [task],
            "memory": True,
            "long_term_memory_config": self.long_term_memory,
            "short_term_memory": self.short_term_memory,
            "entity_memory": self.entity_memory,
            "verbose": True,
        }

        # Add knowledge sources if enabled
        if self._should_enable_knowledge_source() and data_for_knowledge_tool:
            custom_source = OntologyKnowledgeTool(
                data_for_knowledge_tool,
                self.knowledge_config["search_key"]
            )

            logger.debug("Knowledge source result:")
            logger.debug(custom_source)

            ksrc = StringKnowledgeSource(content=custom_source)
            crew_config["knowledge_sources"] = [ksrc]

        return Crew(**crew_config)

    @start()
    def process_inputs(self):
        """Start processing the input data."""
        logger.info("Starting structured information processing flow")

    @listen(process_inputs)
    async def extracted_structured_information(self):
        """Extract structured information from the source text."""
        logger.info("Starting structured information extraction")

        # Initialize extractor components
        extractor_agent, extractor_task = self._initialize_agent_and_task(
            "extractor_agent",
            "extraction_task",
            ExtractedTermsDynamic
        )

        if not extractor_agent or not extractor_task:
            logger.error("Extractor initialization failed")
            return None

        agent_name = "extractor_agent"

        # Create and run the extractor crew
        inputs = {"literature": self.source_text}
        extractor_crew = self._create_crew_with_knowledge(extractor_agent, extractor_task)

        # Provide observation before extraction
        self.human.provide_observation(
            message="Starting extraction process with the following input:",
            data=f"Text length: {len(self.source_text)} characters",
            agent_name=agent_name
        )

        extractor_result = extractor_crew.kickoff(inputs=inputs)

        if not extractor_result:
            logger.warning("Extractor crew returned no results")
            return None

        # Update state and return results
        self.state["current_step"] = "extracted_structured_information"
        result_dict = extractor_result.to_dict()
        logger.info(f"Extraction complete with {len(result_dict.get('terms', []))} terms")

        # Request human feedback on extraction results
        result_dict = self.human.request_feedback(
            data=result_dict,
            step_name="extract_structured_information",
            agent_name=agent_name
        )

        return result_dict

    @listen(extracted_structured_information)
    async def align_structured_information(self, extracted_info):
        """Performs the alignment task.

        This function listens to the result produced by the `extracted_structured_information` function,
        captures the output (stored in `extracted_info`), and then executes the alignment agent to carry out
        the necessary alignment operations.

        :param extracted_info: Data output from the extractor agent.
        :return aligned_info: Data output from the alignment agent that judge agent listens to

        Example:
        Input, i.e., extracted_info

        {
               "extracted_terms": {
                    "1": [
                        {
                            "entity": "APOE",
                            "label": "GENE",
                            "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
                            "start": 26,
                            "end": 30,
                            "paper_location": "literature",
                            "paper_title": null,
                            "doi": null
                        },
                        {
                            "entity": "astrocytes",
                            "label": "CELL_TYPE",
                            "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
                            "start": 93,
                            "end": 103,
                            "paper_location": "literature",
                            "paper_title": null,
                            "doi": null
                        },
                        {
                            "entity": "microglia",
                            "label": "CELL_TYPE",
                            "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
                            "start": 108,
                            "end": 117,
                            "paper_location": "literature",
                            "paper_title": null,
                            "doi": null
                        }
                    ]
                }
        }

        Output, i.e., aligned_info: It adds the new information such as "ontology_id" and "ontology_label"
        {
              "aligned_structured_information": {
                "1": [
                  {
                    "entity": "APOE",
                    "label": "GENE",
                    "ontology_id": "HGNC:613",
                    "ontology_label": "APOE",
                    "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
                    "start": 26,
                    "end": 30,
                    "paper_location": "literature",
                    "paper_title": null,
                    "doi": null
                  },
                  {
                    "entity": "astrocytes",
                    "label": "CELL_TYPE",
                    "ontology_id": "CL:0000127",
                    "ontology_label": "Astrocyte",
                    "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
                    "start": 93,
                    "end": 103,
                    "paper_location": "literature",
                    "paper_title": null,
                    "doi": null
                  },
                  {
                    "entity": "microglia",
                    "label": "CELL_TYPE",
                    "ontology_id": "CL:0000129",
                    "ontology_label": "Microglial Cell",
                    "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
                    "start": 108,
                    "end": 117,
                    "paper_location": "literature",
                    "paper_title": null,
                    "doi": null
                  }
                ]
              }
        }

        """
        if not extracted_info:
            logger.warning("No structured information extracted. Skipping alignment.")
            return None

        logger.info("Starting structured information alignment")

        # Initialize alignment components
        alignment_agent, alignment_task = self._initialize_agent_and_task(
            "alignment_agent",
            "alignment_task",
            AlignedTermsDynamic
        )

        if not alignment_agent or not alignment_task:
            logger.error("Alignment initialization failed")
            return None

        agent_name = "alignment_agent"

        # Create and run the alignment crew
        alignment_crew = self._create_crew_with_knowledge(
            alignment_agent,
            alignment_task,
            extracted_info
        )

        # Note: this is different from the human feedback, i.e., request_feedback function that is something being
        # called below. The request_feedback option allows agent to get feedback on its output while providing an option
        # even to edit the agent's output by the user such that it will be taken into account. The request_approval
        # steps is something that tells whether or not to proceed further in the execution i.e., performing its task.
        # # Request human approval before alignment
        # if not self.human.request_approval(
        #         message="Proceed with aligning the extracted information?",
        #         details=f"Extracted terms: {len(extracted_info.get('terms', []))}",
        #         agent_name=agent_name
        # ):
        #     raise HumanInterventionRequired(f"Alignment aborted by human for agent {agent_name}")

        # Provide observation before alignment
        self.human.provide_observation(
            message="Starting alignment process with the following extracted information:",
            data=f"Number of terms: {len(extracted_info.get('terms', []))}",
            agent_name=agent_name
        )

        alignment_result = alignment_crew.kickoff(inputs={
            "extracted_structured_information": extracted_info,
        })

        if not alignment_result:
            logger.warning("Alignment crew returned no results")
            return None

        # Update state and return results
        self.state["aligned_structured_information"] = alignment_result
        self.state["current_step"] = "aligned_structured_information"
        result_dict = alignment_result.to_dict()
        logger.info(f"Alignment complete with {len(result_dict.get('aligned_terms', []))} aligned terms")

        # Request human feedback on alignment results
        # Note: this is different from the human feedback, i.e., request_feedback function that is something being
        # called below. The request_feedback option allows agent to get feedback on its output while providing an option
        # even to edit the agent's output by the user such that it will be taken into account. The request_approval
        # steps is something that tells whether or not to proceed further in the execution i.e., performing its task.
        result_dict = self.human.request_feedback(
            data=result_dict,
            step_name="align_structured_information",
            agent_name=agent_name
        )

        return result_dict

    @listen(align_structured_information)
    async def judge_alignment(self, aligned_info):
        """Judge the quality of the alignment between extracted and reference terms, i.e., the output of the alignment
        agent and assigns the judgement_score based on the quality.

        Example:
        Input, i.e., aligned_info, from the alignment agent:
        {
              "aligned_structured_information": {
                "1": [
                  {
                    "entity": "APOE",
                    "label": "GENE",
                    "ontology_id": "HGNC:613",
                    "ontology_label": "APOE",
                    "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
                    "start": 26,
                    "end": 30,
                    "paper_location": "literature",
                    "paper_title": null,
                    "doi": null
                  },
                  {
                    "entity": "astrocytes",
                    "label": "CELL_TYPE",
                    "ontology_id": "CL:0000127",
                    "ontology_label": "Astrocyte",
                    "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
                    "start": 93,
                    "end": 103,
                    "paper_location": "literature",
                    "paper_title": null,
                    "doi": null
                  },
                  {
                    "entity": "microglia",
                    "label": "CELL_TYPE",
                    "ontology_id": "CL:0000129",
                    "ontology_label": "Microglial Cell",
                    "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
                    "start": 108,
                    "end": 117,
                    "paper_location": "literature",
                    "paper_title": null,
                    "doi": null
                  }
                ]
              }
        }

        Output in json format:
        {
          "judged_structured_information": {
            "1": [
              {
                "entity": "APOE",
                "label": "GENE",
                "ontology_id": "HGNC:613",
                "ontology_label": "APOE",
                "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
                "start": 26,
                "end": 30,
                "judge_score": 1.0, #this has been added by the judge agent.
                "paper_location": "literature",
                "paper_title": null,
                "doi": null
              },
              {
                "entity": "astrocytes",
                "label": "CELL_TYPE",
                "ontology_id": "CL:0000127",
                "ontology_label": "Astrocyte",
                "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
                "start": 93,
                "end": 103,
                "judge_score": 1.0,
                "paper_location": "literature",
                "paper_title": null,
                "doi": null
              },
              {
                "entity": "microglia",
                "label": "CELL_TYPE",
                "ontology_id": "CL:0000129",
                "ontology_label": "Microglial Cell",
                "sentence": "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function.",
                "start": 108,
                "end": 117,
                "judge_score": 1.0,
                "paper_location": "literature",
                "paper_title": null,
                "doi": null
              }
            ]
          }
        }


        """
        if not aligned_info:
            logger.warning("No aligned information available. Skipping judgment.")
            return None

        logger.info("Starting judgment of aligned information")

        # Initialize judge components
        judge_agent, judge_task = self._initialize_agent_and_task(
            "judge_agent",
            "judge_task",
            JudgedTermsDynamic
        )

        if not judge_agent or not judge_task:
            logger.error("Judge initialization failed")
            return None

        agent_name = "judge_agent"

        # Create and run the judge crew
        judge_crew = self._create_crew_with_knowledge(
            judge_agent,
            judge_task,
            aligned_info
        )

        # Request human approval before judgment
        # Note: this is different from the human feedback, i.e., request_feedback function that is something being
        # called below. The request_feedback option allows agent to get feedback on its output while providing an option
        # even to edit the agent's output by the user such that it will be taken into account. The request_approval
        # steps is something that tells whether or not to proceed further in the execution i.e., performing its task.
        if not self.human.request_approval(
                message="Proceed with judging the aligned information?",
                details=f"Aligned terms: {len(aligned_info.get('aligned_terms', []))}",
                agent_name=agent_name
        ):
            raise HumanInterventionRequired(f"Judgment aborted by human for agent {agent_name}")

        # Provide observation before judgment
        self.human.provide_observation(
            message="Starting judgment process with the following aligned information:",
            data=f"Number of aligned terms: {len(aligned_info.get('aligned_terms', []))}",
            agent_name=agent_name
        )

        judge_result = judge_crew.kickoff(inputs={
            "aligned_structured_information": aligned_info,
        })

        if not judge_result:
            logger.warning("Judge crew returned no results")
            return None

        # Update state and return results
        self.state["judge_information"] = judge_result
        self.state["current_step"] = "judge"
        result_dict = judge_result.to_dict()
        logger.info(f"Judgment complete with {len(result_dict.get('judged_terms', []))} judged terms")

        # Request human feedback on judgment results
        # Note: this is different from the human feedback, i.e., request_feedback function that is something being
        # called below. The request_feedback option allows agent to get feedback on its output while providing an option
        # even to edit the agent's output by the user such that it will be taken into account. The request_approval
        # steps is something that tells whether or not to proceed further in the execution i.e., performing its task.
        result_dict = self.human.request_feedback(
            data=result_dict,
            step_name="judge_alignment",
            agent_name=agent_name
        )

        # Final human approval of results
        # Note: this is different from the human feedback, i.e., request_feedback function that is something being
        # called below. The request_feedback option allows agent to get feedback on its output while providing an option
        # even to edit the agent's output by the user such that it will be taken into account. The request_approval
        # steps is something that tells whether or not to proceed further in the execution i.e., performing its task.
        if not self.human.request_approval(
                message="Accept final results?",
                details=f"Final judged terms: {len(result_dict.get('judged_terms', []))}",
                agent_name=agent_name
        ):
            logger.warning("Final results rejected by human")
            # We still return the results but log the rejection

        return result_dict


def kickoff(
        agentconfig: Union[str, dict],
        taskconfig: Union[str, dict],
        embedderconfig: Union[str, dict],
        input_source: Union[str, dict],
        knowledgeconfig: Optional[Union[str, dict]] = None,
        enable_human_feedback: bool = True,
        agent_feedback_config: Optional[Dict[str, bool]] = None,
        feedback_handler: Optional[ProgrammaticFeedbackHandler] = None,
) -> Union[Dict[str, Any], str]:
    """
    Run the StructSense flow with the given configurations.

    Args:
        agentconfig: Agent configuration file path or dict
        taskconfig: Task configuration file path or dict
        embedderconfig: Embedder configuration file path or dict
        input_source: Input text source file path or direct text
        knowledgeconfig: Optional knowledge configuration file path or dict
        enable_human_feedback: Whether to enable human-in-the-loop functionality
        agent_feedback_config: Optional dictionary mapping agent names to feedback enabled status
        feedback_handler: Optional custom feedback handler for programmatic feedback

    Returns:
        Dictionary with the final results of the flow, or "feedback" if feedback is required
    """
    try:

        # Process input data
        processed_string = process_input_data(input_source)

        enable_human_feedback = enable_human_feedback if enable_human_feedback else True #just for the safe side as sometimes it's none, we want to enable it always.
        processed_agent_feedback_config =  load_config(agent_feedback_config, "agent_feedback_config") if agent_feedback_config else {
                "extractor_agent": True,
                "alignment_agent": True,
                "judge_agent": True
            }

        # Initialize and run the flow
        flow = StructSenseFlow(
            agent_config=agentconfig,
            task_config=taskconfig,
            embedder_config=embedderconfig,
            knowledge_config=knowledgeconfig,
            source_text=processed_string,
            enable_human_feedback=enable_human_feedback,
            agent_feedback_config=processed_agent_feedback_config
        )

        # Use custom feedback handler if provided, e.g., programmatic feedback
        if feedback_handler:
            flow.human = feedback_handler

        result = flow.kickoff()

        # Handle feedback if required
        if result == "feedback" and feedback_handler:
            pending_feedback = feedback_handler.get_pending_feedback()
            if pending_feedback:
                feedback_result = feedback_handler.provide_feedback(
                    choice="1",  # Default to approve
                    modified_data=None
                )
                feedback_handler.clear_pending_feedback()
                return flow.continue_flow(feedback_result)

        logger.info(f"Flow completed successfully")
        return result

    except Exception as e:
        logger.error(f"Flow execution failed: {str(e)}")
        raise
