import warnings
warnings.filterwarnings("ignore")

import logging
import os
import tracemalloc
from dotenv import load_dotenv
import weave
import mlflow

from utils.utils import load_config
from crew.dynamic_agent import DynamicAgent
from crew.dynamic_agent_task import DynamicAgentTask
from crewai import Crew
from crewai.flow.flow import Flow, start
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from utils.ontology_knowedge_tool import OntologyKnowledgeTool

from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

tracemalloc.start()
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StructSenseFlow")

weave.init(project_name="StructSense")
mlflow.crewai.autolog()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URL", "http://localhost:5000"))
mlflow.set_experiment("StructSense")


class StructSenseFlow(Flow):
    def __init__(self, agent_config, task_config, embedder_config, flow_config, source_text):
        super().__init__()

        self.source_text = source_text
        raw_agent_config = load_config(agent_config, "agent")
        raw_task_config = load_config(task_config, "task")
        self.agentconfig = {agent["id"]: agent for agent in raw_agent_config["agents"]}
        self.taskconfig = {task["id"]: task for task in raw_task_config["tasks"]}

        self.embedderconfig = load_config(embedder_config, "embedder")

        self.flowconfig = load_config(flow_config, "flow")

        self.long_term_memory = LongTermMemory(
            storage=LTMSQLiteStorage(db_path="crew_memory/long_term_memory_storage.db")
        )
        self.short_term_memory = ShortTermMemory(
            storage=RAGStorage(embedder_config=self.embedderconfig.get("embedder_config"),
                               type="short_term", path="crew_memory/")
        )
        self.entity_memory = EntityMemory(
            storage=RAGStorage(embedder_config=self.embedderconfig.get("embedder_config"),
                               type="short_term", path="crew_memory/")
        )

        if self.state is None:
            self.state = {}

        self.state["source_text"] = source_text
        self.state["output_of"] = {}

    def interpolate(self, template, context):
        """Simple templating function for '{{variable}}' interpolation."""
        for k, v in context.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    template = template.replace(f"{{{{{k}.{sub_k}}}}}", str(sub_v))
            else:
                template = template.replace(f"{{{{{k}}}}}", str(v))
        return template

    def run_step(self, step):
        logger.info(f"Running step: {step['id']}")

        agent_key = step["agent_key"]
        task_key = step["task_key"]
        tools = []

        # Build agent
        agent_builder = DynamicAgent([self.agentconfig[agent_key]], self.embedderconfig, tools)
        agents_by_id = agent_builder.build_agents()
        agent = agents_by_id[agent_key]

        # Build task
        task_builder = DynamicAgentTask([self.taskconfig[task_key]])
        tasks = task_builder.build_tasks(agents_by_id)
        task = tasks[0]

        # Prepare inputs
        inputs = {}
        for k, v in step.get("inputs", {}).items():
            inputs[k] = self.interpolate(v, self.state)

        # Prepare knowledge source
        ksrc = None
        if "knowledge_source" in step:
            src_key = step["knowledge_source"]
            if src_key in self.state["output_of"]:
                ksrc = StringKnowledgeSource(OntologyKnowledgeTool(self.state["output_of"][src_key]))

        # Launch Crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            memory=True,
            long_term_memory_config=self.long_term_memory,
            short_term_memory=self.short_term_memory,
            entity_memory=self.entity_memory,
            knowledge_sources=[ksrc] if ksrc else [],
            verbose=True,
        )

        result = crew.kickoff(inputs=inputs)
        logger.info(f"Result of {step['id']}: {result}")
        self.state["output_of"][step["id"]] = result.to_dict() if result else None

    @start("kickoff")
    def kickoff_flow(self):
        for step in self.flowconfig["flow"]:
            self.run_step(step)
        return self.state["output_of"]


def kickoff(agentconfig: str, taskconfig: str, embedderconfig: str, flowconfig: str, source_text: str):
    """Kick off the StructSense flow."""
    flow = StructSenseFlow(agent_config=agentconfig,
                           task_config=taskconfig,
                           embedder_config=embedderconfig,
                           flow_config=flowconfig,
                           source_text=source_text)
    result = flow.kickoff()
    print(result)
