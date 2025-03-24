"""
It then calls this function if the script is run as the main program.
"""

from utils.utils import load_config
from crew.extractor_agent import InformationExtractorAgent
from crew.extractor_task import InformationExtractorTask
from crewai import Crew


def StrucSense(agent_config, task_config, source="asdf"):
    agentconfig = load_config(agent_config, "agent")
    taskconfig = load_config(task_config, "task")
    inputs = {"literature": source}
    extractor_agent_init = InformationExtractorAgent(agentconfig)
    extractor_task_init = InformationExtractorTask(taskconfig)
    extractor_agent = extractor_agent_init.extractor_agent()
    extractor_task = extractor_task_init.extractor_task(extractor_agent)
    extractor_crew = Crew(
        agents=[
            extractor_agent,
        ],
        tasks=[extractor_task],
        verbose=True,
    )
    extractor_crew_result = extractor_crew.kickoff(inputs=inputs)

    print("*" * 100)
    print(extractor_crew_result)
    print("*" * 100)
