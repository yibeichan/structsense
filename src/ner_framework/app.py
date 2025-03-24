"""
It then calls this function if the script is run as the main program.
"""

from utils.utils import load_config
from crew.extractor_agent import InformationExtractorAgentCrew
from crewai import Crew


def StrucSense(agent_config, task_config, source="asdf"):
    agentconfig = load_config(agent_config, "agent")
    taskconfig = load_config(task_config, "task")
    inputs = {"literature": source}
    ex = InformationExtractorAgentCrew(agentconfig, taskconfig)
    ex_agent = ex.extractor_agent()
    ex_tasks = ex.extractor_task(ex_agent)
    extractor_crew = Crew(
        agents=[
            ex_agent,
        ],
        tasks=[ex_tasks],
        verbose=True,
    )
    extractor_crew_result = extractor_crew.kickoff(inputs=inputs)

    print("*" * 100)
    print(extractor_crew_result)
    print("*" * 100)
