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
# @File    : crew.py
# @Software: PyCharm


from crewai import Agent, Task
from utils.types import ExtractedStructuredTerms

class AgentTask:
    """Information Extractor Crew Task.

    This crew is responsible for extracting the structured information based on the passed configuration.
    """

    def __init__(self, tasks_config):
        """
        Initializes the Information Extractor Crew.

        Args:
            tasks_config (dict): Dictionary containing configuration for tasks.
        """
        self.tasks_config = tasks_config

    def agent_task(self, agent: Agent) -> Task:
        """Creates and returns an extraction task assigned to the agent.

        Args:
            agent (Agent): The agent that will execute the task.

        Returns:
            Task: A configured CrewAI task.
        """
        return Task(config=self.tasks_config, output_pydantic=ExtractedStructuredTerms, agent=agent)
