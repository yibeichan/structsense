# -*- coding: utf-8 -*-
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
# @File    : alignment_task.py
# @Software: PyCharm

from crewai import Agent, Task
from utils.utils import hybrid_search


class ConceptAlignmentTask:
    """Alignment Crew Task.

    This crew is responsible for performing the concept alignment.
    """

    def __init__(self, tasks_config):
        """
        Initializes the Alignment Crew.

        Args:
            tasks_config (dict): Dictionary containing configuration for tasks.
        """
        self.tasks_config = tasks_config

    def alignment_task(self, agent: Agent) -> Task:
        """Creates and returns an Alignment task assigned to the agent.

        Args:
            agent (Agent): The agent that will execute the task.

        Returns:
            Task: A configured CrewAI task.
        """
        return Task(config=self.tasks_config, agent=agent)
