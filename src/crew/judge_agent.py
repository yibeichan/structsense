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
# @File    : judge_agent.py
# @Software: PyCharm

from crewai import LLM, Agent

class JudgeAgent:
    def __init__(self, agents_config):
        """Initializes the Judge Agent.

        Args:
            agents_config (dict): Dictionary containing configuration for agents.
        """
        self.agents_config = agents_config

    def judge_agent(self) -> Agent:
        """Creates and returns an alignment agent based on the configuration.

        The judge agent is defined with a role, goal, backstory, and an LLM model.

        Returns:
            Agent: A configured CrewAI agent.
        """
        extractor_config = self.agents_config
        role_config = extractor_config.get("role")
        goal_config = extractor_config.get("goal")
        backstory_config = extractor_config.get("backstory")
        llm_config = extractor_config.get("llm")

        return Agent(
            role=role_config,
            goal=goal_config,
            backstory=backstory_config,
            # meaning the agent has to complete the task,
            # it would not be able delegate task to other agent to complete the task
            # we want each agent to complete the assigned task
            allow_delegation=False,
            verbose=True,
            llm=LLM(**llm_config),
        )
