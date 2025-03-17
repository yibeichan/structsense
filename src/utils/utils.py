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
# @File    : utils.py
# @Software: PyCharm
import logging
import sys
from pathlib import Path
from typing import Union, Dict, List
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)



def required_config_exists(data: dict, type: str) -> bool:
    """
    Check if all required configuration keys exist for the specified type (agent or task).

    - If `type` is "agent", the function verifies whether each agent in the provided
      dictionary contains the required configuration keys.
    - If `type` is "task", the function checks if all tasks contain the necessary keys.

    Required keys for an agent:
    - "role": Defines the agent's function and expertise.
    - "goal": Specifies the agent's objective for decision-making.
    - "backstory": Provides context and personality to enrich interactions.
    - "llm": Specifies the language model used.

    Required keys for a task:
    - "description": Provides details about the task.
    - "expected_output": Defines what the task should produce.
    - "agent": Specifies which agent is responsible for executing the task.

    Parameters:
        data (dict): Dictionary containing configurations for agents or tasks.
        type (str): The type of configuration to check ("agent" or "task").

    Returns:
        bool: True if all agents or tasks have the required keys, False otherwise.

    Example:
        >>> config = {
        ...     "extractor_agent": {
        ...         "role": "role of the agent for {topic}",
        ...         "goal": "goal of the agent for a {topic}",
        ...         "backstory": "You are an experienced research specialist...",
        ...         "llm": "openai/gpt-4o-mini"
        ...     },
        ...     "judge_agent": {
        ...         "role": "Senior Research Specialist for {topic}",
        ...         "goal": "Find comprehensive and accurate information...",
        ...         "backstory": "You are an experienced research specialist...",
        ...         "llm": "openai/gpt-4o-mini"
        ...     }
        ... }
        >>> required_config_exists(config, "agent")
        True

        >>> task_config = {
        ...     "extractor_agent_task": {
        ...         "description": "some description for {topic}",
        ...         "expected_output": "some output {topic}",
        ...         "agent": "extractor_agent"
        ...     },
        ...     "judge_agent_task": {
        ...         "description": "some description for {topic}",
        ...         "expected_output": "some output {topic}",
        ...         "agent": "judge_agent"
        ...     }
        ... }
        >>> required_config_exists(task_config, "task")
        True

        >>> incomplete_task_config = {
        ...     "extractor_agent_task": {
        ...         "description": "some description for {topic}"
        ...     }
        ... }
        >>> required_config_exists(incomplete_task_config, "task")
        False
    """
    required_keys = {
        "agent": {"role", "goal", "backstory", "llm"},
        "task": {"description", "expected_output", "agent"}
    }

    if type in required_keys:
        for item_name, item_config in data.items():
            if not isinstance(item_config, dict) or not required_keys[type].issubset(item_config.keys()):
                return False  # Return False immediately if any item is missing a required key

    return True


def load_config(config: Union[str, Path, Dict], type: str) -> dict:
    """
    Loads the configuration from a YAML file

    Args:
        config (Union[str, Path, dict]): The configuration source.
        type (str): The type of the configuration, e.g., agents or tasks

    Returns:
        dict: Parsed LLM configuration.

    Raises:
        FileNotFoundError: If the YAML file is not found.
        ValueError: If the input is not a valid YAML file or dictionary.
        yaml.YAMLError: If there is an error parsing the YAML configuration.
    """
    if isinstance(config, dict):
        if required_config_exists(config, type):
            return config  # Directly use the dictionary
        else:
            raise KeyError(f"Required config details not found in config file")

    # Try different path resolutions for config file
    if isinstance(config, str):
        paths_to_try = [
            Path(config),  # As provided
            Path.cwd() / config,  # Relative to current directory
            Path(config).absolute(),  # Absolute path
            Path(config).resolve()  # Resolved path (handles .. and .)
        ]

        logger.info(f"Trying config paths: {[str(p) for p in paths_to_try]}")

        # Find first existing path with valid extension
        config_path = next(
            (p for p in paths_to_try if p.exists() and p.suffix.lower() in {".yml", ".yaml"}),
            paths_to_try[0]  # Default to first path if none exist
        )
    else:
        config_path = Path(config)

    if not config_path.exists() or config_path.suffix.lower() not in {".yml", ".yaml"}:
        error_msg = (
                f"Invalid configuration: {config}\n"
                f"Expected a YAML file (.yml or .yaml) or a dictionary.\n"
                "Tried the following paths:\n"
                + "\n".join(f"- {p}" for p in paths_to_try)
        )
        raise ValueError(error_msg)

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config_file_content = yaml.safe_load(file)
            logger.info(f"file processing - {file}, type: {type}")
            if required_config_exists(config_file_content, type):
                return config_file_content
            else:
                raise KeyError(f"Required config details not found in config file")

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config}: {e}")
