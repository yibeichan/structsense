import os
import yaml
from pathlib import Path

DEFAULT_CONFIG_DIR = Path(__file__).parent / "default_config_sie"

# Utility to load YAML config
def load_yaml_config(filename):
    with open(DEFAULT_CONFIG_DIR / filename, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Utility to recursively replace api_key in agent config
def replace_api_key(config, new_api_key):
    if isinstance(config, dict):
        for key, value in config.items():
            if key == "api_key":
                config[key] = new_api_key
            else:
                replace_api_key(value, new_api_key)
    return config

# Load configs from YAML files
def get_agent_config(api_key=None):
    config = load_yaml_config("ner_agent.yaml")
    if api_key:
        config = replace_api_key(config, api_key)
    return config

NER_TASK_CONFIG = load_yaml_config("ner_task.yaml")
EMBEDDER_CONFIG = load_yaml_config("embedding.yaml")
HUMAN_IN_LOOP_CONFIG = load_yaml_config("human_in_loop.yaml")
SEARCH_ONTOLOGY_KNOWLEDGE_CONFIG = load_yaml_config("search_ontology_knowledge.yaml") 