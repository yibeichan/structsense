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


from rdflib import Graph, RDF, RDFS, OWL, URIRef, Namespace
import pandas as pd
import os
import re
from urllib.parse import urlparse

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
                return False

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


#  ontology namespaces
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
DC = Namespace("http://purl.org/dc/elements/1.1/")
DCT = Namespace("http://purl.org/dc/terms/")
IAO = Namespace("http://purl.obolibrary.org/obo/IAO_")
OBO = Namespace("http://purl.obolibrary.org/obo/")

# common annotation properties
IAO_EDITORS_NOTE = URIRef("http://purl.obolibrary.org/obo/IAO_0000116")  # Editor's note
IAO_CURATORS_NOTE = URIRef("http://purl.obolibrary.org/obo/IAO_0000233")  # Curator's note


def extract_ontology_metadata(file_path, output_format="dataframe"):
    """
    Extract comprehensive metadata from an ontology file.

    Parameters:
    -----------
    file_path : str
        Path to the ontology file (OWL, RDF, TTL, etc.).
    output_format : str, optional
        Output format: "dataframe" (default) or "dict".

    Returns:
    --------
    pandas.DataFrame or list of dict
        Metadata about ontology classes.
    """
    g = Graph()

    # Determine RDF format based on file extension
    format_map = {"owl": "xml", "ttl": "turtle", "rdf": "xml", "n3": "n3", "jsonld": "json-ld", "nt": "nt"}
    file_extension = file_path.split('.')[-1].lower()
    rdf_format = format_map.get(file_extension, "xml")

    try:
        g.parse(file_path, format=rdf_format)
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        return None

    classes_data = []
    all_classes = set()
    class_types = [OWL.Class, RDFS.Class]

    for class_type in class_types:
        for cls in g.subjects(RDF.type, class_type):
            all_classes.add(cls)

    for cls in all_classes:
        if not isinstance(cls, URIRef):
            continue

        class_uri = str(cls)
        class_name = class_uri.split("/")[-1] if "/" in class_uri else class_uri

        class_record = {
            'class_id': class_name,
            'class_uri': class_uri,
            'label': None,
            'definition': None,
            'synonyms': []
        }

        # Extract label
        labels = list(g.objects(cls, RDFS.label))
        if labels:
            class_record['label'] = str(labels[0])

        # Extract definition
        definitions = list(g.objects(cls, RDFS.comment))
        if definitions:
            class_record['definition'] = str(definitions[0])

        # Extract synonyms
        synonym_properties = [SKOS.altLabel, IAO_EDITORS_NOTE]
        for syn_prop in synonym_properties:
            for syn in g.objects(cls, syn_prop):
                class_record['synonyms'].append(str(syn))

        classes_data.append(class_record)

    return pd.DataFrame(classes_data) if output_format == "dataframe" else classes_data


def process_ontology(file_path, output_file=None):
    """ Ontology Metadata Extraction Module

    This module provides functionality to extract and process metadata from ontology files
    in various RDF formats (OWL, TTL, RDF, etc.). It supports extracting labels, synonyms,
    annotations, and relationships among ontology classes.

    Finally, the result is saved into a CSV file.

    Parameters:
    -----------
    file_path : str
        Path to the ontology file (OWL, RDF, TTL, etc.).
    output_file : str, optional
        Path to save the processed metadata as a CSV file.

    Returns:
    --------
    pandas.DataFrame
        Extracted ontology metadata.

    Example output for the cell ontology
    class_id	class_uri	type	equivalent_to	broader	narrower	related	label	definition	related_synonyms	synonyms	exact_synonyms	alt_definitions	narrow_synonyms	broad_synonyms	editors_note	description	consider	curators_note
    UBERON_0009571	http://purl.obolibrary.org/obo/UBERON_0009571	uberon	[]	[]	[]	[]	ventral midline	In protostomes (such as insects, snails and worms) as well as deuterostomes (vertebrates), the midline is an embryonic region that functions in patterning of the adjacent nervous tissue. The ventral midline in insects is a cell population extending along the ventral surface of the embryo and is the region from which cells detach to form the ventrally located nerve cords. In vertebrates, the midline is originally located dorsally. During development, it folds inwards and becomes the ventral part of the dorsally located neural tube and is then called the ventral midline, or floor plate.	[]	[]	[]	[]	[]	[]	[]		[]	[]
    GO_2000973	http://purl.obolibrary.org/obo/GO_2000973	go	[]	[]	[]	[]	regulation of pro-B cell differentiation	Any process that modulates the frequency, rate or extent of pro-B cell differentiation.	['regulation of pro-B cell development']	['regulation of pro-B cell development', 'regulation of pro-B lymphocyte differentiation']	['regulation of pro-B lymphocyte differentiation']	[]	[]	[]	[]		[]	[]
    CL_4033072	http://purl.obolibrary.org/obo/CL_4033072	cl	[]	[]	[]	[]	cycling gamma-delta T cell	A(n) gamma-delta T cell that is cycling.	[]	['proliferating gamma-delta T cell']	['proliferating gamma-delta T cell']	[]	[]	[]	[]		[]	[]
    """
    df = extract_ontology_metadata(file_path, output_format="dataframe")

    if df is None:
        logger.error(f"Failed to process ontology file: {file_path}")
        return None

    if output_file:
        df.to_csv(output_file, index=False)
        logger.info(f"Ontology metadata saved to {output_file}")

    return df
