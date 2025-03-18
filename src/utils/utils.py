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


#  common ontology namespaces and properties
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
DC = Namespace("http://purl.org/dc/elements/1.1/")
DCT = Namespace("http://purl.org/dc/terms/")
IAO = Namespace("http://purl.obolibrary.org/obo/IAO_")
OBO = Namespace("http://purl.obolibrary.org/obo/")
OBO_SYNONYM = Namespace("http://purl.obolibrary.org/obo/synonym_")

# specific OBO synonym types
OBO_EXACT_SYNONYM = URIRef("http://purl.obolibrary.org/obo/IAO_0000118")
OBO_RELATED_SYNONYM = URIRef("http://purl.obolibrary.org/obo/IAO_0000118")
OBO_NARROW_SYNONYM = URIRef("http://purl.obolibrary.org/obo/IAO_0000118")
OBO_BROAD_SYNONYM = URIRef("http://purl.obolibrary.org/obo/IAO_0000118")

#  additional semantic relationship properties
SKOS_BROADER = SKOS.broader  # More general concept or superclass
SKOS_NARROWER = SKOS.narrower  # More specific concept or subclass
SKOS_RELATED = SKOS.related  # Related concepts

#  additional annotation properties
RDFS_SEE_ALSO = RDFS.seeAlso  # Additional resources or references
DC_SUBJECT = DC.subject  # The subject or domain of the class
DC_COVERAGE = DC.coverage  # The scope of the class

#  editor's note and curator's note properties
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

    # Determine format based on file extension
    file_extension = file_path.split('.')[-1].lower()
    format_map = {
        "owl": "xml",
        "ttl": "turtle",
        "rdf": "xml",
        "n3": "n3",
        "jsonld": "json-ld",
        "nt": "nt"
    }
    rdf_format = format_map.get(file_extension, "xml")

    try:
        g.parse(file_path, format=rdf_format)
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        try:
            # Try alternative format if parsing fails
            alternative_format = "xml" if rdf_format != "xml" else "turtle"
            logger.info(f"Trying alternative format: {alternative_format}")
            g.parse(file_path, format=alternative_format)
        except Exception as e2:
            logger.error(f"Failed to parse with alternative format: {e2}")
            return None


    filename = os.path.basename(file_path)
    ontology_name = filename.split('.')[0].lower()

    # Try to determine ontology name from the ontology IRI if available
    for ont in g.subjects(RDF.type, OWL.Ontology):
        # Get ontology IRI
        ont_str = str(ont)

        # Parse the ontology IRI to extract a better name
        parsed_uri = urlparse(ont_str)
        path_parts = parsed_uri.path.strip('/').split('/')

        if path_parts:
            # Use the last meaningful segment
            for part in reversed(path_parts):
                if part and not part.endswith(('.owl', '.rdf', '.ttl')):
                    ontology_name = part.lower()
                    break

        # Check if there's an explicit ontology label
        for label in g.objects(ont, RDFS.label):
            label_str = str(label).lower()
            # Extract the main part of the ontology name from the label
            match = re.search(r'(\w+)\s+ontology', label_str)
            if match:
                ontology_name = match.group(1)
            else:
                words = label_str.split()
                if len(words) > 0:
                    ontology_name = words[0]
            break

    # Define important annotation properties to extract
    annotation_properties = [
        RDFS.label,
        RDFS.comment,
        RDFS.isDefinedBy,
        RDFS.seeAlso,  # Additional resources or references
        SKOS.definition,
        SKOS.prefLabel,
        SKOS.altLabel,
        SKOS.notation,
        SKOS.broader,  # More general concept
        SKOS.narrower,  # More specific concept
        SKOS.related,  # Related concepts
        DC.title,
        DC.description,
        DC.subject,  # The subject or domain of the class
        DC.coverage,  # The scope of the class
        DCT.title,
        DCT.description,
        OWL.versionInfo,
        OWL.deprecated,
        OWL.priorVersion,
        OWL.incompatibleWith,
        URIRef("http://purl.obolibrary.org/obo/IAO_0000115"),  # definition
        URIRef("http://purl.obolibrary.org/obo/IAO_0000111"),  # preferred label
        URIRef("http://purl.obolibrary.org/obo/IAO_0000112"),  # example of usage
        URIRef("http://purl.obolibrary.org/obo/IAO_0000118"),  # alternative term
        IAO_EDITORS_NOTE,  # Editor's note
        IAO_CURATORS_NOTE,  # Curator's note
        URIRef("http://purl.obolibrary.org/obo/IAO_0000232"),  # curator comment
    ]

    # Add synonym properties - these are commonly used in ontologies
    synonym_properties = [
        URIRef("http://purl.obolibrary.org/obo/IAO_0000118"),  # alternative term
        URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"),
        URIRef("http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym"),
        URIRef("http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym"),
        URIRef("http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym"),
        URIRef("http://purl.org/obo/owl/oboInOwl#hasExactSynonym"),
        URIRef("http://purl.org/obo/owl/oboInOwl#hasRelatedSynonym"),
        URIRef("http://purl.org/obo/owl/oboInOwl#hasNarrowSynonym"),
        URIRef("http://purl.org/obo/owl/oboInOwl#hasBroadSynonym"),
        SKOS.exactMatch,
        SKOS.closeMatch,
        SKOS.relatedMatch,
        SKOS.broadMatch,
        SKOS.narrowMatch,
        SKOS.related,
        SKOS.altLabel,
    ]

    # Lists to store class data
    classes_data = []

    # Extract classes (both OWL classes and RDFS classes)
    class_types = [OWL.Class, RDFS.Class]
    all_classes = set()

    for class_type in class_types:
        for cls in g.subjects(RDF.type, class_type):
            all_classes.add(cls)

    # Also check for implicit classes (subjects of rdfs:subClassOf)
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        all_classes.add(s)
        all_classes.add(o)

    # Process each class
    for cls in all_classes:
        # Skip blank nodes and non-URI classes
        if not isinstance(cls, URIRef):
            continue

        # Extract class name intelligently
        class_uri = str(cls)

        # Extract local name from URI
        if '#' in class_uri:
            class_name = class_uri.split('#')[-1]
        elif '/' in class_uri:
            class_name = class_uri.split('/')[-1]
        else:
            class_name = class_uri

        # If an OBO class, extract the proper identifier
        if class_uri.startswith("http://purl.obolibrary.org/obo/"):
            match = re.search(r'([A-Za-z]+)_\d+', class_name)
            if match:
                ontology_name = match.group(1).lower()

        # Initialize class record
        class_record = {
            'class_id': class_name,
            'class_uri': class_uri,
            'ontology': ontology_name,
            'equivalent_to': [],
            'broader': [],  # SKOS broader concepts
            'narrower': [],  # SKOS narrower concepts
            'related': [],  # SKOS related concepts
        }

        # Extract labels first (priority)
        labels = list(g.objects(cls, RDFS.label))
        if not labels:
            # Try alternative label properties
            for label_prop in [SKOS.prefLabel, DC.title, DCT.title,
                               URIRef("http://purl.obolibrary.org/obo/IAO_0000111")]:
                labels = list(g.objects(cls, label_prop))
                if labels:
                    break

        if labels:
            class_record['label'] = str(labels[0])
        else:
            class_record['label'] = class_name

        # Extract definitions
        definitions = []
        for def_prop in [SKOS.definition, DC.description, DCT.description, RDFS.comment,
                         URIRef("http://purl.obolibrary.org/obo/IAO_0000115")]:
            for defn in g.objects(cls, def_prop):
                if str(defn) not in definitions:
                    definitions.append(str(defn))

        if definitions:
            class_record['definition'] = definitions[0]  # Primary definition
            if len(definitions) > 1:
                class_record['alt_definitions'] = definitions[1:]  # Alternative definitions

        # Extract SKOS broader concepts
        for broader in g.objects(cls, SKOS.broader):
            if isinstance(broader, URIRef):
                broader_uri = str(broader)
                class_record['broader'].append(broader_uri)

        # Extract SKOS narrower concepts
        for narrower in g.objects(cls, SKOS.narrower):
            if isinstance(narrower, URIRef):
                narrower_uri = str(narrower)
                class_record['narrower'].append(narrower_uri)

        # Extract SKOS related concepts
        for related in g.objects(cls, SKOS.related):
            if isinstance(related, URIRef):
                related_uri = str(related)
                class_record['related'].append(related_uri)

        # Extract equivalent classes
        for eq_class in g.objects(cls, OWL.equivalentClass):
            if isinstance(eq_class, URIRef):
                class_record['equivalent_to'].append(str(eq_class))

                # Extract editor's note
        editors_notes = []
        for note in g.objects(cls, IAO_EDITORS_NOTE):
            editors_notes.append(str(note))
        if editors_notes:
            class_record['editors_note'] = editors_notes

        # Extract curator's note
        curators_notes = []
        for note in g.objects(cls, IAO_CURATORS_NOTE):
            curators_notes.append(str(note))
        if curators_notes:
            class_record['curators_note'] = curators_notes

        # Extract description
        descriptions = []
        for desc_prop in [DC.description, DCT.description]:
            for desc in g.objects(cls, desc_prop):
                descriptions.append(str(desc))
        if descriptions:
            class_record['description'] = descriptions[0] if len(descriptions) == 1 else descriptions

        # Extract synonyms - categorized by type
        exact_synonyms = []
        related_synonyms = []
        narrow_synonyms = []
        broad_synonyms = []
        alt_labels = []

        # Check for exact synonyms
        for exact_syn_prop in [
            URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"),
            URIRef("http://purl.org/obo/owl/oboInOwl#hasExactSynonym"),
            SKOS.exactMatch
        ]:
            for syn in g.objects(cls, exact_syn_prop):
                if str(syn) not in exact_synonyms:
                    exact_synonyms.append(str(syn))

        # Check for related synonyms
        for related_syn_prop in [
            URIRef("http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym"),
            URIRef("http://purl.org/obo/owl/oboInOwl#hasRelatedSynonym"),
            SKOS.relatedMatch,
            SKOS.related
        ]:
            for syn in g.objects(cls, related_syn_prop):
                if str(syn) not in related_synonyms:
                    related_synonyms.append(str(syn))

        # Check for narrow synonyms
        for narrow_syn_prop in [
            URIRef("http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym"),
            URIRef("http://purl.org/obo/owl/oboInOwl#hasNarrowSynonym"),
            SKOS.narrowMatch
        ]:
            for syn in g.objects(cls, narrow_syn_prop):
                if str(syn) not in narrow_synonyms:
                    narrow_synonyms.append(str(syn))

        # Check for broad synonyms
        for broad_syn_prop in [
            URIRef("http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym"),
            URIRef("http://purl.org/obo/owl/oboInOwl#hasBroadSynonym"),
            SKOS.broadMatch
        ]:
            for syn in g.objects(cls, broad_syn_prop):
                if str(syn) not in broad_synonyms:
                    broad_synonyms.append(str(syn))

        # Check for alternative labels/terms
        for alt_label_prop in [SKOS.altLabel, URIRef("http://purl.obolibrary.org/obo/IAO_0000118")]:
            for alt in g.objects(cls, alt_label_prop):
                alt_str = str(alt)
                if alt_str not in alt_labels:
                    alt_labels.append(alt_str)

        # Add synonyms to the class record
        if exact_synonyms:
            class_record['exact_synonyms'] = exact_synonyms
        if related_synonyms:
            class_record['related_synonyms'] = related_synonyms
        if narrow_synonyms:
            class_record['narrow_synonyms'] = narrow_synonyms
        if broad_synonyms:
            class_record['broad_synonyms'] = broad_synonyms
        if alt_labels:
            class_record['alt_labels'] = alt_labels

        # Combine all types of synonyms into one field for convenience
        all_synonyms = exact_synonyms + related_synonyms + narrow_synonyms + broad_synonyms + alt_labels
        if all_synonyms:
            class_record['all_synonyms_combined'] = list(set(all_synonyms))  # Remove duplicates

        # Add the class record to our collection
        classes_data.append(class_record)

    # Return the data in the requested format
    if output_format == "dataframe":
        return pd.DataFrame(classes_data)
    else:  # Return as dict/list
        return classes_data


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
    # Extract metadata
    df = extract_ontology_metadata(file_path, output_format="dataframe")

    if df is None:
        logger.error(f"Failed to process ontology file: {file_path}")
        return None

    columns_to_keep = [
        'class_id', 'class_uri', 'ontology',
        'equivalent_to', 'broader', 'narrower', 'related',
        'label', 'definition', 'related_synonyms', 'all_synonyms_combined',
        'exact_synonyms', 'alt_definitions', 'narrow_synonyms',
        'broad_synonyms', 'editors_note', 'description', 'curators_note'
    ]

    # Ensure list columns are properly formatted
    list_columns = [
        'synonyms', 'exact_synonyms', 'related_synonyms', 'narrow_synonyms',
        'broad_synonyms', 'alt_labels',
        'equivalent_to', 'broader', 'narrower', 'related',
        'editors_note', 'curators_note', 'alt_definitions'
    ]

    for col in list_columns:
        if col in df.columns:
            # Convert any non-list values to lists
            df[col] = df[col].apply(
                lambda x: x if isinstance(x, list) else [str(x)] if pd.notna(x) else []
            )

    # Keep only columns that exist in the dataframe and are in the columns_to_keep list
    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[available_columns]

    # Save to file if requested
    if output_file:
        df.to_csv(output_file, index=False)
        logger.info(f"Saved ontology metadata to {output_file}")

    return df






