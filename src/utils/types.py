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
# @File    : types.py
# @Software: PyCharm

from pydantic import BaseModel
from typing import List, Dict


from pydantic import BaseModel
from typing import List, Dict


class NEREntity(BaseModel):
    """Represents a named entity recognized (NER) in text."""
    entity: str
    label: str
    sentence: str
    start: int
    end: int
    paper_location: str
    paper_title: str
    doi: str


class AlignedNEREntity(NEREntity):
    """Represents an aligned Named Entity Recognition (NER) term with ontology mapping."""
    ontology_id: str
    ontology_label: str


class AlignedNERTerms(BaseModel):
    """Pydantic model to validate aligned NER terms."""
    aligned_ner_terms: Dict[str, List[AlignedNEREntity]]

class ExtractedNERTerms(BaseModel):
    """Pydantic model to validate extracted NER terms."""
    extracted_ner_terms: Dict[str, List[NEREntity]]
