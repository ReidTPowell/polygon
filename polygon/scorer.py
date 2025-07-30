from typing import Iterable, List
from rdkit.Chem import Mol

from .scoring import load_scoring


def score_molecules(molecules: Iterable[Mol], scoring_yaml: str) -> List[float]:
    """Score molecules according to objectives defined in a YAML file."""
    objectives = load_scoring(scoring_yaml)
    scores = []
    for mol in molecules:
        total_score = sum(obj.evaluate(mol) for obj in objectives)
        scores.append(total_score)
    return scores
