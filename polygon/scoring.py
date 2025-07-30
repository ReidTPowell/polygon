from typing import List
from rdkit.Chem import Mol
import yaml

from .utils.custom_scoring_fcn import (
    QED_custom,
    SAScorer,
    LigandEfficancy,
    BBBScore,
)
from .utils.scoring_function import MinMaxGaussianModifier


class ScoringObjective:
    def __init__(self, cfg: dict):
        self.category = cfg["category"]
        self.name = cfg["name"]
        self.minimize = cfg["minimize"]
        self.mu = cfg["mu"]
        self.sigma = cfg["sigma"]
        self.file = cfg.get("file")

        modifier = MinMaxGaussianModifier(mu=self.mu, sigma=self.sigma, minimize=self.minimize)
        if self.category == "ligand_efficiency":
            if not self.file:
                raise ValueError("ligand_efficiency objective requires 'file' for model path")
            self.model = LigandEfficancy(score_modifier=modifier, model_path=self.file)
        elif self.category == "qed":
            self.model = QED_custom(score_modifier=modifier)
        elif self.category == "sa":
            self.model = SAScorer(score_modifier=modifier)
        elif self.category == "bbb":
            self.model = BBBScore(score_modifier=modifier)
        else:
            raise ValueError(f"Unknown category: {self.category}")

    def evaluate(self, molecule: Mol) -> float:
        """Compute normalized score for a molecule."""
        if isinstance(molecule, str):
            smiles = molecule
        else:
            from rdkit import Chem
            smiles = Chem.MolToSmiles(molecule)
        return self.model.score(smiles)


def load_scoring(yaml_path: str) -> List[ScoringObjective]:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return [ScoringObjective(o) for o in config.get("scoring", [])]
