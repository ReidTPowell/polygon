from typing import List
import pandas as pd
from admet_ai import DescriptorCalculator


def compute_admet(smiles_list: List[str]) -> pd.DataFrame:
    """Compute ADMET-AI descriptors for a list of SMILES."""
    calc = DescriptorCalculator()
    admet_df = calc.calculate_descriptors(smiles_list)
    return admet_df


def prepare_dataset(smiles_csv: str, output_csv: str) -> str:
    """Load SMILES from ``smiles_csv`` and save a feature table with ADMET descriptors."""
    df = pd.read_csv(smiles_csv)
    smiles = df["smiles"].tolist()
    admet_df = compute_admet(smiles)
    out_df = pd.concat([df, admet_df], axis=1)
    out_df.to_csv(output_csv, index=False)
    return output_csv
