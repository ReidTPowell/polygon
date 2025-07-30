import argparse
import pandas as pd

from polygon.data.prep import compute_admet
from polygon.scoring import load_scoring


def train_models(data_path: str, admet_df: pd.DataFrame, config):
    """Placeholder for model training pipeline."""
    # Implementation would train models using data_path and descriptors
    pass


def run_search(smiles, objectives, output_dir: str):
    """Placeholder for molecule search pipeline."""
    # Implementation would perform generative search using objectives
    pass


def main():
    parser = argparse.ArgumentParser(description="POLYGON pipeline")
    parser.add_argument("--scoring_yaml", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    smiles = df["smiles"].tolist()

    admet_df = compute_admet(smiles)
    train_models(args.data_path, admet_df, args)
    objectives = load_scoring(args.scoring_yaml)
    run_search(smiles, objectives, args.output_dir)


if __name__ == "__main__":
    main()
