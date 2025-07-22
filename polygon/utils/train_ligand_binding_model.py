import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

from .chemprop_utils import chemprop_train


def train_ligand_binding_model(
    training_csv: str,
    output_path: str,
    dataset_type: str = "regression",
    epochs: int = 30,
) -> Path:
    """Train a Chemprop reward model.

    ``training_csv`` must contain ``smiles`` and ``affinity`` columns.
    The best model is saved to ``output_path``.
    """

    df = pd.read_csv(training_csv)
    if "smiles" not in df.columns or "affinity" not in df.columns:
        raise ValueError("CSV must contain 'smiles' and 'affinity' columns")

    smiles = df["smiles"].tolist()
    props = df["affinity"].tolist()

    train_smi, val_smi, train_p, val_p = train_test_split(
        smiles, props, test_size=0.1, random_state=0
    )

    model = chemprop_train(
        dataset_type=dataset_type,
        train_smiles=train_smi,
        val_smiles=val_smi,
        property_name="affinity",
        train_properties=train_p,
        val_properties=val_p,
        epochs=epochs,
        save_path=Path(output_path),
        num_workers=0,
        use_gpu=False,
    )

    logging.info("Model saved to %s", output_path)
    return Path(output_path)
