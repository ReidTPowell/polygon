from pathlib import Path
from typing import List

from chemprop.args import TrainArgs
from chemprop.train import run_training
from chemprop.utils import load_checkpoint
from chemprop.data import MoleculeDatapoint, MoleculeDataset

from ..utils.chemprop_utils import chemprop_build_data_loader


def train_chemprop(data_path: str, save_dir: str, **hyperparameters) -> None:
    """Train a ChemProp regression model and save best checkpoint."""
    arg_list = ["--data_path", data_path, "--save_dir", save_dir, "--quiet"]
    for key, val in hyperparameters.items():
        arg_list.extend([f"--{key}", str(val)])
    args = TrainArgs().parse_args(arg_list)
    run_training(args)


def predict_with_chemprop(model_path: str, smiles_list: List[str]) -> List[float]:
    """Load a ChemProp model and return predictions for a list of SMILES."""
    model = load_checkpoint(model_path, device="cpu").eval()
    loader = chemprop_build_data_loader(smiles_list)
    preds = []
    import torch
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(model.device) if hasattr(batch, "to") else batch
            out = model(batch)
            if isinstance(out, (list, tuple)):
                out = out[0]
            preds.extend(out.squeeze().cpu().numpy().tolist())
    return preds
