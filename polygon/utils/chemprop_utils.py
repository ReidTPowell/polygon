import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDatapoint, MoleculeDataset
from chemprop.train import get_loss_func, train as _chemprop_train
from chemprop.models import MoleculeModel
from chemprop.utils import build_optimizer, build_lr_scheduler, load_checkpoint, save_checkpoint, load_scalers
from sklearn.metrics import average_precision_score, mean_absolute_error
from tqdm import trange


def chemprop_build_data_loader(
    smiles: List[str],
    properties: Optional[List[float]] = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> MoleculeDataLoader:
    """Build a MoleculeDataLoader for Chemprop."""
    if properties is None:
        properties = [None] * len(smiles)
    else:
        properties = [[float(p)] for p in properties]

    dataset = MoleculeDataset(
        [MoleculeDatapoint(smiles=[s], targets=prop) for s, prop in zip(smiles, properties)]
    )
    return MoleculeDataLoader(dataset=dataset, shuffle=shuffle, num_workers=num_workers)


def chemprop_predict(model: MoleculeModel, smiles: List[str], num_workers: int = 0) -> np.ndarray:
    """Predict properties using a Chemprop model."""
    loader = chemprop_build_data_loader(smiles, num_workers=num_workers)
    preds = np.array(model(loader))[:, 0]
    return preds


def chemprop_train(
    dataset_type: str,
    train_smiles: List[str],
    val_smiles: List[str],
    property_name: str,
    train_properties: List[float],
    val_properties: List[float],
    epochs: int,
    save_path: Path,
    num_workers: int = 0,
    use_gpu: bool = False,
) -> MoleculeModel:
    """Train a Chemprop model and save the best model to ``save_path``."""

    arg_list = [
        "--data_path",
        "foo.csv",
        "--dataset_type",
        dataset_type,
        "--save_dir",
        "foo",
        "--epochs",
        str(epochs),
        "--quiet",
    ] + ([] if use_gpu else ["--no_cuda"])

    args = TrainArgs().parse_args(arg_list)
    args.task_names = [property_name]
    args.train_data_size = len(train_smiles)

    torch.manual_seed(0)
    if not use_gpu:
        torch.use_deterministic_algorithms(True)

    train_loader = chemprop_build_data_loader(train_smiles, train_properties, shuffle=True, num_workers=num_workers)
    val_loader = chemprop_build_data_loader(val_smiles, val_properties, shuffle=False, num_workers=num_workers)

    model = MoleculeModel(args)
    loss_func = get_loss_func(args)
    optimizer = build_optimizer(model, args)
    scheduler = build_lr_scheduler(optimizer, args)

    best_score = float("inf") if args.minimize_score else -float("inf")
    val_metric = "PRC-AUC" if dataset_type == "classification" else "MAE"
    best_epoch = n_iter = 0
    for epoch in trange(args.epochs):
        n_iter = _chemprop_train(
            model=model,
            data_loader=train_loader,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            n_iter=n_iter,
        )

        val_preds = chemprop_predict(model=model, smiles=val_smiles)

        if dataset_type == "classification":
            val_score = average_precision_score(val_properties, val_preds)
            new_best = val_score > best_score
        elif dataset_type == "regression":
            val_score = mean_absolute_error(val_properties, val_preds)
            new_best = val_score < best_score
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        if new_best:
            best_score, best_epoch = val_score, epoch
            save_checkpoint(path=str(save_path), model=model, args=args)

    print(f"Best validation {val_metric} = {best_score:.6f} on epoch {best_epoch}")
    model = load_checkpoint(str(save_path), device=args.device)
    return model


def chemprop_load(model_path: Path, device: Optional[torch.device] = torch.device("cpu")) -> MoleculeModel:
    """Load a saved Chemprop model."""
    return load_checkpoint(path=str(model_path), device=device).eval()


def chemprop_load_scaler(model_path: Path):
    """Load a Chemprop model's scaler if it exists."""
    try:
        return load_scalers(path=str(model_path))[0]
    except Exception:
        return None
