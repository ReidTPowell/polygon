import numpy as np

if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = type(
        "VisibleDeprecationWarning",
        (Warning,),
        {},
    )

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Optional
from chemprop.args import TrainArgs
from chemprop.data import MoleculeDatapoint, MoleculeDataset

# Chemprop versions prior to 2.0 expose a ``MoleculeDataLoader`` class while
# newer versions export ``build_dataloader``.  Import whichever is available and
# provide a unified ``_build_dataloader`` function.
try:  # Chemprop >=2.0
    from chemprop.data import build_dataloader as _build_dataloader
except Exception:  # pragma: no cover - older Chemprop versions
    try:  # Chemprop <2.0
        from chemprop.data import MoleculeDataLoader as _ChempropDataLoader

        def _build_dataloader(
            dataset: MoleculeDataset,
            batch_size: int = 64,
            shuffle: bool = False,
            num_workers: int = 0,
        ) -> DataLoader:
            """Fallback DataLoader builder for older Chemprop releases."""

            return _ChempropDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )

    except Exception as exc:  # pragma: no cover - should not happen
        raise ImportError(
            "Could not import a Chemprop DataLoader."
        ) from exc
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
) -> DataLoader:
    """Build a ``DataLoader`` compatible with Chemprop."""
    if properties is None:
        properties = [None] * len(smiles)
    else:
        properties = [[float(p)] for p in properties]

    dataset = MoleculeDataset(
        [MoleculeDatapoint(smiles=[s], targets=prop) for s, prop in zip(smiles, properties)]
    )
    return _build_dataloader(
        dataset=dataset,
        batch_size=len(dataset),
        shuffle=shuffle,
        num_workers=num_workers,
    )


def _to_device(batch, device):
    """Move a Chemprop TrainingBatch to the specified device."""
    if hasattr(batch, "to"):
        return batch.to(device)

    try:
        bmg, V_d, X_d, *rest = batch
    except Exception:
        raise AttributeError(
            "Unexpected batch type returned by Chemprop dataloader"
        )

    bmg = bmg.to(device)
    V_d = V_d.to(device) if V_d is not None else None
    X_d = X_d.to(device) if X_d is not None else None
    moved_rest = [x.to(device) if x is not None else None for x in rest]
    return (bmg, V_d, X_d, *moved_rest)


def chemprop_predict(model: MoleculeModel, smiles: List[str], num_workers: int = 0) -> np.ndarray:
    """Predict properties using a Chemprop model."""
    loader = chemprop_build_data_loader(smiles, num_workers=num_workers)

    all_preds = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            if isinstance(batch, (list, tuple)):
                bmg, V_d, X_d, *_ = batch
                batch_preds = model(bmg, V_d, X_d)
            else:
                batch_preds = model(batch)
            if isinstance(batch_preds, (list, tuple)):
                batch_preds = batch_preds[0]
            all_preds.append(batch_preds.detach().cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)[:, 0]
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
    use_gpu: bool = True,
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
    args.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    args.task_names = [property_name]
    args.train_data_size = len(train_smiles)

    torch.manual_seed(0)
    if not use_gpu:
        torch.use_deterministic_algorithms(True)

    train_loader = chemprop_build_data_loader(
        train_smiles,
        train_properties,
        shuffle=True,
        num_workers=num_workers,
    )
    chemprop_build_data_loader(
        val_smiles,
        val_properties,
        shuffle=False,
        num_workers=num_workers,
    )

    model = MoleculeModel(args).to(args.device)
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


def chemprop_load(
    model_path: Path,
    device: Optional[torch.device] = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
) -> MoleculeModel:
    """Load a saved Chemprop model."""
    return load_checkpoint(path=str(model_path), device=device).eval()


def chemprop_load_scaler(model_path: Path):
    """Load a Chemprop model's scaler if it exists."""
    try:
        return load_scalers(path=str(model_path))[0]
    except Exception:
        return None
