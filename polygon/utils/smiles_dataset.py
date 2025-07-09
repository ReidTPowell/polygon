import os
from torch.utils.data import Dataset

class SmilesDataset(Dataset):
    """Dataset for reading SMILES strings lazily from a file."""

    def __init__(self, path: str):
        self.path = path
        if not os.path.isfile(self.path):
            raise ValueError(f"File not found: {self.path}")
        self.offsets = []
        offset = 0
        with open(self.path, "rb") as f:
            for line in f:
                self.offsets.append(offset)
                offset = f.tell()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx: int):
        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline().decode("utf-8")
        return line.rstrip("\n")

