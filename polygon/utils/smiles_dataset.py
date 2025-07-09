import os
from torch.utils.data import IterableDataset

class SmilesDataset(IterableDataset):
    """Iterably read SMILES strings from a text file with minimal memory."""

    def __init__(self, path: str):
        self.path = path
        if not os.path.isfile(self.path):
            raise ValueError(f"File not found: {self.path}")
        # count lines once to provide __len__ without storing offsets
        with open(self.path, "rb") as f:
            self._length = sum(1 for _ in f)

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")

    def __len__(self):
        return self._length

