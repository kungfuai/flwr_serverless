from pathlib import Path
import pickle
from flwr.common import Parameters


class LocalStorageBackend:
    def __init__(self, directory: str = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.suffix = ".params"
    
    def get(self, key, default=None):
        filepath = self.directory / (key + self.suffix)
        if filepath.exists():
            with open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            return default

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value: Parameters):
        filepath = self.directory / (key + self.suffix)
        with open(filepath, "wb") as f:
            pickle.dump(value, f)
    
    def __len__(self):
        return len(self.directory.glob(f"*{self.suffix}"))
    
    def items(self):
        for filepath in self.directory.glob(f"*{self.suffix}"):
            with open(filepath, "rb") as f:
                yield filepath.name[:-len(self.suffix)], pickle.load(f)
