from pathlib import Path
import pickle
import time
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
        return len(list(self.directory.glob(f"*{self.suffix}")))

    def items(self):
        for filepath in self.directory.glob(f"*{self.suffix}"):
            key_and_parameter = self.get_parameter(filepath)
            while key_and_parameter is None:
                print("EOFError, trying again")
                time.sleep(1)
            yield key_and_parameter

    def get_parameter(self, filepath):
        with open(filepath, "rb") as f:
            try:
                return filepath.name[: -len(self.suffix)], pickle.load(f)
            except EOFError:
                return None
