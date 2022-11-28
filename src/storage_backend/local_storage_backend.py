from pathlib import Path
import pickle
import time
from typing import Any
from flwr.common import Parameters


class LocalStorageBackend:
    def __init__(self, directory: str = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.suffix = ".params"

    def get(self, key, default=None):
        success_flag_file = self._get_success_flag_file(key)
        patience = 3
        while not success_flag_file.exists():
            print(f"\nwaiting for success flag of {key}")
            time.sleep(3)
            patience -= 1
            if patience == 0:
                return default
        filepath = self.directory / (key + self.suffix)
        if filepath.exists():
            with open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            return default

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value: Any):
        filepath = self.directory / (key + self.suffix)
        if value is None:
            raise ValueError("value must not be None")
        self._delete_success_flag(key)
        with open(filepath, "wb") as f:
            pickle.dump(value, f)
        self._put_success_flag(key)

    def _get_success_flag_file(self, key):
        return self.directory / ("success_" + key)

    def _delete_success_flag(self, key):
        filepath = self._get_success_flag_file(key)
        if filepath.exists():
            filepath.unlink()

    def _put_success_flag(self, key):
        filepath = self._get_success_flag_file(key)
        with open(filepath, "w") as f:
            f.write("")

    def __len__(self):
        return len(list(self.directory.glob(f"*{self.suffix}")))

    def items(self):
        for filepath in self.directory.glob(f"*{self.suffix}"):
            key_and_parameter = self.get_parameter(filepath)
            yield key_and_parameter

    def get_parameter(self, filepath):
        with open(filepath, "rb") as f:
            try:
                key = filepath.name[: -len(self.suffix)]
                parameters = self.get(key)
                return key, parameters
            except EOFError as e:
                print(f"EOFError: {e}")
                return None, None
