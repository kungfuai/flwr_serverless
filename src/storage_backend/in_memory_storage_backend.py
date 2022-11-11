class InMemoryStorageBackend:
    def __init__(self):
        self.model_store = {}
    
    def get(self, key, default=None):
        return self.model_store[key] if key in self.model_store else default

    def __getitem__(self, key):
        return self.model_store[key]

    def __setitem__(self, key, value):
        self.model_store[key] = value
