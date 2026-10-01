class DatasetRegistry:
    def __init__(self):
        self.datasets = {}

    def register(self, name, adapter_cls):
        self.datasets[name] = adapter_cls

    def get(self, name):
        return self.datasets.get(name)

registry = DatasetRegistry()
