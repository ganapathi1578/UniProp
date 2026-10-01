from .registry import registry

class AGQAAdapter:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        return []

registry.register('agqa', AGQAAdapter)
