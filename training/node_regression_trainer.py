from .base_trainer import BaseTrainer

class NodeRegressionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
