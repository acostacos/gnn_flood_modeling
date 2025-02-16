import time

class TrainingStats:
    def __init__(self):
        self.end_time = None
    
    def start_train(self):
        self.train_start_time = time.time()
        self.train_epoch_loss = []
    
    def add_train_loss(self, loss):
        self.train_epoch_loss.append(loss)

    def end_train(self):
        self.train_end_time = time.time()
