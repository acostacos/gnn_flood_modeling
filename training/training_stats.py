import time
import matplotlib.pyplot as plt
import numpy as np

class TrainingStats:
    def __init__(self):
        self.train_epoch_loss = []

    def start_train(self):
        self.train_start_time = time.time()

    def end_train(self):
        self.train_end_time = time.time()

    def get_train_time(self):
        return self.train_end_time - self.train_start_time

    def add_train_loss(self, loss):
        self.train_epoch_loss.append(loss)

    def get_train_loss(self):
        return self.train_epoch_loss

    def start_val(self):
        self.val_start_time = time.time()

    def end_val(self, num_samples: int = 1):
        self.val_end_time = time.time()
        self.inference_time = (self.val_end_time - self.val_start_time) / num_samples

    def get_val_time(self):
        return self.inference_time

    def set_val_loss(self, loss):
        self.val_loss = loss

    def get_val_loss(self):
        return self.val_loss

    def print_stats_summary(self):
        if len(self.train_epoch_loss) > 0:
            print(f'Final training Loss: {self.train_epoch_loss[-1]:.4f}')
            print(f'Average training Loss: {np.mean(self.train_epoch_loss):.4f}')
            print(f'Minimum training Loss: {np.min(self.train_epoch_loss):.4f}')
            print(f'Maximum training Loss: {np.max(self.train_epoch_loss):.4f}')

        if self.train_start_time is not None and self.train_end_time is not None:
            print(f'Total training time: {self.get_train_time():.4f} seconds')

        if self.val_loss is not None:
            print(f'Validation Loss: {self.val_loss:.4f}')

        if self.val_start_time is not None and self.val_end_time is not None: 
            print(f'Inference time: {self.get_val_time():.4f} seconds')

    def plot_train_loss(self):
        plt.plot(self.train_epoch_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
