from .base_trainer import BaseTrainer

class NodeRegressionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.stats.start_train()
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            num_batches = 0
            for dataloader in self.train_datasets:
                for batch in dataloader:
                    self.optimizer.zero_grad()

                    batch = batch.to(self.device)
                    pred = self.model(batch)

                    label = batch.y
                    loss = self.loss_func(pred, label)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    num_batches += 1

            epoch_loss = running_loss / num_batches
            self.stats.add_train_loss(epoch_loss)
            self.log(f'Epoch [{epoch + 1}/{self.num_epochs}], Training Loss: {epoch_loss:.4f}')

            if self.debug:
                self.print_memory_usage(epoch)

        self.stats.end_train()
