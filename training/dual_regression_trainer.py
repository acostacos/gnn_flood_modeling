from .base_trainer import BaseTrainer

class DualRegressionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.stats.start_train()
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            running_node_loss = 0.0
            running_edge_loss = 0.0

            len_training_samples = 0
            for dataloader in self.train_datasets:
                len_training_samples += len(dataloader.dataset)
                for batch in dataloader:
                    self.optimizer.zero_grad()

                    batch = batch.to(self.device)
                    node_pred, edge_pred = self.model(batch)

                    loss = self.loss_func(node_pred, edge_pred, batch)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    node_loss, edge_loss = self.loss_func.get_loss_components()
                    running_node_loss += node_loss
                    running_edge_loss += edge_loss

            epoch_loss = running_loss / len_training_samples
            epoch_node_loss = running_node_loss / len_training_samples
            epoch_edge_loss = running_edge_loss / len_training_samples

            self.stats.add_train_loss(epoch_loss)
            self.log(f'Epoch [{epoch + 1}/{self.num_epochs}]:')
            self.log(f'\tNode Loss: {epoch_node_loss:.4f}')
            self.log(f'\tEdge Loss: {epoch_edge_loss:.4f}')
            self.log(f'\tTotal Loss: {epoch_loss:.4f}')

            if self.debug:
                self.print_memory_usage(epoch)

        additional_info = {
            'Final Node Loss': epoch_node_loss,
            'Final Edge Loss': epoch_edge_loss,
        }
        self.stats.update_additional_info(additional_info)
        self.stats.end_train()
