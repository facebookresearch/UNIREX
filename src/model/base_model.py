from typing import Optional

import pytorch_lightning as pl


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # update in `setup`
        self.total_steps = None

    def forward(self, **kwargs):
        raise NotImplementedError

    def calc_loss(self, preds, targets):
        raise NotImplementedError

    def calc_acc(self, preds, targets):
        raise NotImplementedError

    def run_step(self, batch, split):
        raise NotImplementedError

    def aggregate_epoch(self, outputs, split):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # # freeze encoder for initial few epochs based on p.freeze_epochs
        # if self.current_epoch < self.freeze_epochs:
        # 	freeze_net(self.text_encoder)
        # else:
        # 	unfreeze_net(self.text_encoder)

        return self.run_step(batch, 'train', batch_idx)

    def training_epoch_end(self, outputs):
        self.aggregate_epoch(outputs, 'train')

    def validation_step(self, batch, batch_idx, dataset_idx):
        assert dataset_idx in [0, 1]
        eval_splits = {0: 'dev', 1: 'test'}
        return self.run_step(batch, eval_splits[dataset_idx], batch_idx)

    def validation_epoch_end(self, outputs):
        self.aggregate_epoch(outputs, 'dev')

    def test_step(self, batch, batch_idx):
        return self.run_step(batch, 'test', batch_idx)

    def test_epoch_end(self, outputs):
        self.aggregate_epoch(outputs, 'test')

    def setup(self, stage: Optional[str] = None):
        """calculate total steps"""
        if stage == 'fit':
            # Get train dataloader
            train_loader = self.trainer.datamodule.train_dataloader()
            ngpus = self.trainer.num_gpus

            # Calculate total steps
            eff_train_batch_size = (self.trainer.datamodule.train_batch_size *
                                    max(1, ngpus) * self.trainer.accumulate_grad_batches)
            assert eff_train_batch_size == self.trainer.datamodule.eff_train_batch_size
            self.total_steps = int(
                (len(train_loader.dataset) // eff_train_batch_size) * float(self.trainer.max_epochs))

    def configure_optimizers(self):
        raise NotImplementedError
