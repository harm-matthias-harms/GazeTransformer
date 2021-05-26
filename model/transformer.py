from torch.nn import Transformer
import pytorch_lightning as pl


class GazeTransformer(pl.LightningModule):
    def __init__(self):
        self.transformer = Transformer()
        pass

    def forward(self, src, tgt):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
