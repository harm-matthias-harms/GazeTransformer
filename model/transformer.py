import os
from torch.nn import Transformer
from torch.optim import AdamW
import pytorch_lightning as pl

from dataloader.loader import loadTrainingData, loadTestData


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
        t_opt = AdamW(self.transformer.parameters())
        return t_opt

    def train_dataloader(self):
        return loadTrainingData(os.path.join(os.path.dirname(__file__), "../dataset/dataset/FixationNet_150_CrossScene/FixationNet_150_Scene1/"), 512, 10)

    def val_dataloader(self):
        return loadTestData(os.path.join(os.path.dirname(__file__), "../dataset/dataset/FixationNet_150_CrossScene/FixationNet_150_Scene1/"), 512, 10)

    def test_dataloader(self):
        return loadTestData(os.path.join(os.path.dirname(__file__), "../dataset/dataset/FixationNet_150_CrossScene/FixationNet_150_Scene1/"), 512, 10)
