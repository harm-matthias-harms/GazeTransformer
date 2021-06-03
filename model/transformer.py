import torch
from torch.nn import Transformer
from torch.optim import AdamW
import pytorch_lightning as pl

from dataloader.loader import loadTrainingData, loadTestData
from dataloader.utility import get_user_labels
from .loss import AngularLoss
from .positional_encoding import PositionalEncoding


class GazeTransformer(pl.LightningModule):
    def __init__(self, feature_number, batch_size=1, num_worker=0):
        super().__init__()
        self.feature_number = feature_number
        self.batch_size = batch_size
        self.num_worker = num_worker

        self.positional_encoding = PositionalEncoding(self.feature_number)
        self.transformer = Transformer(d_model = self.feature_number)
        self.loss = AngularLoss()

    def forward(self, src, tgt):
        src = src.transpose(-3, -2)
        tgt = tgt.transpose(-3, -2)
        src = self.positional_encoding(src)
        pred = self.transformer(src, tgt)
        return pred.transpose(-2, -3)

    def training_step(self, batch, batch_idx):
        src = batch['sequence']
        tgt = batch['label']
        pred = self(src, tgt)
        loss = self.loss(pred[:, :, :2], tgt[:, :, :2])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src = batch['sequence']
        tgt = batch['label']
        pred = self(src, tgt)
        val_loss = self.loss(pred[:, :, :2], tgt[:, :, :2])
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        src = batch['sequence']
        tgt = batch['label']
        pred = self(src, tgt)
        test_loss = self.loss(pred[:, :, :2], tgt[:, :, :2])
        self.log('test_loss', test_loss)
        return test_loss

    def configure_optimizers(self):
        t_opt = AdamW(self.transformer.parameters())
        return t_opt

    def train_dataloader(self):
        return loadTrainingData(get_user_labels(1), self.batch_size, self.num_worker)

    def val_dataloader(self):
        return loadTestData(get_user_labels(1), self.batch_size, self.num_worker)

    def test_dataloader(self):
        return loadTestData(get_user_labels(1), self.batch_size, self.num_worker)
