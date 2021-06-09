from torch.nn import  MSELoss
from torch.optim import AdamW
import pytorch_lightning as pl

from dataloader.loader import loadTrainingData, loadTestData
from dataloader.utility import get_user_labels
from .loss import AngularLoss
from .positional_encoding import PositionalEncoding, LearnedPositionalEncoding, Time2VecPositionalEncoding
from .transformers import Transformer1, Transformer2, Transformer3


class GazeTransformer(pl.LightningModule):
    def __init__(self, feature_number, pos_kernel_size=8, batch_size=1, num_worker=0):
        super().__init__()
        self.save_hyperparameters()
        self.pos_kernel_size = pos_kernel_size
        self.feature_number = feature_number + self.pos_kernel_size
        self.batch_size = batch_size
        self.num_worker = num_worker

        self.positional_encoding = Time2VecPositionalEncoding(feature_number, self.pos_kernel_size)
        self.transformer = Transformer2(self.feature_number)
        self.loss = MSELoss()
        self.angular_loss = AngularLoss()

    def forward(self, src, tgt):
        src = src.transpose(-3, -2)
        tgt = tgt.transpose(-3, -2)
        src = self.positional_encoding(src)
        return self.transformer(src, tgt)
        

    def training_step(self, batch, batch_idx):
        src = batch['sequence']
        tgt = batch['label']
        pred = self(src, tgt)
        loss = self.angular_loss(pred, tgt[:, :, :2])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src = batch['sequence']
        tgt = batch['label']
        pred = self(src, tgt)
        val_loss = self.angular_loss(pred, tgt[:, :, :2])
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        src = batch['sequence']
        tgt = batch['label']
        pred = self(src, tgt)
        test_loss = self.angular_loss(pred, tgt[:, :, :2])
        self.log('test_loss', test_loss)
        return test_loss

    def configure_optimizers(self):
        t_opt = AdamW(self.parameters())
        return t_opt

    def train_dataloader(self):
        return loadTrainingData(get_user_labels(1), self.batch_size, self.num_worker)

    def val_dataloader(self):
        return loadTestData(get_user_labels(1), self.batch_size, self.num_worker)

    def test_dataloader(self):
        return loadTestData(get_user_labels(1), self.batch_size, self.num_worker)
