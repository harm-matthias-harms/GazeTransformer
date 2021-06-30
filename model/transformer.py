from typing_extensions import Literal
from torch.nn import MSELoss, TransformerEncoderLayer, TransformerEncoder
from torch.optim import AdamW
import pytorch_lightning as pl

from dataloader.loader import loadTrainingData, loadTestData, loadOriginalData, loadOriginalTestData
from dataloader.utility import get_user_labels, get_original_data_path
from .loss import AngularLoss
from .positional_encoding import Time2VecPositionalEncoding
from .head import Head


class GazeTransformer(pl.LightningModule):
    def __init__(self, pos_kernel_size=8, batch_size=1, num_worker=0, model_type: Literal['original', 'original-no-images', 'no-images', 'saliency', 'flatten', 'patches', 'resnet', 'dino'] = 'original', loss: Literal['angular', 'mse']= 'angular'):
        super().__init__()
        self.save_hyperparameters()
        self.pos_kernel_size = pos_kernel_size
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.model_type = model_type
        self.set_feature_number()

        self.positional_encoding = Time2VecPositionalEncoding(
            self.feature_number - self.pos_kernel_size, self.pos_kernel_size)
        encoder_layers = TransformerEncoderLayer(
            self.feature_number, nhead=8, dim_feedforward=self.feature_number)
        self.encoder = TransformerEncoder(encoder_layers, num_layers=6)
        self.decoder = Head(self.feature_number)
        self.angular_loss = AngularLoss()
        if loss == 'angular':
            self.loss = self.angular_loss
        elif loss == 'mse':
            self.loss = MSELoss()

    def forward(self, src):
        src = src.transpose(-3, -2)
        src = self.positional_encoding(src)
        memory = self.encoder(src).transpose(-2, -3)
        return self.decoder(memory)

    def training_step(self, batch, batch_idx):
        src, y = batch
        pred = self(src)
        loss = self.loss(pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, y = batch
        pred = self(src)
        val_loss = self.loss(pred, y)
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        src, y = batch
        pred = self(src)
        test_loss = self.angular_loss(pred, y)
        self.log('test_loss', test_loss)
        return test_loss

    def configure_optimizers(self):
        t_opt = AdamW(self.parameters())
        return t_opt

    def train_dataloader(self):
        if self.model_type in ['original-no-images', 'original']:
            return loadOriginalData(get_original_data_path(1), self.batch_size, self.num_worker, True, self.model_type == 'original-no-images')
        return loadTrainingData(get_user_labels(1), self.batch_size, self.num_worker, self.model_type)

    def val_dataloader(self):
        if self.model_type in ['original-no-images', 'original']:
            return loadOriginalTestData(get_original_data_path(1), self.batch_size, self.num_worker, True, self.model_type == 'original-no-images')
        return loadTestData(get_user_labels(1), self.batch_size, self.num_worker, self.model_type)

    def test_dataloader(self):
        if self.model_type in ['original-no-images', 'original']:
            return loadOriginalTestData(get_original_data_path(1), self.batch_size, self.num_worker, True, self.model_type == 'original-no-images')
        return loadTestData(get_user_labels(1), self.batch_size, self.num_worker, self.model_type)

    def set_feature_number(self):
        if self.model_type in ['original-no-images', 'no-images']:
            self.feature_number = 16
        elif self.model_type in ['original', 'saliency']:
            self.feature_number = 592
        elif self.model_type == 'flatten':
            self.feature_number = 1040
        elif self.model_type == 'patches':
            self.feature_number = 784
        elif self.model_type == 'resnet':
            self.feature_number = 2064
        elif self.model_type == 'dino':
            self.feature_number = 400

        self.feature_number += self.pos_kernel_size
