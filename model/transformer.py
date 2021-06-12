from typing_extensions import Literal
from torch.nn import MSELoss, TransformerEncoderLayer, TransformerEncoder
from torch.optim import AdamW
import pytorch_lightning as pl

from dataloader.loader import loadTrainingData, loadTestData
from dataloader.utility import get_user_labels
from .loss import AngularLoss
from .positional_encoding import PositionalEncoding, LearnedPositionalEncoding, Time2VecPositionalEncoding
from .head import Head
from .image import FlattenImages


class GazeTransformer(pl.LightningModule):
    def __init__(self, feature_number, pos_kernel_size=8, batch_size=1, num_worker=0, model_type: Literal['saliency', 'flatten'] = 'flatten'):
        super().__init__()
        self.save_hyperparameters()
        self.pos_kernel_size = pos_kernel_size
        self.feature_number = feature_number + self.pos_kernel_size
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.model_type = model_type

        self.positional_encoding = Time2VecPositionalEncoding(feature_number, self.pos_kernel_size)
        encoder_layers = TransformerEncoderLayer(
            self.feature_number, nhead=8, dim_feedforward=self.feature_number)
        self.encoder = TransformerEncoder(encoder_layers, num_layers=6)
        self.decoder = Head(self.feature_number)
        self.loss = MSELoss()
        self.angular_loss = AngularLoss()

        if self.model_type == 'flatten':
            self.backbone = FlattenImages()

    def forward(self, src, images):
        src = src.transpose(-3, -2)
        if self.model_type != 'saliency':
            src = self.backbone(src, images)
        src = self.positional_encoding(src)
        memory = self.encoder(src).transpose(-2, -3)
        return self.decoder(memory)
        

    def training_step(self, batch, batch_idx):
        src, y, images = batch
        pred = self(src, images)
        loss = self.angular_loss(pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, y, images = batch
        pred = self(src, images)
        val_loss = self.angular_loss(pred, y)
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        src, y, images = batch
        pred = self(src, images)
        test_loss = self.angular_loss(pred, y)
        self.log('test_loss', test_loss)
        return test_loss

    def configure_optimizers(self):
        t_opt = AdamW(self.parameters())
        return t_opt

    def train_dataloader(self):
        return loadTrainingData(get_user_labels(1), self.batch_size, self.num_worker, self.get_loader_data_type())

    def val_dataloader(self):
        return loadTestData(get_user_labels(1), self.batch_size, self.num_worker, self.get_loader_data_type())

    def test_dataloader(self):
        return loadTestData(get_user_labels(1), self.batch_size, self.num_worker, self.get_loader_data_type())

    def get_loader_data_type(self):
        if self.model_type == 'saliency':
            return 'saliency'
        return 'video'