from typing_extensions import Literal
import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl
import progressbar

from dataloader.loader import loadTrainingData, loadTestData, loadOriginalData, loadOriginalTestData
from dataloader.utility import get_user_labels, get_scene_labels, get_original_data_path
from .loss import AngularLoss
from .positional_encoding import Time2VecPositionalEncoding
from .head import Head


class GazeTransformer(pl.LightningModule):
    def __init__(self, predict_delta=False, image_to_features=True, pos_kernel_size=8, nhead=8, num_layers=6, backbone_features=128, inner_head_features=128, learning_rate=0.001, batch_size=1, num_worker=0, model_type: Literal['original', 'original-no-images', 'no-images', 'saliency', 'flatten', 'patches', 'resnet', 'dino'] = 'original', loss: Literal['angular', 'mse'] = 'angular', cross_eval_type: Literal['user', 'scene'] = 'user', cross_eval_exclude=1, use_all_images=False):
        super().__init__()
        self.save_hyperparameters()
        self.pos_kernel_size = pos_kernel_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.model_type = model_type
        self.cross_eval_type = cross_eval_type
        self.cross_eval_exclude = cross_eval_exclude
        self.use_all_images = use_all_images
        self.backbone_features = backbone_features
        self.image_to_features = image_to_features
        self.set_feature_and_backbone_number()
        self.predict_delta = predict_delta

        self.backbone = nn.Sequential(
            nn.Linear(self.backbone_number, self.backbone_features), nn.ReLU(), nn.Dropout(0.1))
        self.positional_encoding = Time2VecPositionalEncoding(
            self.feature_number - self.pos_kernel_size, self.pos_kernel_size)
        encoder_layers = nn.TransformerEncoderLayer(
            self.feature_number, nhead=nhead, dim_feedforward=self.feature_number)
        self.encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers)
        self.decoder = Head(self.feature_number, inner_head_features)
        self.angular_loss = AngularLoss()
        if loss == 'angular':
            self.loss = self.angular_loss
        elif loss == 'mse':
            self.loss = nn.MSELoss()

    def forward(self, src):
        currentGaze = src[:, -1, :2]
        if self.image_to_features:
            src = torch.cat((src[:, :, :16], self.backbone(src[:, :, 16:])), 2)
        src = src.transpose(-3, -2)
        src = self.positional_encoding(src)
        memory = self.encoder(src).transpose(-2, -3)
        out = self.decoder(memory)
        if self.predict_delta:
            out = currentGaze + out
        return out

    def training_step(self, batch, batch_idx):
        src, y = batch
        pred = self(src)
        loss = self.loss(pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, y = batch
        pred = self(src)
        val_loss = self.angular_loss(pred, y)
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        src, y = batch
        pred = self(src)
        return pred, y

    def test_epoch_end(self, output_results):
        preds = torch.Tensor([]).to('cuda')
        ys = torch.Tensor([]).to('cuda')
        for output in output_results:
            preds = torch.cat((preds, output[0]), dim=0)
            ys = torch.cat((ys, output[1]), dim=0)
        all_loss = torch.zeros((ys.shape[0], 1)).to('cuda')
        for i in progressbar.progressbar(range(ys.shape[0])):
            all_loss[i] = self.angular_loss(preds[i], ys[i])
        test_loss = all_loss.mean()
        std = all_loss.std()
        self.log("test_loss", test_loss)
        self.log("std", std)

    def configure_optimizers(self):
        t_opt = AdamW(self.parameters(), lr=self.learning_rate)
        return t_opt

    def train_dataloader(self):
        if self.model_type in ['original-no-images', 'original']:
            return loadOriginalData(get_original_data_path(self.cross_eval_exclude, is_user=self.cross_eval_type == 'user'), self.batch_size, self.num_worker, True, self.model_type == 'original-no-images')
        return loadTrainingData(self.cross_eval_labels(), self.batch_size, self.num_worker, self.model_type, use_all_images=self.use_all_images)

    def val_dataloader(self):
        if self.model_type in ['original-no-images', 'original']:
            return loadOriginalTestData(get_original_data_path(self.cross_eval_exclude, is_user=self.cross_eval_type == 'user'), self.batch_size, self.num_worker, True, self.model_type == 'original-no-images')
        return loadTestData(self.cross_eval_labels(), self.batch_size, self.num_worker, self.model_type, use_all_images=self.use_all_images)

    def test_dataloader(self):
        if self.model_type in ['original-no-images', 'original']:
            return loadOriginalTestData(get_original_data_path(self.cross_eval_exclude, is_user=self.cross_eval_type == 'user'), self.batch_size, self.num_worker, True, self.model_type == 'original-no-images')
        return loadTestData(self.cross_eval_labels(), self.batch_size, self.num_worker, self.model_type, use_all_images=self.use_all_images)

    def cross_eval_labels(self):
        if self.cross_eval_type == 'user':
            return get_user_labels(self.cross_eval_exclude)
        return get_scene_labels(self.cross_eval_exclude)

    def set_feature_and_backbone_number(self):
        self.backbone_number = 1

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

        if self.image_to_features:
            self.backbone_number = self.feature_number - 16
            self.feature_number = 16 + self.backbone_features

        self.feature_number += self.pos_kernel_size
