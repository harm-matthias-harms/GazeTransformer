from typing_extensions import Literal
import os
from torch.optim import Adam
import pytorch_lightning as pl

from dataloader.utility import get_user_labels, get_scene_labels, get_original_data_path
from dataloader.loader import loadOriginalData, loadOriginalTestData, loadTrainingData, loadTestData
from .loss import AngularLoss
from .fixationnet.model import FixationNet


class FixationNetPL(pl.LightningModule):
    def __init__(self, batch_size=2, num_worker=0, with_original_data=False, predict_delta=True, cross_eval_type: Literal['user', 'scene'] = 'user', cross_eval_exclude=1):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.with_original_data = with_original_data
        self.cross_eval_type = cross_eval_type
        self.cross_eval_exclude = cross_eval_exclude
        self.model_type = 'saliency'

        cluster_path = os.path.join(os.path.dirname(__file__),
                                    "../dataset/dataset",
                                    get_original_data_path(self.cross_eval_exclude,
                                                           is_user=self.cross_eval_type == 'user'),
                                    "clusterCenters.npy")

        self.model = FixationNet(80, 80, 480, 1152, cluster_path, predict_delta=predict_delta)
        self.angular_loss = AngularLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.angular_loss(pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        val_loss = self.angular_loss(pred, y)
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        test_loss = self.angular_loss(pred, y)
        self.log('test_loss', test_loss)
        return test_loss

    def configure_optimizers(self):
        t_opt = Adam(self.parameters(), lr=1e-2, weight_decay=5e-5)
        return t_opt

    def train_dataloader(self):
        if self.with_original_data:
            return loadOriginalData(get_original_data_path(self.cross_eval_exclude, is_user=self.cross_eval_type == 'user'), self.batch_size, self.num_worker)
        return loadTrainingData(self.cross_eval_labels(), self.batch_size, self.num_worker, mode='saliency', fixationnet=True)

    def val_dataloader(self):
        if self.with_original_data:
            return loadOriginalTestData(get_original_data_path(self.cross_eval_exclude, is_user=self.cross_eval_type == 'user'), self.batch_size, self.num_worker)
        return loadTestData(self.cross_eval_labels(), self.batch_size, self.num_worker, mode='saliency', fixationnet=True)

    def test_dataloader(self):
        if self.with_original_data:
            return loadOriginalTestData(get_original_data_path(self.cross_eval_exclude, is_user=self.cross_eval_type == 'user'), self.batch_size, self.num_worker)
        return loadTestData(self.cross_eval_labels(), self.batch_size, self.num_worker, mode='saliency', fixationnet=True)

    def cross_eval_labels(self):
        if self.cross_eval_type == 'user':
            return get_user_labels(self.cross_eval_exclude)
        return get_scene_labels(self.cross_eval_exclude)
