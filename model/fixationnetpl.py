import os
from torch.optim import Adam
import pytorch_lightning as pl

from dataloader.utility import get_user_labels
from dataloader.loader import loadOriginalData, loadOriginalTestData, loadTrainingData, loadTestData
from .loss import AngularLoss
from .fixationnet.model import FixationNet


class FixationNetPL(pl.LightningModule):
    def __init__(self, batch_size=2, num_worker=0, with_original_data=False):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.with_original_data = with_original_data

        cluster_path = os.path.join(os.path.dirname(__file__), "../dataset/dataset/FixationNet_150_CrossUser/FixationNet_150_User1/clusterCenters.npy")

        self.model = FixationNet(80, 80, 480, 1152, cluster_path)
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
            return loadOriginalData(self.batch_size, self.num_worker)
        return loadTrainingData(get_user_labels(1), self.batch_size, self.num_worker, as_row=True)

    def val_dataloader(self):
        if self.with_original_data:
            return loadOriginalTestData(self.batch_size, self.num_worker)
        return loadTestData(get_user_labels(1), self.batch_size, self.num_worker, as_row=True)

    def test_dataloader(self):
        if self.with_original_data:
            return loadOriginalTestData(self.batch_size, self.num_worker)
        return loadTestData(get_user_labels(1), self.batch_size, self.num_worker, as_row=True)
