import decord
import torch
from torchvision.io import read_video
import torchvision.transforms.functional as TF

decord.bridge.set_bridge("torch")


class VideoParser():
    def __init__(self, path, video_timestamp, start_timestamp):
        self.begin_timestamp = video_timestamp
        self.vr = decord.VideoReader(path, ctx=decord.cpu())

        self.actual_start_index = round(
            (start_timestamp - video_timestamp) / (1000 / 60.0))

    def __len__(self):
        # last three frames are not working
        # -9 because we need data for target from 150ms in the future
        return len(self.vr) - 24 - self.actual_start_index - 3

    def get_timestamp(self, idx):
        idx += 24 + self.actual_start_index
        return self.begin_timestamp + round(idx * (1000 / 60.0))

    def get_frames(self, idx):
        idx += 24 + self.actual_start_index
        batch = self.vr.get_batch([idx - 24, idx - 12])
        return batch.permute(0, 3, 1, 2).float() / 255.0

class InMemoryVideoParser():
    def __init__(self, path, video_timestamp, start_timestamp, grayscale):
        self.begin_timestamp = video_timestamp
        self.data = read_video(path)[0]
        if grayscale:
            self.data = self.data[:, :, :, :1] # only keep one channel because of greyscale video
        self.data = self.data.permute(0, 3, 1, 2).float() / 255.0

        self.actual_start_index = round(
            (start_timestamp - video_timestamp) / (1000 / 60.0))

    def __len__(self):
        # last three frames are not working
        # -9 because we need data for target from 150ms in the future
        return len(self.data) - 24 - self.actual_start_index - 3

    def get_timestamp(self, idx):
        idx += 24 + self.actual_start_index
        return self.begin_timestamp + round(idx * (1000 / 60.0))

    def get_frames(self, idx):
        idx += 24 + self.actual_start_index

        indices = torch.tensor([idx - 24, idx - 12])
        return self.data.index_select(0, indices)
