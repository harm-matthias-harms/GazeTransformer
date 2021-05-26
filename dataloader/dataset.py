import dill
from torch.utils.data import Dataset

from .video import VideoParser


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'sequence': self.features[idx],
            'label': self.labels[idx]
        }


class VideoDataset(Dataset):
    def __init__(self, video_path, sequence_path, base_timestamp, start_timestamp):
        self.video = VideoParser(video_path, base_timestamp, start_timestamp)
        with open(sequence_path, 'rb') as f:
            self.data = dill.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_point['images'] = self.video.get_frames(idx)
        return data_point
