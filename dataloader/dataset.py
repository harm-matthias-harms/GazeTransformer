import dill
import torch
from torch.utils.data import Dataset

from .video import VideoParser


class VideoDataset(Dataset):
    def __init__(self, video_path, sequence_path, base_timestamp, start_timestamp):
        self.video = VideoParser(video_path, base_timestamp, start_timestamp)
        with open(sequence_path, 'rb') as f:
            self.data = dill.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        return {
            'sequence': self.feature_to_sequence(torch.tensor(data_point['sequence'])),
            'label': torch.tensor([data_point['label']]),
            'images': self.video.get_frames(data_point['video'])
        }

    def feature_to_sequence(self, feature):
        gazes = torch.reshape(feature[:80], (40, 2))
        head = torch.reshape(feature[80:160], (40, 2))
        task = torch.reshape(feature[160: 640], (40, 12))
        return torch.cat((gazes, head, task), 1)
