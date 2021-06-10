import dill
import torch
from torch.utils.data import Dataset

from .video import VideoParser, SaliencyVideoParser


class TimeSequenceVideoDataset(Dataset):
    def __init__(self, video_path, sequence_path, base_timestamp, start_timestamp, is_saliency=False):
        if is_saliency:
            self.video = SaliencyVideoParser(
                video_path, base_timestamp, start_timestamp)
        else:
            self.video = VideoParser(
                video_path, base_timestamp, start_timestamp)
        with open(sequence_path, 'rb') as f:
            self.data = dill.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        images = self.video.get_frames(data_point['video'])
        sequence = self.feature_to_sequence(
            torch.FloatTensor(data_point['sequence']), images[:-1])
        label = torch.FloatTensor([data_point['label'][:2]])
        return sequence, label

    def feature_to_sequence(self, feature, images):
        images = images.flatten(1)
        # time descending, last image -> first image
        images = torch.cat((images[-1].repeat(20, 1), images[0].repeat(20, 1)))

        gazes = torch.reshape(feature[:80], (40, 2))
        head = torch.reshape(feature[80:160], (40, 2))
        task = torch.reshape(feature[160: 640], (40, 12))
        return torch.cat((gazes, head, task, images), 1)

class VideoDataset(Dataset):
    def __init__(self, video_path, sequence_path, base_timestamp, start_timestamp, is_saliency=False):
        if is_saliency:
            self.video = SaliencyVideoParser(
                video_path, base_timestamp, start_timestamp)
        else:
            self.video = VideoParser(
                video_path, base_timestamp, start_timestamp)
        with open(sequence_path, 'rb') as f:
            self.data = dill.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        images = self.video.get_frames(data_point['video'])
        sequence = self.feature_to_sequence(
            torch.FloatTensor(data_point['sequence']), images[:-1])
        label = torch.FloatTensor([data_point['label'][:2]])
        return sequence, label

    def feature_to_sequence(self, feature, images):
        images = images.flatten(1)
        # time descending, last image -> first image
        images = torch.cat((images[-1], images[0]))
        return torch.cat((feature, images))

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
