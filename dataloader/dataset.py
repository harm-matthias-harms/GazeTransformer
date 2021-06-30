import dill
import torch
from torch.utils.data import Dataset

from .video import InMemoryVideoParser, PTVideoParser


class TimeSequenceVideoDataset(Dataset):
    def __init__(self, video_path, sequence_path, base_timestamp, start_timestamp, grayscale=False, ignore_images=False, is_pt=False):
        self.ignore_images = ignore_images

        if not self.ignore_images:
            if is_pt:
                self.video = PTVideoParser(
                    video_path, base_timestamp, start_timestamp)
            else:
                self.video = InMemoryVideoParser(
                    video_path, base_timestamp, start_timestamp, grayscale)
        with open(sequence_path, 'rb') as f:
            self.data = dill.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        if self.ignore_images:
            images = torch.Tensor()
        else:
            images = self.video.get_frames(data_point['video'])
        sequence = self.feature_to_sequence(
            torch.FloatTensor(data_point['sequence']), images)
        label = torch.FloatTensor([data_point['label']])
        return sequence, label

    def feature_to_sequence(self, feature, images):
        gazes = torch.reshape(feature[:80], (40, 2))
        head = torch.reshape(feature[80:160], (40, 2))
        task = torch.reshape(feature[160: 640], (40, 12))
        if not self.ignore_images:
            images = images.flatten(1)
            # time descending, last image -> first image
            images = torch.cat(
                (images[-1].repeat(20, 1), images[0].repeat(20, 1)))
            return torch.cat((gazes, head, task, images), 1)
        return torch.cat((gazes, head, task), 1)


class FixationnetVideoDataset(Dataset):
    def __init__(self, video_path, sequence_path, base_timestamp, start_timestamp, grayscale=False, ignore_images=False, is_pt=False):
        self.video = InMemoryVideoParser(
            video_path, base_timestamp, start_timestamp, grayscale)
        with open(sequence_path, 'rb') as f:
            self.data = dill.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        images = self.video.get_frames(data_point['video'])
        sequence = self.feature_to_sequence(
            torch.FloatTensor(data_point['sequence']), images)
        label = torch.FloatTensor([data_point['label']])
        return sequence, label

    def feature_to_sequence(self, feature, images):
        images = images.flatten(1)
        # time descending, last image -> first image
        images = torch.cat((images[-1], images[0]))
        return torch.cat((feature, images))


class FixationnetDataset(Dataset):
    def __init__(self, features, labels, as_sequence=False, ignore_images=False):
        self.features = features
        self.labels = labels
        self.as_sequence = as_sequence
        self.ignore_images = ignore_images

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx]

        if self.as_sequence:
            features = self.features_to_sequence(features)
        return features, labels

    def features_to_sequence(self, features):
        gazes = torch.reshape(features[:80], (40, 2))
        head = torch.reshape(features[80:160], (40, 2))
        task = torch.reshape(features[160: 640], (40, 12))
        if not self.ignore_images:
            image_0 = features[640: 1216]
            image_1 = features[1216:]
            # time descending, last image -> first image
            images = torch.cat((image_1.repeat(20, 1), image_0.repeat(20, 1)))
            return torch.cat((gazes, head, task, images), 1)
        return torch.cat((gazes, head, task), 1)
