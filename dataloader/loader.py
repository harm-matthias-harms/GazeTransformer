import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from .utility import get_filenames, get_start_timestamps, get_video_timstamps, get_sequence_name
from .dataset import VideoDataset

def loadTrainingData(should_train, batch_size, num_workers, sequence_prefix='../dataset/dataset/FixationNet_150_Images/'):
    filenames = get_filenames()
    video_timestamps = get_video_timstamps()
    start_timestamps = get_start_timestamps()
    datasets = []
    for idx, files in enumerate(filenames):
        if should_train[idx]:
            dataset = VideoDataset(files[0], os.path.join(os.path.dirname(
                __file__), sequence_prefix) + get_sequence_name(files[0]), video_timestamps[idx], start_timestamps[idx])
            datasets.append(dataset)
            break

    concatenated_datasets = ConcatDataset(datasets)
    return DataLoader(dataset=concatenated_datasets, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)


def loadTestData(should_train, batch_size, num_workers, sequence_prefix='../dataset/dataset/FixationNet_150_Images/'):
    filenames = get_filenames()
    video_timestamps = get_video_timstamps()
    start_timestamps = get_start_timestamps()
    datasets = []
    for idx, files in enumerate(filenames):
        if not should_train[idx]:
            dataset = VideoDataset(files[0], os.path.join(os.path.dirname(
                __file__), sequence_prefix) + get_sequence_name(files[0]), video_timestamps[idx], start_timestamps[idx])
            datasets.append(dataset)
            break

    concatenated_datasets = ConcatDataset(datasets)
    return DataLoader(dataset=concatenated_datasets, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
