import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from utility import get_filenames, get_start_timestamps, get_video_timstamps, get_sequence_name
from dataset import FeatureDataset, VideoDataset


def loadTrainingData(datasetDir, batch_size):
    trainingX = torch.from_numpy(np.load(datasetDir + 'trainingX.npy')).float()
    trainingY = torch.from_numpy(np.load(datasetDir + 'trainingY.npy')).float()

    dataset = FeatureDataset(trainingX, trainingY)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def loadTestData(datasetDir, batch_size):
    testX = torch.from_numpy(np.load(datasetDir + 'testX.npy')).float()
    testY = torch.from_numpy(np.load(datasetDir + 'testY.npy')).float()

    dataset = FeatureDataset(testX, testY)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True)


def loadImageTrainingData(batch_size, should_train, sequence_prefix='../dataset/dataset/FixationNet_150_Images/'):
    filenames = get_filenames()
    video_timestamps = get_video_timstamps()
    start_timestamps = get_start_timestamps()
    datasets = []
    for idx, files in enumerate(filenames):
        if should_train[idx]:
            dataset = VideoDataset(files[0], sequence_prefix + get_sequence_name(
                files[0]), video_timestamps[idx], start_timestamps[idx])
            datasets.append(dataset)

    concatenated_datasets = ConcatDataset(datasets)
    return DataLoader(dataset=concatenated_datasets, batch_size=batch_size, shuffle=True, drop_last=True)


def loadImageTestData(batch_size, should_train, sequence_prefix='../dataset/dataset/FixationNet_150_Images/'):
    filenames = get_filenames()
    video_timestamps = get_video_timstamps()
    start_timestamps = get_start_timestamps()
    datasets = []
    for idx, files in enumerate(filenames):
        if not should_train[idx]:
            dataset = VideoDataset(files[0], sequence_prefix + get_sequence_name(
                files[0]), video_timestamps[idx], start_timestamps[idx])
            datasets.append(dataset)

    concatenated_datasets = ConcatDataset(datasets)
    return DataLoader(dataset=concatenated_datasets, batch_size=batch_size, shuffle=False, drop_last=True)
