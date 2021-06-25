import os
from typing_extensions import Literal
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from .utility import get_filenames, get_start_timestamps, get_video_timstamps, get_sequence_name, get_video_path
from .dataset import TimeSequenceVideoDataset, FixationnetVideoDataset, FixationnetDataset


def loadTrainingData(should_train, batch_size, num_workers, mode: Literal['no-images', 'saliency', 'flatten', 'patches', 'resnet', 'dino'] = 'no-images', sequence_prefix='../dataset/dataset/FixationNet_150_Images/GazeLabel/', fixationnet=False):
    datasets = []
    filenames = get_filenames()
    video_timestamps = get_video_timstamps()
    start_timestamps = get_start_timestamps()

    dataset_type = FixationnetVideoDataset if fixationnet else TimeSequenceVideoDataset

    for idx, files in enumerate(filenames):
        if should_train[idx]:
            video_path = get_video_path(files[0], mode)

            dataset = dataset_type(
                video_path,
                os.path.join(os.path.dirname(__file__),
                             sequence_prefix) + get_sequence_name(files[0]),
                video_timestamps[idx],
                start_timestamps[idx],
                grayscale=mode in ['saliency', 'flatten'],
                ignore_images=mode == 'no-images',
                is_pt=mode in ['resnet', 'dino']
            )
            datasets.append(dataset)

    concatenated_datasets = ConcatDataset(datasets)
    return DataLoader(dataset=concatenated_datasets, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)


def loadTestData(should_train, batch_size, num_workers, mode: Literal['no-images', 'saliency', 'flatten', 'patches', 'resnet', 'dino'] = 'no-images', sequence_prefix='../dataset/dataset/FixationNet_150_Images/GazeLabel/', fixationnet=False):
    datasets = []
    filenames = get_filenames()
    video_timestamps = get_video_timstamps()
    start_timestamps = get_start_timestamps()

    dataset_type = FixationnetVideoDataset if fixationnet else TimeSequenceVideoDataset

    for idx, files in enumerate(filenames):
        if not should_train[idx]:
            video_path = get_video_path(files[0], mode)

            dataset = dataset_type(
                video_path,
                os.path.join(os.path.dirname(__file__),
                             sequence_prefix) + get_sequence_name(files[0]),
                video_timestamps[idx],
                start_timestamps[idx],
                grayscale=mode in ['saliency', 'flatten'],
                ignore_images=mode == 'no-images',
                is_pt=mode in ['resnet', 'dino']
            )
            datasets.append(dataset)

    concatenated_datasets = ConcatDataset(datasets)
    return DataLoader(dataset=concatenated_datasets, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)


def loadOriginalData(path, batch_size, num_workers, as_sequence=False, ignore_images=False):
    dataset_path = os.path.join(os.path.dirname(__file__), "../dataset/dataset/", path)
    trainingX = torch.from_numpy(np.load(dataset_path + 'trainingX.npy')).float()
    trainingY = torch.from_numpy(np.load(dataset_path + 'trainingY.npy')).float()

    dataset = FixationnetDataset(trainingX, trainingY, as_sequence, ignore_images)
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)


def loadOriginalTestData(path, batch_size, num_workers, as_sequence=False, ignore_images=False):
    dataset_path = os.path.join(os.path.dirname(__file__), "../dataset/dataset/", path)#FixationNet_150_CrossUser/FixationNet_150_User1/")
    testX = torch.from_numpy(np.load(dataset_path + 'testX.npy')).float()
    testY = torch.from_numpy(np.load(dataset_path + 'testY.npy')).float()

    dataset = FixationnetDataset(testX, testY, as_sequence, ignore_images)
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
