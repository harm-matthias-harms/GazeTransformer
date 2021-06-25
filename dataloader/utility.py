import os
from typing_extensions import Literal

import pandas as pd
import numpy as np

RAW_DATASET_PATH = '../dataset/rawData'
DATASET_PATH = '../dataset/dataset'

FIXATION_DIR = 'FixationData'
GAZE_DIR = 'GazeData'
VIDEO_DIR = 'Videos'
HEAD_DIR = 'HeadData'
TASK_DIR = 'TaskData'
LABEL_PATH = 'Labels/FixationNetDataset'

SALIENCY_PATH = 'SaliencyVideos'
CROPPED_PATH = 'CroppedVideos'
FLATTEN_PATH = 'FlattenedVideos'
PATCH_PATH = 'PatchVideos'
RESNET_PATH = 'ResNetVideos'
DINO_PATH = 'DinoVideos'


def get_sequence_name(video_path):
    return video_path.split("bandicam ")[-1].replace(".avi", ".dill")


def get_filenames():
    result = []
    dirs = [VIDEO_DIR, GAZE_DIR, HEAD_DIR, TASK_DIR]
    for dir in dirs:
        filelist = pd.read_csv(os.path.join(
            os.path.dirname(__file__), RAW_DATASET_PATH, dir, 'filelist.txt'), header=None)[0]
        filenames = np.array(
            [os.path.join(os.path.dirname(__file__), RAW_DATASET_PATH, dir, f) for f in filelist])
        result.append(filenames)
    return np.array(result).transpose((1, 0))


def get_video_timstamps():
    return np.array(list(pd.read_csv(os.path.join(os.path.dirname(__file__), RAW_DATASET_PATH, VIDEO_DIR, 'videoBaseTimestamps.txt'), header=None)[0]))


def get_start_timestamps():
    timestamps = get_video_timstamps()
    dirs = [GAZE_DIR, HEAD_DIR, TASK_DIR]
    file_lists = [list(pd.read_csv(os.path.join(
        os.path.dirname(__file__), RAW_DATASET_PATH, dir, 'filelist.txt'), header=None)[0]) for dir in dirs]
    result = [0] * len(timestamps)
    for idx, timestamp in enumerate(timestamps):
        result[idx] = timestamp
        for l_idx, f_list in enumerate(file_lists):
            # l_idx == 2 is the TaskData
            time_name = "Timestamp" if l_idx < 2 else "Time"
            seperator = '\t' if l_idx < 2 else ','
            data_min_timestamp = pd.read_csv(os.path.join(
                os.path.dirname(__file__), RAW_DATASET_PATH, dirs[l_idx], f_list[idx]), sep=seperator, nrows=1, index_col=False)[time_name][0]
            if data_min_timestamp > result[idx]:
                result[idx] = data_min_timestamp
    return np.array(result)


def get_user_labels(test_user):
    labels = pd.read_csv(os.path.join(
        os.path.dirname(__file__), DATASET_PATH, LABEL_PATH, 'UserLabels.txt'), header=None)[0]
    return labels != test_user


def get_scene_labels(test_scene):
    labels = pd.read_csv(os.path.join(
        os.path.dirname(__file__), DATASET_PATH, LABEL_PATH, 'SceneLabels.txt'), header=None)[0]
    return labels != test_scene

def get_original_data_path(number, is_user=True):
    cross_eval_type = "User" if is_user else "Scene"
    return f"FixationNet_150_Cross{cross_eval_type}/FixationNet_150_{cross_eval_type}{number}/"


def get_scene_labels(test_scene):
    labels = pd.read_csv(os.path.join(
        os.path.dirname(__file__), DATASET_PATH, LABEL_PATH, 'SceneLabels.txt'), header=None)[0]
    return labels != test_scene


def get_video_path(video_path, mode: Literal['saliency', 'flatten', 'patches', 'resnet', 'dino', None]):
    path = CROPPED_PATH
    if mode == 'salience':
        path = SALIENCY_PATH
    elif mode == 'flatten':
        path = FLATTEN_PATH
    elif mode == 'patches':
        path = PATCH_PATH
    elif mode == 'resnet':
        path = RESNET_PATH
    elif mode == 'dino':
        path = DINO_PATH

    filename = video_path.split("bandicam ")[-1]
    if mode in ["resnet", "dino"]:
        filename = filename.replace('.avi', '.pt')

    return os.path.join(os.path.dirname(__file__), DATASET_PATH, path, filename)
