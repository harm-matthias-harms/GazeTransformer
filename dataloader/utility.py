import os

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


def get_sequence_name(video_path):
    return video_path.split("bandicam ")[-1].replace(".avi", ".dill")


def get_filenames():
    result = []
    dirs = [VIDEO_DIR, GAZE_DIR, HEAD_DIR, TASK_DIR]
    for dir in dirs:
        filelist = pd.read_csv(os.path.join(
            RAW_DATASET_PATH, dir, 'filelist.txt'), header=None)[0]
        filenames = np.array(
            [os.path.join(RAW_DATASET_PATH, dir, f) for f in filelist])
        result.append(filenames)
    return np.array(result).transpose((1, 0))


def get_video_timstamps():
    return np.array(list(pd.read_csv(os.path.join(RAW_DATASET_PATH, VIDEO_DIR, 'videoBaseTimestamps.txt'), header=None)[0]))


def get_start_timestamps():
    timestamps = get_video_timstamps()
    dirs = [GAZE_DIR, HEAD_DIR, TASK_DIR]
    file_lists = [list(pd.read_csv(os.path.join(
        RAW_DATASET_PATH, dir, 'filelist.txt'), header=None)[0]) for dir in dirs]
    result = [0] * len(timestamps)
    for idx, timestamp in enumerate(timestamps):
        result[idx] = timestamp
        for l_idx, f_list in enumerate(file_lists):
            # l_idx == 2 is the TaskData
            time_name = "Timestamp" if l_idx < 2 else "Time"
            seperator = '\t' if l_idx < 2 else ','
            data_min_timestamp = pd.read_csv(os.path.join(
                RAW_DATASET_PATH, dirs[l_idx], f_list[idx]), sep=seperator, nrows=1, index_col=False)[time_name][0]
            if data_min_timestamp > result[idx]:
                result[idx] = data_min_timestamp
    return np.array(result)


def get_user_labels(test_user):
    labels = pd.read_csv(os.path.join(
        DATASET_PATH, LABEL_PATH, 'UserLabels.txt'), header=None)[0]
    return labels != test_user


def get_scene_labels(test_scene):
    labels = pd.read_csv(os.path.join(
        DATASET_PATH, LABEL_PATH, 'SceneLabels.txt'), header=None)[0]
    return labels != test_scene
