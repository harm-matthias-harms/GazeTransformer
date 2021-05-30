import progressbar
import dill
import numpy as np

from utility import get_filenames, get_start_timestamps, get_video_timstamps, get_sequence_name
from video import VideoParser
from datareader import DataReader

GENERATE_PATH = '../dataset/dataset/FixationNet_150_Images/'

if __name__ == '__main__':
    filenames = get_filenames()
    video_timestamps = get_video_timstamps()
    max_timestamps = get_start_timestamps()
    for idx, files in enumerate(filenames):
        print(idx)
        result = []

        video = VideoParser(
            files[0], video_timestamps[idx], max_timestamps[idx])
        gazeReader = DataReader(files[1], "\t", max_timestamps[idx])
        headReader = DataReader(files[2], "\t", max_timestamps[idx])
        taskReader = DataReader(files[3], ",", max_timestamps[idx])
        for i in progressbar.progressbar(range(len(video))):
            timestamp = video.get_timestamp(i)
            # video_data = video.get_frames(i)
            try:
                gaze_data = gazeReader.get_data_for_timestamp(timestamp)
                head_data = headReader.get_data_for_timestamp(timestamp)
                task_data = taskReader.get_data_for_timestamp(timestamp)

                gaze_label = gazeReader.get_label(timestamp)
                head_label = headReader.get_label(timestamp)
                task_label = taskReader.get_label(timestamp)

                task_data = task_data.reshape((40, 5, 4))[:, :3, :].flatten()
                task_label = task_label[:12]

                sample = {
                    'label': np.concatenate((gaze_label, head_label, task_label)),
                    'sequence': np.concatenate((gaze_data, head_data, task_data)),
                    'video': i
                }

                if sample['label'].shape != (16,):
                    raise "label not right shape"
                if sample['sequence'].shape != (640, ):
                    raise "sequence not right shape"

                result.append(sample)
            except:
                pass
        filename = GENERATE_PATH + get_sequence_name(files[0])
        with open(filename, "wb") as f:
            dill.dump(result, f)
