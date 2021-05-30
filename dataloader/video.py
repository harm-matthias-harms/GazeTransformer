import decord
from torchvision.transforms.functional import resize

decord.bridge.set_bridge("torch")


class VideoParser():
    def __init__(self, path, video_timestamp, start_timestamp):
        self.begin_timestamp = video_timestamp
        self.vr = decord.VideoReader(path, ctx=decord.cpu())

        self.actual_start_index = round(
            (start_timestamp - video_timestamp) / (1000 / 60.0))

    def __len__(self):
        # last three frames are not working
        # -9 because we need data for target from 150ms in the future
        return len(self.vr) - 24 - self.actual_start_index - 3 - 9

    def get_timestamp(self, idx):
        idx += 24 + self.actual_start_index
        return self.begin_timestamp + round(idx * (1000 / 60.0))

    def get_frames(self, idx):
        idx += 24 + self.actual_start_index
        return resize(self.vr.get_batch([idx - 24, idx - 12, idx + 9]).permute(0, 3, 1, 2), 224)
