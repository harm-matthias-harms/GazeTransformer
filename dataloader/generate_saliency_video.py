# import requi9red module

import torch
import cv2
import numpy as np
import decord
import progressbar
import sys

# append the path of the
# parent directory
sys.path.append("../model")

from saliency import SaliencyMap
from utility import get_filenames


GENERATE_PATH = '../dataset/dataset/SaliencyVideos/'

decord.bridge.set_bridge("torch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    filenames = get_filenames()
    model = SaliencyMap((600, 540)).to(device)
    for idx, files in enumerate(filenames):
        print(idx)
        result = []

        video = decord.VideoReader(files[0], ctx=decord.cpu())
        video_length = len(video)
        filename = GENERATE_PATH + files[0].split("bandicam ")[-1]
        out = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*'mp4v'), 60, (24, 24), 0)
        n = 32
        batches = [list(range(video_length))[i * n:(i + 1) * n]
                   for i in range((video_length + n - 1) // n)]
        for batch in progressbar.progressbar(batches):
            images = video.get_batch(batch).to(device).float()
            images /= 255.0
            images = images.permute(0, 3, 1, 2)
            pred = model(images)
            pred = pred.permute(0, 2, 3, 1)
            pred = pred.cpu().numpy()
            for image in pred:
                out.write((image*255).astype(np.uint8))

        out.release()
