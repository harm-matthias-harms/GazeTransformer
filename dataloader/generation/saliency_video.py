import os
import torch
import cv2
import numpy as np
import decord
import progressbar
import sys
import torchvision.transforms.functional as TF

sys.path.append("../model")
from saliency import SaliencyMap
from utility import get_filenames


def generate(path_prefix=""):
    print("Generate saliency videos")

    GENERATE_PATH = os.path.join(path_prefix, '../dataset/dataset/SaliencyVideos/')
    batch_size = 128

    if not os.path.exists(GENERATE_PATH):
        os.makedirs(GENERATE_PATH)

    decord.bridge.set_bridge("torch")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filenames = get_filenames()
    model = SaliencyMap((256, 256)).to(device)
    for idx, files in enumerate(filenames):
        print(idx)

        video = decord.VideoReader(files[0], ctx=decord.gpu())
        video_length = len(video)
        filename = GENERATE_PATH + files[0].split("bandicam ")[-1]
        out = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*'XVID'), 60, (24, 24), 0)

        batches = [list(range(video_length))[i * batch_size:(i + 1) * batch_size]
                   for i in range((video_length + batch_size - 1) // batch_size)]
        for batch in progressbar.progressbar(batches):
            images = video.get_batch(batch).float()
            images = images.permute(0, 3, 1, 2)
            images = TF.center_crop(images, (256, 256))
            images /= 255.0
            pred = model(images)
            pred = pred.permute(0, 2, 3, 1)
            pred = pred.cpu().numpy()
            for image in pred:
                out.write((image*255).astype(np.uint8))

        out.release()


if __name__ == '__main__':
    generate(path_prefix="../")
