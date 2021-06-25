import os
import cv2
import numpy as np
import decord
import progressbar
import torchvision.transforms.functional as TF

from utility import get_filenames


def generate(path_prefix=""):
    print("Generate center-cropped videos")

    GENERATE_PATH = os.path.join(path_prefix, '../dataset/dataset/CroppedVideos/')
    batch_size = 1024

    if not os.path.exists(GENERATE_PATH):
        os.makedirs(GENERATE_PATH)

    decord.bridge.set_bridge("torch")

    filenames = get_filenames()
    for idx, files in enumerate(filenames):
        print(idx)

        video = decord.VideoReader(
            files[0], ctx=decord.gpu(), width=256, height=285)
        video_length = len(video)
        filename = GENERATE_PATH + files[0].split("bandicam ")[-1]
        out = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*'XVID'), 60, (128, 128))

        batches = [list(range(video_length))[i * batch_size:(i + 1) * batch_size]
                   for i in range((video_length + batch_size - 1) // batch_size)]
        for batch in progressbar.progressbar(batches):
            images = video.get_batch(batch).float()
            images = images.permute(0, 3, 1, 2)
            images = TF.center_crop(images, (128, 128))
            images = images.permute(0, 2, 3, 1)
            images = images.cpu().numpy()
            for image in images:
                out.write(cv2.cvtColor(image.astype(
                    np.uint8), cv2.COLOR_RGB2BGR))

        out.release()


if __name__ == '__main__':
    generate(path_prefix="../")
