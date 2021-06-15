import torch
import cv2
import numpy as np
import decord
import progressbar
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize, Grayscale

from utility import get_filenames


INPUT_PATH = '../dataset/dataset/CroppedVideos/'
OUTPUT_PATH = '../dataset/dataset/FlattenedVideos/'
batch_size = 1024

decord.bridge.set_bridge("torch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    filenames = get_filenames()

    to_grayscale = Grayscale()
    resize = Resize((32, 32))

    for idx, files in enumerate(filenames):
        print(idx)

        input_filename = INPUT_PATH + files[0].split("bandicam ")[-1]
        output_filename = OUTPUT_PATH + files[0].split("bandicam ")[-1]

        video = decord.VideoReader(input_filename, ctx=decord.gpu())
        video_length = len(video)
        out = cv2.VideoWriter(
            output_filename, cv2.VideoWriter_fourcc(*'XVID'), 60, (32, 32), 0)

        batches = [list(range(video_length))[i * batch_size:(i + 1) * batch_size]
                   for i in range((video_length + batch_size - 1) // batch_size)]
        for batch in progressbar.progressbar(batches):
            images = video.get_batch(batch).float()
            images = images.permute(0, 3, 1, 2)
            images = resize(to_grayscale(images))
            images = images.permute(0, 2, 3, 1)
            images = images.cpu().numpy()
            for image in images:
                out.write(image.astype(np.uint8))

        out.release()
