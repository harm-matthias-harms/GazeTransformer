import os
import torch
import torch.nn as nn
import decord
import progressbar
from torchvision.models import resnet50

from utility import get_filenames


def generate(path_prefix=""):
    print("Generate ResNet data")

    INPUT_PATH = os.path.join(path_prefix, '../dataset/dataset/CroppedVideos/')
    OUTPUT_PATH = os.path.join(path_prefix, '../dataset/dataset/ResNetVideos/')
    batch_size = 128

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    decord.bridge.set_bridge("torch")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filenames = get_filenames()

    resnet = nn.Sequential(
        *(list(resnet50(pretrained=True).children())[:-1])).eval().to(device)
    for param in resnet.parameters():
        param.requires_grad = False

    for idx, files in enumerate(filenames):
        print(idx)

        result = torch.Tensor()

        input_filename = INPUT_PATH + files[0].split("bandicam ")[-1]
        output_filename = OUTPUT_PATH + \
            files[0].split("bandicam ")[-1].replace('.avi', '.pt')

        video = decord.VideoReader(input_filename, ctx=decord.gpu())
        video_length = len(video)

        batches = [list(range(video_length))[i * batch_size:(i + 1) * batch_size]
                   for i in range((video_length + batch_size - 1) // batch_size)]
        for batch in progressbar.progressbar(batches):
            images = video.get_batch(batch).float() / 255.0
            images = images.permute(0, 3, 1, 2)
            pred = resnet(images).flatten(1)
            pred = pred.cpu()
            result = torch.cat((result, pred), 0)

        torch.save(result, output_filename)


if __name__ == '__main__':
    generate(path_prefix="../")
