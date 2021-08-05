# Generates a video output for given parameters
# python eval_video.py -v ./dataset/rawData/Videos/bandicam\ 2020-08-24\ 21-34-24-935.avi -vi 0 -o ./examples/saliency/2020-08-24\ 21-34-24-935.avi -c ./model/checkpoints/GazeTransformer/saliency/nhead/2-1/epoch=4-val_loss=3.37-2-1-delta=False.ckpt
import argparse
import os
import math
import torch
from dataloader.utility import get_sequence_name, get_video_path, get_video_timstamps, get_start_timestamps
from dataloader.dataset import TimeSequenceVideoDataset, FixationnetVideoDataset
from model.transformer import GazeTransformer
from model.fixationnetpl import FixationNetPL
import decord
import cv2
import numpy as np
import progressbar

BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


def main(args):
    model_type = FixationNetPL if "FixationNet" in args.checkpoint else GazeTransformer
    dataset_type = FixationnetVideoDataset if model_type == FixationNetPL else TimeSequenceVideoDataset

    model = model_type.load_from_checkpoint(args.checkpoint).to('cuda').eval()
    vr = decord.VideoReader(args.video)
    out = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*'XVID'), 60, (540, 600))
    dataset = dataset_type(get_video_path(args.video, model.model_type),
                           os.path.join(os.path.dirname(
                               __file__), 'dataset/dataset/FixationNet_150_Images/GazeLabel/', get_sequence_name(args.video)),
                           get_video_timstamps()[args.videoIndex],
                           get_start_timestamps()[args.videoIndex],
                           ignore_images=model.model_type == 'no-images',
                           is_pt=model.model_type in ['resnet', 'dino'],
                           grayscale=model.model_type in ['saliency', 'flatten'])
    for idx in progressbar.progressbar(range(len(dataset))):
        x, y = dataset[idx]
        video_idx = dataset.get_video_idx(idx)
        with torch.no_grad():
            pred = model(x.to('cuda').unsqueeze(0))
        image = vr[video_idx]

        baseline = x[:2] if model_type == FixationNetPL else x[0, :2]

        baseline_coord = CalcScreenCoordinates(baseline)
        pred_coord = CalcScreenCoordinates(pred.flatten(0).to('cpu'))
        ground_truth_coord = CalcScreenCoordinates(y.flatten(0))

        image = np.array(image)
        image = cv2.circle(image, baseline_coord, radius=5,
                           color=BLUE, thickness=-1)
        image = cv2.circle(image, ground_truth_coord, radius=5,
                           color=GREEN, thickness=-1)
        image = cv2.circle(image, pred_coord, radius=5,
                           color=RED, thickness=-1)

        out.write(cv2.cvtColor(image.astype(
            np.uint8), cv2.COLOR_RGB2BGR))

    out.release()


# Adapted from the loss function
def CalcScreenCoordinates(coordinates):
    # the parameters of our Hmd (HTC Vive).
    # Vertical FOV.
    VerticalFov = math.pi*110/180
    # Size of a half screen.
    ScreenWidth = 540  # changed to video width
    ScreenHeight = 600  # changed to video height
    ScreenCenterX = 0.5*ScreenWidth
    ScreenCenterY = 0.5*ScreenHeight
    # the pixel distance between eye and the screen center.
    ScreenDist = 0.5 * ScreenHeight/math.tan(VerticalFov/2)

    x, y = coordinates.split(1)
    # transform the angular coords to screen coords.
    # the X coord.
    x = ScreenDist * torch.tan(math.pi*x / 180) + 0.5*ScreenWidth
    y = ScreenDist * torch.tan(-math.pi*y / 180) + 0.5*ScreenHeight
    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video Generation")

    parser.add_argument('-v', '--video', type=str, help="video path")
    parser.add_argument('-vi', '--videoIndex', type=int,
                        help="index for video base timestamps")
    parser.add_argument('-o', '--output', type=str, help="output path")
    parser.add_argument('-c', '--checkpoint', type=str, help="checkpoint path")

    main(parser.parse_args())
