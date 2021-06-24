import torch
import decord
import progressbar

from utility import get_filenames


INPUT_PATH = '../dataset/dataset/CroppedVideos/'
OUTPUT_PATH = '../dataset/dataset/DinoVideos/'
batch_size = 128

decord.bridge.set_bridge("torch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    filenames = get_filenames()

    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').eval().to(device)
    for param in dino.parameters():
        param.requires_grad = False

    for idx, files in enumerate(filenames):
        print(idx)

        result = torch.Tensor()

        input_filename = INPUT_PATH + files[0].split("bandicam ")[-1]
        output_filename = OUTPUT_PATH + files[0].split("bandicam ")[-1].replace('.avi', '.pt')

        video = decord.VideoReader(input_filename, ctx=decord.gpu())
        video_length = len(video)

        batches = [list(range(video_length))[i * batch_size:(i + 1) * batch_size]
                for i in range((video_length + batch_size - 1) // batch_size)]
        for batch in progressbar.progressbar(batches):
            images = video.get_batch(batch).float() / 255.0
            images = images.permute(0, 3, 1, 2)
            pred = dino(images).flatten(1)
            pred = pred.cpu()
            result = torch.cat((result, pred), 0)

        print(result.shape)
        torch.save(result, output_filename)
