from generation import features, centercropped_videos, saliency_video, flattened_videos, patch_videos, resnet_data, dino_data

if __name__ == '__main__':
  features.generate()
  centercropped_videos.generate()
  saliency_video.generate()
  flattened_videos.generate()
  patch_videos.generate()
  resnet_data.generate()
  dino_data.generate()
