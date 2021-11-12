# GazeTransformer: Egocentric Gaze Forecasting with Transformers
This repository contains the source code for my [Masterthesis](https://drive.google.com/file/d/1HMXRWQW_EImSi00U5RxdOEqee1yyxS6v/view?usp=sharing). The checkpoints used for the comparison in the evaluation can be found in this [folder](https://drive.google.com/drive/folders/10Xq1S9SJA7XwYjRe-4d8AB0d0pNnoMag?usp=sharing).


## Abstract

>During the last decade, convolutional neural networks have become the state-of-the-art approach for many computer vision problems. Recent publications in natural language processing boost the state-of-the-art performance for sequence-to-sequence models significantly by applying a novel Transformer architecture based on self-attention. Recently, researchers applied Transformers to computer vision tasks, such as object detection, image completion, and saliency prediction, competing with the state-of-the-art. <br/><br/>
Human gaze information in virtual reality is essential for many applications, such as gaze-contingent rendering or eye movement-based interactions. By defining gaze forecasting as a time-series prediction problem, we propose a novel Transformer-based architecture, called GazeTransformer, forecasting users' gaze in dynamic virtual reality environments. Based on provided raw data, we generated an unfiltered dataset containing all gaze behavior and compared GazeTransformer to two state-of-the-art methods for gaze forecasting. Further, we evaluated different image encodings, enabling us to combine data from different sources in virtual reality, building a time-dependent sequence. As a result, GazeTransformer improved the baseline, using the current gaze for the prediction, by 8.2\% (from a mean error of 3.67° to 3.37°). Further, GazeTransformer beat the prior state-of-the-art significantly (3.37° vs. 7.04° mean error), tested on the generated dataset containing all gaze behavior.


## Usage

### Requirements
The requirements are listed in the requirements.txt.

### Dataset
Step 1: Download the dataset from the FixationNet project homepage: https://cranehzm.github.io/FixationNet

Step 2: Place the dataset in the `./dataset` folder. E.g. `./dataset/rawData` and `./dataset/dataset`

Step 3: Generate our unfiltered dataset. Either run `./dataloader/generate.py` or run specific scripts `./dataloader/generation/`.

### Training and Evaluation
Run the `train.*.py` and `test.*.py` scripts. The models can be found in the `./model` folder. The checkpoints are stored in the same folder. `./scripts` contains all scripts used during the evaluation of the thesis. `./eval_video.py` generates videos for qualitative analysis.