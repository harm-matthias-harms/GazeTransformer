import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from model.transformer import GazeTransformer


def main(args):
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        verbose=True
    )

    checkpoint_path = os.path.join(os.path.dirname(__file__),
                                   'model/checkpoints/GazeTransformer', args.model, args.folder)

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename='{epoch}-{val_loss:.2f}' +
        f'-{args.nheads}-{args.numLayers}-delta={args.delta}',
        monitor='val_loss',
        mode='min',
        verbose=True,
        save_top_k=1,
    )
    resume_from_checkpoint = None
    if args.resume:
        resume_from_checkpoint = os.path.join(checkpoint_path, args.resume)

    model = GazeTransformer(model_type=args.model, loss=args.loss, predict_delta=args.delta, nhead=args.nheads, num_layers=args.numLayers,
                            image_to_features=args.imageToFeature, backbone_features=args.backboneFeatures,
                            inner_head_features=args.innerHeadFeatures, learning_rate=args.learningRate,
                            batch_size=args.batchSize, num_worker=args.worker,
                            cross_eval_type=args.crossEvalType, cross_eval_exclude=args.crossEvalExclude, use_all_images=args.useAllImages)
    trainer = pl.Trainer(
        gpus=-1, callbacks=[early_stopping_callback, model_checkpoint_callback],
        limit_train_batches=args.limitTrainBatches, resume_from_checkpoint=resume_from_checkpoint)

    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GazeTransformer")
    parser.add_argument('-m', '--model', default='original', type=str,
                        help="the model of the network: original | original-no-images | no-images | saliency | flatten | patches | resnet | dino (default: original)")
    parser.add_argument('-l', '--loss', default='angular', type=str,
                        help="the loss function: angular | mse (default: angular)")
    parser.add_argument('--crossEvalType', default='user', type=str,
                        help="type for the cross evaluation: user | scene (default: user)")
    parser.add_argument('--crossEvalExclude', default=1, type=int,
                        help="the set to exclude: user: 1 | 2 | 3, scene: 1 | 2 | 3 | 4 (default: 1)")
    parser.add_argument('--useAllImages', default=False, type=bool,
                        help="use frames closest to data, instead of -400ms and -200ms (default: False)")
    parser.add_argument('-nh', '--nheads', default=8, type=int,
                        help="nhead of the transformer (default: 8)")
    parser.add_argument('-nl', '--numLayers', default=6, type=int,
                        help="num_layers of the transformer (default: 6)")
    parser.add_argument('-ih', '--innerHeadFeatures', default=128,
                        type=int, help="number of inner features in the head (default: 8)")
    parser.add_argument('--backboneFeatures', default=128, type=int,
                        help="compress preprocessed image to this number of features (requires --imageToFeature True, default: 128)")
    parser.add_argument('--delta', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="predict the delta and add to last know gaze position (default: True)")
    parser.add_argument('--imageToFeature', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="compress the preprocessed image to a set of features (default: False)")
    parser.add_argument('-lr', '--learningRate', default=0.001,
                        type=float, help="the learning rate (default: 0.001)")
    parser.add_argument('-b', '--batchSize', default=256,
                        type=int, help="the batch size (default: 256)")
    parser.add_argument('-w', '--worker', default=12, type=int,
                        help="the number of workers (default: 12)")
    parser.add_argument('--limitTrainBatches', default=1.0, type=float,
                        help="limit the number of train batches in an epoch (default: 1.0)")
    parser.add_argument('-f', '--folder', default='', type=str,
                        help="specifies a subfolder for the checkpoint (default: '')")
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help="file name of checkpoint to resume training (default: None)")

    main(parser.parse_args())
