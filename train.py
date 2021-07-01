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

    checkpoint_path = f'./model/checkpoints/GazeTransformer/{args.model}/'

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename='{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        verbose=True,
        save_top_k=1,
    )
    resume_from_checkpoint = None
    if args.resume:
        resume_from_checkpoint = os.path.join(checkpoint_path, args.resume)

    model = GazeTransformer(model_type=args.model, loss=args.loss, nhead=args.nheads, num_layers=args.numLayers, learning_rate=args.learningRate,
                            batch_size=args.batchSize, num_worker=args.worker)
    trainer = pl.Trainer(
        gpus=-1, callbacks=[early_stopping_callback, model_checkpoint_callback],
        limit_train_batches=args.limitTrainBatches, resume_from_checkpoint=resume_from_checkpoint)

    trainer.fit(model)

    best_model = model.load_from_checkpoint(
        model_checkpoint_callback.best_model_path)

    trainer.test(best_model)

    # possible improvements, when we need to run with small batchsize
    # trainer = Trainer(accumulate_grad_batches=1)
    # trainer = Trainer(auto_scale_batch_size=None|'power'|'binsearch'); trainer.tune(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GazeTransformer")
    parser.add_argument('-m', '--model', default='original', type=str,
                        help="the model of the network: original | original-no-images | no-images | saliency | flatten | patches | resnet | dino (default: original)")
    parser.add_argument('-l', '--loss', default='angular', type=str,
                        help="the loss function: angular | mse (default: angular)")
    parser.add_argument('-nh', '--nheads', default=8, type=int,
                        help="nhead of the transformer (default: 8)")
    parser.add_argument('-nl', '--numLayers', default=6, type=int,
                        help="num_layers of the transformer (default: 6)")
    parser.add_argument('-lr', '--learningRate', default=0.001,
                        type=float, help="the learning rate (default: 0.001)")
    parser.add_argument('-b', '--batchSize', default=256,
                        type=int, help="the batch size (default: 256)")
    parser.add_argument('-w', '--worker', default=12, type=int,
                        help="the number of workers (default: 12)")
    parser.add_argument('--limitTrainBatches', default=1.0, type=float,
                        help="limit the number of train batches in an epoch (default: 1.0)")
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help="file name of checkpoint to resume training (default: None)")

    main(parser.parse_args())
