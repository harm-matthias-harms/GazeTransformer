import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from model.fixationnetpl import FixationNetPL


def main(args):
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        verbose=True
    )

    checkpoint_folder = 'Original' if args.originalData else 'Generated'
    checkpoint_path = os.path.join(os.path.dirname(__file__),
                                   'model/checkpoints/FixationNet', checkpoint_folder, args.folder)

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename='{epoch}-{val_loss:.2f}' + f'-delta={args.delta}',
        monitor='val_loss',
        mode='min',
        verbose=True,
        save_top_k=1,
    )

    model = FixationNetPL(batch_size=args.batchSize,
                          num_worker=args.worker, with_original_data=args.originalData, predict_delta=args.delta,
                          cross_eval_type=args.crossEvalType, cross_eval_exclude=args.crossEvalExclude)
    trainer = pl.Trainer(
        gpus=-1, callbacks=[early_stopping_callback, model_checkpoint_callback])

    trainer.fit(model)

    best_model = model.load_from_checkpoint(
        model_checkpoint_callback.best_model_path)

    trainer.test(best_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FixationNet")
    parser.add_argument('-d', '--originalData', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="use original FixationNet dataset (default: True)")
    parser.add_argument('-b', '--batchSize', default=512,
                        type=int, help="the batch size (default: 512)")
    parser.add_argument('-w', '--worker', default=12, type=int,
                        help="the number of workers (default: 12)")
    parser.add_argument('--delta', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="predict the delta and add to last know gaze position (default: True)")
    parser.add_argument('-f', '--folder', default='', type=str,
                        help="specifies a subfolder for the checkpoint (default: '')")
    parser.add_argument('--crossEvalType', default='user', type=str,
                        help="type for the cross evaluation: user | scene (default: user)")
    parser.add_argument('--crossEvalExclude', default=1, type=int,
                        help="the set to exclude: user: 1 | 2 | 3, scene: 1 | 2 | 3 | 4 (default: 1)")

    main(parser.parse_args())
