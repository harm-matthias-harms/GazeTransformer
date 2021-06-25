import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from model.fixationnetpl import FixationNetPL


def main(args):
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        verbose=True
    )

    checkpoint_folder = 'Original' if args.original_data else 'Generated'
    model_checkpoint_callback = ModelCheckpoint(
        dirpath=f'./model/checkpoints/FixationNet/{checkpoint_folder}/',
        filename='{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        verbose=True,
        save_top_k=1,
    )

    model = FixationNetPL(batch_size=args.batchSize,
                          num_worker=args.worker, with_original_data=args)
    trainer = pl.Trainer(
        gpus=-1, callbacks=[early_stopping_callback, model_checkpoint_callback])

    trainer.fit(model)

    best_model = model.load_from_checkpoint(
        model_checkpoint_callback.best_model_path)

    trainer.test(best_model)

    # possible improvements, when we need to run with small batchsize
    # trainer = Trainer(accumulate_grad_batches=1)
    # trainer = Trainer(auto_scale_batch_size=None|'power'|'binsearch'); trainer.tune(model)
    # trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GazeTransformer")
    parser.add_argument('-d', '--original-data', default=True, type=bool,
                        help="use original FixationNet dataset (default: True)")
    parser.add_argument('-b', '--batchSize', default=256,
                        type=int, help="the batch size (default: 256)")
    parser.add_argument('-w', '--worker', default=12, type=int,
                        help="the number of workers (default: 12)")

    main(parser.parse_args())
