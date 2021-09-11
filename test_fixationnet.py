import argparse
import pytorch_lightning as pl

from model.fixationnetpl import FixationNetPL


def main(args):
    model = FixationNetPL.load_from_checkpoint(args.path)
    trainer = pl.Trainer(gpus=-1)

    trainer.test(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FixationNet")
    parser.add_argument('-p', '--path', default='', type=str,
                        help="checkpoint path")

    main(parser.parse_args())
