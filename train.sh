python train_fixationnet.py -d True
python train_fixationnet.py -d False
python train.py -m original
python train.py -m original-no-images -b 512
python train.py -m no-images -b 512
python train.py -m saliency
python train.py -m saliency -nh 1 -nl 1
python train.py -m flatten --limitTrainBatches 0.25
python train.py -m patches --limitTrainBatches 0.25
python train.py -m resnet -b 128 --limitTrainBatches 0.25
python train.py -m dino
