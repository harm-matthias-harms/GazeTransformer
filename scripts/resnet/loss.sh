for i in {1..2}; do
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -l angular -nh 8 -nl 1 -f 'loss/angular'
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -l mse -nh 8 -nl 1 -f 'loss/mse'
done