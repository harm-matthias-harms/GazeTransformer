for i in {1..2}; do
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -nh 8 -nl 1 --delta True -f 'delta/true'
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -nh 8 -nl 1 --delta False -f 'delta/false'
done