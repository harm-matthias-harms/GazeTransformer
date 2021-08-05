for i in {1..2}; do
  python ${0%/*}/../../train.py -m saliency -l angular -nh 2 -nl 1 -f 'loss/angular'
  python ${0%/*}/../../train.py -m saliency -l mse -nh 2 -nl 1 -f 'loss/mse'
done