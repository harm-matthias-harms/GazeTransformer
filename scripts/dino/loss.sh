for i in {1..2}; do
  python ${0%/*}/../../train.py -m dino -l angular -nh 4 -nl 1 -f 'loss/angular'
  python ${0%/*}/../../train.py -m dino -l mse -nh 4 -nl 1 -f 'loss/mse'
done
