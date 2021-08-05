for i in {1..2}; do
  python ${0%/*}/../../train.py -m patches -l angular -nh 6 -nl 1 -f 'loss/angular'
  python ${0%/*}/../../train.py -m patches -l mse -nh 6 -nl 1 -f 'loss/mse'
done