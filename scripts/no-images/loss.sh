for i in {1..2}; do
  python ${0%/*}/../../train.py -m no-images -l angular -nh 4 -nl 6 -f 'loss/angular'
  python ${0%/*}/../../train.py -m no-images -l mse -nh 4 -nl 6 -f 'loss/mse'
done