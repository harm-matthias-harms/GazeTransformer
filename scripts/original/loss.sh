for i in {1..2}; do
  python ${0%/*}/../../train.py -m original -l angular -nh 1 -nl 1 --delta True -f 'loss/angular'
  python ${0%/*}/../../train.py -m original -l mse -nh 1 -nl 1 --delta True -f 'loss/mse'
done