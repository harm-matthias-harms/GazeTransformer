for i in {1..2}; do
  python ${0%/*}/../../train.py -m dino --useAllImages True -l angular -nh 4 -nl 1 -f 'all/loss/angular'
  python ${0%/*}/../../train.py -m dino --useAllImages True -l mse -nh 4 -nl 1 -f 'all/loss/mse'
done
