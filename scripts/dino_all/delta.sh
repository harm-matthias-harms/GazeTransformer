for i in {1..2}; do
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --delta True -f 'all/delta/true'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --delta False -f 'all/delta/false'
done
