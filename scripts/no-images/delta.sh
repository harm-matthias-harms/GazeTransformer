for i in {1..2}; do
  python ${0%/*}/../../train.py -m no-images -nh 4 -nl 6 --delta True -f 'delta/true'
  python ${0%/*}/../../train.py -m no-images -nh 4 -nl 6 --delta False -f 'delta/false'
done