for i in {1..2}; do
  python ${0%/*}/../../train.py -m saliency -nh 2 -nl 1 --delta True -f 'delta/true'
  python ${0%/*}/../../train.py -m saliency -nh 2 -nl 1 --delta False -f 'delta/false'
done