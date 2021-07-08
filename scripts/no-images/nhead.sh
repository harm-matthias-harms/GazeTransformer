for i in {1..3}; do
  python ${0%/*}/../../train.py -m no-images -nh 1 -nl 1 -f 'nhead/1-1'
  python ${0%/*}/../../train.py -m no-images -nh 1 -nl 2 -f 'nhead/1-2'
  python ${0%/*}/../../train.py -m no-images -nh 1 -nl 3 -f 'nhead/1-3'
  python ${0%/*}/../../train.py -m no-images -nh 2 -nl 1 -f 'nhead/2-1'
  python ${0%/*}/../../train.py -m no-images -nh 2 -nl 2 -f 'nhead/2-2'
  python ${0%/*}/../../train.py -m no-images -nh 2 -nl 4 -f 'nhead/2-4'
  python ${0%/*}/../../train.py -m no-images -nh 4 -nl 1 -f 'nhead/4-1'
  python ${0%/*}/../../train.py -m no-images -nh 4 -nl 4 -f 'nhead/4-4'
  python ${0%/*}/../../train.py -m no-images -nh 4 -nl 6 -f 'nhead/4-6'
  python ${0%/*}/../../train.py -m no-images -nh 6 -nl 1 -f 'nhead/6-1'
  python ${0%/*}/../../train.py -m no-images -nh 6 -nl 4 -f 'nhead/6-4'
  python ${0%/*}/../../train.py -m no-images -nh 6 -nl 6 -f 'nhead/6-6'
  python ${0%/*}/../../train.py -m no-images -nh 8 -nl 1 -f 'nhead/8-1'
  python ${0%/*}/../../train.py -m no-images -nh 8 -nl 4 -f 'nhead/8-4'
  python ${0%/*}/../../train.py -m no-images -nh 8 -nl 6 -f 'nhead/8-6'
done