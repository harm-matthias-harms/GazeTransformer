for i in {1..2}; do
  python ${0%/*}/../../train.py -m no-images -nh 4 -nl 6 -ih 256 -f 'head_features/256'
  python ${0%/*}/../../train.py -m no-images -nh 4 -nl 6 -ih 128 -f 'head_features/128'
  python ${0%/*}/../../train.py -m no-images -nh 4 -nl 6 -ih 64 -f 'head_features/64'
  python ${0%/*}/../../train.py -m no-images -nh 4 -nl 6 -ih 32 -f 'head_features/32'
  python ${0%/*}/../../train.py -m no-images -nh 4 -nl 6 -ih 16 -f 'head_features/16'
done