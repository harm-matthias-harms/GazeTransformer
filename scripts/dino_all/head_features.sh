for i in {1..2}; do
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 64 -ih 256 -f 'all/head_features/256'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 64 -ih 128 -f 'all/head_features/128'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 64 -ih 64 -f 'all/head_features/64'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 64 -ih 32 -f 'all/head_features/32'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 64 -ih 16 -f 'all/head_features/16'
done
