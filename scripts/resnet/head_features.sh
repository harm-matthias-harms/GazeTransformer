for i in {1..2}; do
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -nh 8 -nl 1 --imageToFeature True --backboneFeatures 128 -ih 256 -f 'head_features/256'
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -nh 8 -nl 1 --imageToFeature True --backboneFeatures 128 -ih 128 -f 'head_features/128'
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -nh 8 -nl 1 --imageToFeature True --backboneFeatures 128 -ih 64 -f 'head_features/64'
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -nh 8 -nl 1 --imageToFeature True --backboneFeatures 128 -ih 32 -f 'head_features/32'
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -nh 8 -nl 1 --imageToFeature True --backboneFeatures 128 -ih 16 -f 'head_features/16'
done
