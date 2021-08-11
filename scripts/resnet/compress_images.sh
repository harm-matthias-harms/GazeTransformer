for i in {1..2}; do
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -nh 8 -nl 1 -f 'compress_images/off'
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -nh 8 -nl 1 --imageToFeature True --backboneFeatures 16 -f 'compress_images/32'
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -nh 8 -nl 1 --imageToFeature True --backboneFeatures 32 -f 'compress_images/32'
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -nh 8 -nl 1 --imageToFeature True --backboneFeatures 64-f 'compress_images/64'
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -nh 8 -nl 1 --imageToFeature True --backboneFeatures 128 -f 'compress_images/128'
  python ${0%/*}/../../train.py -m resnet -b 128 --limitTrainBatches 0.25 -nh 8 -nl 1 --imageToFeature True --backboneFeatures 256 -f 'compress_images/256'
done