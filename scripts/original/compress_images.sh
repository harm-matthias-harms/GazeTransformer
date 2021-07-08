for i in {1..3}; do
  echo "not implemented: run nhead first"
  # python ${0%/*}/../../train.py -m original --delta True -nh 1 -nl 1 -f 'compress_images/off'
  # python ${0%/*}/../../train.py -m original --delta True -nh 1 -nl 1 --imageToFeature True --backboneFeatures 16 -f 'compress_images/32'
  # python ${0%/*}/../../train.py -m original --delta True -nh 1 -nl 1 --imageToFeature True --backboneFeatures 32 -f 'compress_images/32'
  # python ${0%/*}/../../train.py -m original --delta True -nh 1 -nl 1 --imageToFeature True --backboneFeatures 64-f 'compress_images/64'
  # python ${0%/*}/../../train.py -m original --delta True -nh 1 -nl 1 --imageToFeature True --backboneFeatures 128 -f 'compress_images/128'
  # python ${0%/*}/../../train.py -m original --delta True -nh 1 -nl 1 --imageToFeature True --backboneFeatures 256 -f 'compress_images/256'
done