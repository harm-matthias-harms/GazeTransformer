for i in {1..2}; do
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 -f 'all/compress_images/off'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 16 -f 'all/compress_images/16'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 32 -f 'all/compress_images/32'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 64 -f 'all/compress_images/64'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 128 -f 'all/compress_images/128'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 256 -f 'all/compress_images/256'
done
