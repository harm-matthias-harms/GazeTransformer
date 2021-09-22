for i in {1..3}; do
  # Users
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 64 -ih 128 --crossEvalType user --crossEvalExclude 1 -f 'all/final/user1'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 64 -ih 128 --crossEvalType user --crossEvalExclude 2 -f 'all/final/user2'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 64 -ih 128 --crossEvalType user --crossEvalExclude 3 -f 'all/final/user3'
  # Scenes
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 64 -ih 128 --crossEvalType scene --crossEvalExclude 1 -f 'all/final/scene1'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 64 -ih 128 --crossEvalType scene --crossEvalExclude 2 -f 'all/final/scene2'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 64 -ih 128 --crossEvalType scene --crossEvalExclude 3 -f 'all/final/scene3'
  python ${0%/*}/../../train.py -m dino --useAllImages True -nh 4 -nl 1 --imageToFeature True --backboneFeatures 64 -ih 128 --crossEvalType scene --crossEvalExclude 4 -f 'all/final/scene4'
done
