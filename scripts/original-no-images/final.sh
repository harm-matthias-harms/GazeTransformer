for i in {1..3}; do
  # Users
  python ${0%/*}/../../train.py -m original-no-images --delta True -nh 1 -nl 1 -ih 16 --crossEvalType user --crossEvalExclude 1 -f 'final/user1'
  python ${0%/*}/../../train.py -m original-no-images --delta True -nh 1 -nl 1 -ih 16 --crossEvalType user --crossEvalExclude 2 -f 'final/user2'
  python ${0%/*}/../../train.py -m original-no-images --delta True -nh 1 -nl 1 -ih 16 --crossEvalType user --crossEvalExclude 3 -f 'final/user3'
  # Scenes
  python ${0%/*}/../../train.py -m original-no-images --delta True -nh 1 -nl 1 -ih 16 --crossEvalType scene --crossEvalExclude 1 -f 'final/scene1'
  python ${0%/*}/../../train.py -m original-no-images --delta True -nh 1 -nl 1 -ih 16 --crossEvalType scene --crossEvalExclude 2 -f 'final/scene2'
  python ${0%/*}/../../train.py -m original-no-images --delta True -nh 1 -nl 1 -ih 16 --crossEvalType scene --crossEvalExclude 3 -f 'final/scene3'
  python ${0%/*}/../../train.py -m original-no-images --delta True -nh 1 -nl 1 -ih 16 --crossEvalType scene --crossEvalExclude 4 -f 'final/scene4'
done
