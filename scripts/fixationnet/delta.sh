for i in {1..5}; do
  python ${0%/*}/../../train_fixationnet.py -d False --delta True -f 'delta/true'
  python ${0%/*}/../../train_fixationnet.py -d False --delta False -f 'delta/false'
done
