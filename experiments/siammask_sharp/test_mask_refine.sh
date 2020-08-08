if [ -z "$4" ]
  then
    echo "Need input parameter!"
    echo "Usage: bash `basename "$0"` \$CONFIG \$MODEL \$DATASET \$GPUID"
    exit
fi

ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

mkdir -p logs

config=$1
model=$2
dataset=$3
gpu=$4
# config_davis.json SiamMask_DAVIS.pth ytb_vos 0
CUDA_VISIBLE_DEVICES=$gpu python -u $ROOT/tools/test.py \
    --config $config \config_davis.json
    --resume $model \SiamMask_DAVIS.pth
    --mask --refine \
    --dataset $dataset ytb_vos 2>&1 | tee logs/test_$dataset.log

