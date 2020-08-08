ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

mkdir -p logs

# -u的作用是：force the stdout and stderr streams to be unbuffered; this option has no effect on stdin;
# 由于Python有缓冲机制，他会在输出内容集结到一个大块后才会输出，这会影响实时输出，-u就是取消缓冲，立即输出计算结果
python -u $ROOT/tools/train_siammask.py \
    --config=config.json -b 8 \
    -j 20 \
    --epochs 20 \
    --log logs/log.txt \
    2>&1 | tee logs/train.log
# --resume snapshot/checkpoint_e9.pth
# https://unix.stackexchange.com/questions/20469/difference-between-21-output-log-and-21-tee-output-log/20472#20472?newreg=04ee7b75f9bc4cff83c2185391b746e5
# 上面的链接解释了 2>&1 | tee logs/train.log的作用，本质上就是把标准输出和标准错误汇集，显示到终端以及记录在文件里
# bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4
