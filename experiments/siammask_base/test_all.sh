show_help() {
cat << EOF
Usage: 
    ${0##*/} [-h/--help] [-s/--start] [-e/--end] [-d/--dataset] [-m/--model]  [-g/--gpu]
    e.g.
        bash ${0##*/} -s 1 -e 20 -d VOT2018 -g 4 # for test models
        bash ${0##*/} -m snapshot/checkpoint_e10.pth -n 8 -g 4 # for tune models
EOF
}

ROOT=`git rev-parse --show-toplevel`
source activate siammask
export PYTHONPATH=$ROOT:$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

dataset=VOT2018
NUM=4
START=1
END=20
GPU=0

# $#表示输入的命令行参数的总个数；即便我们不输入命令行参数，也会有一个文件名参数，用$0可以获取到
# 循环执行，直到把所有的命令行参数都识别并进行对应值的赋值才结束，很好的一段代码
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit
            ;;
        -d|--dataset)
            dataset=$2  # 在这里$2是就是$1的值
            shift 2
            ;;
        -n|--num)
            NUM=$2
            shift 2
            ;;
        -s|--start)
            START=$2
            shift 2
            ;;
        -e|--end)
            END=$2
            shift 2
            ;;
        -m|--model)
            model=$2
            shift 2
            ;;
        -g|--gpu)
            GPU=$2
            shift 2
            ;;
        *)
            echo invalid arg [$1]
            show_help
            exit 1  # 退出码是非零，意味着程序执行不成功
            ;;
    esac
done

# 如果以上命令以非零状态退出，则本脚本立即退出
set -e

if [ -z "$model" ]; then # 判断该字符串长度是否为零
    echo test snapshot $START ~ $END on dataset $dataset with $GPU gpus.
    for i in $(seq $START $END)
    do 
        bash test.sh snapshot/checkpoint_e$i.pth $dataset $(($i % $GPU)) & # &表示后台运行该条指令
    done
    wait # 等待子进程的结束

    python $ROOT/tools/eval.py --dataset $dataset --num 20 --tracker_prefix C --result_dir ./test/$dataset 2>&1 | tee logs/eval_test_$dataset.log
else
    echo tuning $model on dataset $dataset with $NUM jobs in $GPU gpus.
    for i in $(seq 1 $NUM)
    do 
        bash tune.sh $model $dataset $(($i % $GPU)) & 
    done
    wait
    rm finish.flag

    python $ROOT/tools/eval.py --dataset $dataset --num 20 --tracker_prefix C  --result_dir ./result/$dataset 2>&1 | tee logs/eval_tune_$dataset.log
fi
