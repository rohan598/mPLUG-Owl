#!/bin/bash
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

len_visible_devices=${#CUDA_VISIBLE_DEVICES}
num_devices=$((len_visible_devices/2+1))

if [ $MASTER_ADDR ];then
	echo $MASTER_ADDR
    echo $MASTER_PORT
    echo $WORLD_SIZE
    echo $RANK
else
	MASTER_ADDR=127.0.0.1
    MASTER_PORT=2$(($RANDOM % 10))$(($RANDOM % 10))15
    WORLD_SIZE=1
    RANK=0
fi

DISTRIBUTED_ARGS="--nproc_per_node ${num_devices} \
                  --nnodes ${WORLD_SIZE} \
                  --node_rank ${RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

EXP_NAME=stage_1
SAVE_NAME="${EXP_NAME}_ft_grad_ckpt_dataset_lr_$3-$4-${DATETIME}"

SAVE_PATH="/local1/rwadhawan/document_understanding/results/mplug_owl/${SAVE_NAME}"  

max_length=512
micro_batch_size=128
# global_batch_size=256
gradient_accumulation_steps=1

# train_iters = total_data * train_epochs // global_batch_size
# 361481 * 3 / 256 = 4236
train_epochs=$5
train_iters=4236

lr_warmup_iters=10

eval_iter=100
save_interval=500

mkdir -p ${SAVE_PATH}

options=" \
	--pretrained-ckpt $1 \
	--seq-length ${max_length} \
	--micro-batch-size ${micro_batch_size} \
	--num-training-steps ${train_iters} \
    --train-epochs ${train_epochs} \
	--num-warmup-steps ${lr_warmup_iters} \
	--gradient-accumulation-steps ${gradient_accumulation_steps} \
	--lr $4 \
	--min-lr 1e-6 \
	--eval-iters ${eval_iter} \
    --save-interval ${save_interval} \
	--save-path ${SAVE_PATH} \
    --clip-grad 1.0 \
	--weight-decay 0.0001 \
	--adam-beta1 0.9 \
	--adam-beta2 0.999 \
	--num-workers 128 \
	--gradient-checkpointing \
	--bf16"

multimodal_options=" \
	--mm-config configs/stage1.yaml 
    "

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./pipeline/train.py $@ ${options} ${multimodal_options} 2>&1 | tee ${SAVE_PATH}/train.log 