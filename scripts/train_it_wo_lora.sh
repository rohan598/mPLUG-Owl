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

EXP_NAME=sft_v0.1
SAVE_NAME="sft_v0.1_ft_grad_ckpt_dataset_lr_$1-$2-${DATETIME}"

SAVE_PATH="/local1/rwadhawan/document_understanding/results/mplug_owl/${SAVE_NAME}/"  
# "./output/${SAVE_NAME}/"
max_length=2048
micro_batch_size=2
global_batch_size=32 # global_batch_size = micro_batch_size * num_gpu * grad_acc_step
gradient_accumulation_steps=4

# train_iters = total_data * train_epochs // global_batch_size
# 361481 * 3 / 256 = 4236
# 16370*10/256 = 
# 1.18% is warm up
train_epochs=10
train_iters=5116

lr_warmup_iters=92
lr_decay_iters=`expr $train_iters - $lr_warmup_iters`

eval_iter=256
# eval_interval=4
save_interval=512

mkdir -p ${SAVE_PATH}
options=" \
	--pretrained-ckpt /local1/rwadhawan/document_understanding/models/mplug-owl-llama-7b-ft \
	--seq-length ${max_length} \
	--micro-batch-size ${micro_batch_size} \
    --train-epochs ${train_epochs} \
	--num-warmup-steps ${lr_warmup_iters} \
	--num-training-steps ${train_iters} \
	--gradient-accumulation-steps ${gradient_accumulation_steps} \
	--lr $2 \
	--min-lr 1e-6 \
	--eval-iters ${eval_iter} \
    --save-interval ${save_interval} \
	--save-path ${SAVE_PATH} \
    --clip-grad 1.0 \
	--weight-decay 0.0001 \
	--adam-beta1 0.9 \
	--adam-beta2 0.999 \
	--num-workers 32 \
	--gradient-checkpointing \
	--bf16 "

multimodal_options=" \
	--mm-config configs/v0.yaml 
    "

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./pipeline/train.py $@ ${options} ${multimodal_options} 2>&1 | tee ${SAVE_PATH}/train.log 