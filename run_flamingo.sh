#!/usr/bin/env bash

export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

sudo chmod 777 -R /mnt/bn/robotics-data-lxh-lq/logs
sudo chmod 777 -R /mnt/bn/robotics-data-lxh-lq/checkpoints
sudo chmod 777 -R /mnt/bn/robotics-data-lxh-lq/cache

source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt

sudo apt-get update
sudo apt install -y screen
sudo apt-get -y install libosmesa6-dev
sudo apt-get -y install patchelf

#pip3 install torch==2.1.0 torchvision torchaudio -i "https://bytedpypi.byted.org/simple"
pip3 install lightning==2.1.0
pip3 install deepspeed==0.12.3
pip3 install transformations
pip3 install torchvision torchaudio thop

pip3 install transformers==4.33.3 -i "https://bytedpypi.byted.org/simple"
pip3 install packaging==21.3 -i "https://bytedpypi.byted.org/simple"
pip3 install tensorboard -i "https://bytedpypi.byted.org/simple"

pip3 install ftfy regex tqdm -i "https://bytedpypi.byted.org/simple"
pip3 install matplotlib decord pandas -i "https://bytedpypi.byted.org/simple"

mkdir -p ~/.cache
cp -r /mnt/bn/robotics-data-hl/lxh/RoboFlamingoV2/.cache/clip ~/.cache/


CUR_DIR=$(cd $(dirname $0); pwd)
# cd $CUR_DIR/../

set -x
export PYTHONUNBUFFERED=1
export NCCL_IB_GID_INDEX=3
export BYTED_TORCH_FX=O1
export BYTED_TORCH_BYTECCL=${BYTED_TORCH_BYTECCL:-O1}

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0
export BYTEPS_DDP_FIND_UNUSED_PARAMS=${BYTEPS_DDP_FIND_UNUSED_PARAMS:-0}

GPUS_PER_NODE=$ARNOLD_WORKER_GPU
# GPUS_PER_NODE=1
MASTER_ADDR=$METIS_WORKER_0_HOST":"$METIS_WORKER_0_PORT
NNODES=$ARNOLD_WORKER_NUM
JOB_ID=107

# convert deepspeed checkpoint first
if [ $ARNOLD_ID == "0" ]; then
  echo "---------- Converting deepspeed checkpoint to fp32. ----------"
  python3 tools/convert_deepspeed_to_fp32.py ${@:1}
fi

# run training
echo torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank ${ARNOLD_ID:-0} \
    --rdzv_endpoint $MASTER_ADDR \
    --rdzv_id $JOB_ID \
    --rdzv_backend c10d \
    scripts/main_flamingo.py \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $NNODES

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank ${ARNOLD_ID:-0} \
    --rdzv_endpoint $MASTER_ADDR \
    --rdzv_id $JOB_ID \
    --rdzv_backend c10d \
    scripts/main_flamingo.py \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $NNODES

